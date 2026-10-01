"""Session-persistent kernels for REMOTE terminal backends (docker/ssh/modal).

Remote backends offer one primitive — ``env.execute(cmd)``, run-to-completion
— so the three things the local kernel gets from owning a child are rebuilt:
a detached runner (``nohup ... &``, PID recorded, ``kill -0`` probed per cell);
a file-based CELL protocol in the kernel dir (``cell_req_NNNNNN.json`` /
``cell_res_NNNNNN.json``), sibling to the unchanged file-based TOOL-RPC protocol
(req_/res_) whose host-side ``_rpc_poll_loop`` starts per cell with the calling
thread's context (= per-cell tool authority); and death detection — a failed
liveness probe reads as *kernel died: state lost* and the next call respawns,
never a hung poll (every wait is bounded by the cell timeout).

Same invariants as local: owner = approval session key with the ``::child::``
qualifier (one resolver in tools.code_kernel), same generated tool stubs, same
output post-processing in the caller. ``reset=true`` kills and respawns. Spawn
failure fails OPEN to the per-call path with a note.
"""
from __future__ import annotations

import atexit
import json
import logging
import secrets
import shlex
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.code_kernel import RUNNER_CELL_SOURCE, KernelRegistry

logger = logging.getLogger(__name__)

# Host poll interval for a cell result file; each poll is one env.execute round-trip
# (0.1-0.4s on ssh/docker), so this is a floor, not a rate.
_CELL_POLL_INTERVAL = 0.5

# Remote persistent and per-call Python targets require a private POSIX process
# group before any receipt, user code, or RPC can run. SIGTERM is catchable, so
# verified retirement depends on this group boundary.
REMOTE_POSIX_SESSION_SOURCE = '''\
import os
if os.name != "posix" or not hasattr(os, "setsid") or not hasattr(os, "getpgid"):
    raise RuntimeError("Remote POSIX process-group capability is required (setsid/getpgid)")
try:
    os.setsid()
    if os.getpgid(0) != os.getpid():
        raise OSError("setsid did not establish the runner process group")
except (OSError, TypeError, AttributeError) as exc:
    raise RuntimeError("Remote POSIX process-group capability failed") from exc
'''

# The remote runner: a forever-loop that polls for cell request files, execs them in one
# persistent namespace, writes response files. Pure files + stdlib only (transport-agnostic);
# cells and tool-RPC share the kernel dir under distinct prefixes.
REMOTE_KERNEL_RUNNER_SOURCE = REMOTE_POSIX_SESSION_SOURCE + '''\
"""Auto-generated Hermes REMOTE session-kernel runner (file cell protocol)."""
import contextlib
import contextvars
import threading
import hermes_tools
import io
import json
import os
import sys
import time
import traceback

KDIR = os.environ["HERMES_KERNEL_DIR"]
# The launcher owns this one-shot directory before dispatch; publishing a receipt
# from the runner lets the host recover the exact launch after a lost reply.
os.makedirs(os.path.join(KDIR, "launch"), exist_ok=True)
with open(os.path.join(KDIR, "launch", "runner-pid.tmp"), "w", encoding="ascii") as f:
    f.write(str(os.getpid()))
os.replace(os.path.join(KDIR, "launch", "runner-pid.tmp"),
           os.path.join(KDIR, "launch", "pid"))
CELLS = os.path.join(KDIR, "cells")
_CAPTURE_LIMIT = {capture_limit}
IDLE_EXIT_SECONDS = {idle_exit}

{cell_source}

# Imported wrappers are reusable, but the CALLER must belong to a live cell.
# Threads retain the cell that created them, never the main runner's next cell.
_rpc_authority = contextvars.ContextVar("hermes_remote_cell", default=None)
hermes_tools._rpc_context = _rpc_authority.get
_thread_start = threading.Thread.start


def _start_thread(thread, *args, **kwargs):
    ctx = contextvars.copy_context()
    run = thread.run
    thread.run = lambda: ctx.run(run)
    return _thread_start(thread, *args, **kwargs)


threading.Thread.start = _start_thread


def main():
    execution_count = 0
    last_activity = time.time()
    while True:
        pending = sorted(
            f for f in os.listdir(CELLS)
            if f.startswith("cell_req_") and f.endswith(".json")
        )
        if not pending:
            if time.time() - last_activity > IDLE_EXIT_SECONDS:
                return  # self-reap: nobody is talking to us anymore
            time.sleep(0.2)
            continue
        for name in pending:
            req_path = os.path.join(CELLS, name)
            try:
                with open(req_path, "r", encoding="utf-8") as f:
                    request = json.load(f)
            except Exception:
                # Partially-written request (ship in progress): retry next tick.
                continue
            os.remove(req_path)
            last_activity = time.time()
            execution_count += 1
            authority = [request["rpc_dir"], request["rpc_token"], True]
            context_token = _rpc_authority.set(authority)
            try:
                payload, _ = run_cell(request, execution_count)
            finally:
                authority[2] = False
                _rpc_authority.reset(context_token)
            res_name = name.replace("cell_req_", "cell_res_")
            tmp = os.path.join(CELLS, res_name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, os.path.join(CELLS, res_name))
            if payload["status"] == "exit":
                return


if __name__ == "__main__":
    main()
'''


def _sh(env, cmd: str, timeout: int = 15) -> str:
    """Run *cmd* on the remote from ``/`` and return its output text."""
    result = env.execute(cmd, cwd="/", timeout=timeout)
    return (result.get("output", "") if isinstance(result, dict) else "") or ""


class _CellAuthority:
    """Order host-tool admission against retirement for one cell."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.active = 0

    def retire(self) -> None:
        # Do not wait for an admitted, non-cancellable host operation. Mark retirement
        # immediately; dispatch admission checks and increments ``active`` under the
        # same lock, so any call not already admitted is refused from this point.
        self.stop.set()

    def dispatch(self, name: str, args: Dict[str, Any]):
        from model_tools import handle_function_call
        from tools.registry import tool_error
        with self.lock:
            if self.stop.is_set():
                return tool_error("Remote cell authority retired")
            self.active += 1
        try:
            return handle_function_call(name, args, task_id=self.task_id)
        finally:
            with self.lock:
                self.active -= 1

    def is_draining(self) -> bool:
        with self.lock:
            return self.active > 0


@dataclass
class RemoteKernel:
    """Host-side record of one detached remote kernel process."""

    env: Any
    env_identity: int
    env_type: str
    kernel_dir: str
    pid: str
    rpc_token: str
    owner: str
    last_used: float = field(default_factory=time.monotonic)
    execution_count: int = 0
    cell_seq: int = 0
    # Cells currently running on this kernel. Reap/evict skip attached
    # kernels: killing one mid-cell tears the runner out from under a live
    # poll loop (same guard as tools.code_kernel, hermes-agent#101861).
    attached: int = 0
    # Owned by a live delegate_task child: exempt from LRU eviction (the child's teardown disposes it).
    pinned: bool = False
    #: Marked BEFORE the kill transport runs, so a concurrent or subsequent
    #: caller sees this record as retiring and cannot reuse the slot while
    #: retirement is still unproven.
    retiring: bool = False

    def sh(self, cmd: str, timeout: int = 15) -> str:
        return _sh(self.env, cmd, timeout)

    def is_alive(self) -> bool:
        """Bounded liveness probe: kill -0 through the transport. Any transport
        failure counts as dead — a dropped ssh connection and a dead runner are
        indistinguishable from here, and both have the same correct answer (respawn).

        NOTE: that collapse is right for RESPAWN and wrong for RETIREMENT. A
        kernel we cannot reach may still be running with the retired cell's
        authority, so retirement must prove death separately — see kill_verified.
        """
        try:
            return "ALIVE" in self.sh(f"kill -0 {shlex.quote(self.pid)} 2>/dev/null && echo ALIVE")
        except Exception:
            return False

    def kill_verified(self) -> bool:
        """Hard retirement, verified on the remote: True only on PROVEN death.

        A launch without a PID is first fenced through its one-shot launch directory:
        if the fence wins, no delayed launch can start; otherwise the runner's receipt
        identifies the exact process group that must be killed. Transport ambiguity is
        never evidence of death.
        """
        if not self.pid:
            launch = shlex.quote(self.kernel_dir + "/launch")
            try:
                receipt = self.sh(
                    f"if mkdir {launch} 2>/dev/null; then touch {launch}/fenced; fi; "
                    f"if test -f {launch}/fenced; then echo FENCED; else cat {launch}/pid; fi",
                    timeout=15,
                ).strip()
            except Exception:
                return False
            if receipt == "FENCED":
                return True
            if not (receipt.isascii() and receipt.isdigit() and int(receipt) > 1):
                return False
            self.pid = receipt

        q_pid = shlex.quote(self.pid)
        try:
            self.sh(
                f"kill -KILL -- -{q_pid} 2>/dev/null; "
                f"pkill -KILL -P {q_pid} 2>/dev/null; "
                f"kill -KILL {q_pid} 2>/dev/null; true",
                timeout=15,
            )
        except Exception:
            logger.debug("remote kernel kill failed (transport?)", exc_info=True)
            return False
        probe_source = (
            "import subprocess,sys,time\n"
            "pid = int(sys.argv[1])\n"
            "deadline = time.monotonic() + 3\n"
            "while True:\n"
            "    rows = subprocess.check_output(['ps', '-eo', 'pid=,pgid=,stat='], text=True, stdin=subprocess.DEVNULL).splitlines()\n"
            "    live = False\n"
            "    for row in rows:\n"
            "        process, group, state = row.split()\n"
            "        if pid in (int(process), int(group)) and not state.startswith('Z'):\n"
            "            live = True\n"
            "    if not live:\n"
            "        print('RETIRED')\n"
            "        break\n"
            "    if time.monotonic() >= deadline:\n"
            "        sys.exit(1)\n"
            "    time.sleep(0.05)\n"
        )
        try:
            out = self.sh(f"python3 -c {shlex.quote(probe_source)} {q_pid}", timeout=10)
        except Exception:
            logger.debug("remote kernel retirement probe failed (transport?)", exc_info=True)
            return False
        try:
            self.sh(f"rm -rf {shlex.quote(self.kernel_dir)}")
        except Exception:
            logger.debug("remote kernel dir cleanup failed", exc_info=True)
        return True

    def kill(self) -> None:
        """Best-effort kill of the runner and its subprocesses, then rm -rf.

        Not a retirement: it reports nothing, so a caller that must know whether
        the process is gone uses kill_verified instead.
        """
        q_pid = shlex.quote(self.pid)
        for cmd, failure in (
            # Kill the runner's children if the shell gave it a group, then the PID itself.
            (f"pkill -TERM -P {q_pid} 2>/dev/null; kill {q_pid} 2>/dev/null; true",
             "remote kernel kill failed (transport?)"),
            (f"rm -rf {shlex.quote(self.kernel_dir)}", "remote kernel dir cleanup failed"),
        ):
            try:
                self.sh(cmd)
            except Exception:
                logger.debug(failure, exc_info=True)


def _is_alive(kernel: "RemoteKernel") -> bool:
    """Bounded liveness probe: kill -0 through the transport.

    A failed probe requires VERIFIED retirement, not immediate respawn: transport
    failure alone is not evidence of remote process death, so this answers "is it
    reachable and running", not "is it gone" — see RemoteKernel.kill_verified for the
    proof obligation a retirement actually needs.
    """
    return kernel.is_alive()


def _kernel_key(owner: str, env_type: str, task_env_id: str, sandbox_tools: frozenset) -> Tuple:
    """The hermes_tools stub module is generated from ``sandbox_tools`` once, at spawn, so a kernel
    is only reusable by calls with the SAME tool set; a different set gets its own kernel.

    The HOST-BOUND authority scope is part of the key for the same reason: a kernel
    holds the cell authority of the turn that spawned it, so a call arriving under a
    different principal/audience must not be able to reuse or revive that namespace.
    Message ids and display names are deliberately NOT part of it — they do not
    constitute a transition.
    """
    return (owner, "remote", env_type, task_env_id, tuple(sorted(sandbox_tools)), _authority_scope())


def _authority_scope() -> Tuple[str, ...]:
    """Host-bound principal/audience and conversation generation, not cell claims.

    Message IDs/display names do not reset state. The embedding host must
    authenticate and bind these real context APIs; callers able to forge
    ContextVars are outside this component's trust boundary.
    """
    from gateway.session_context import get_session_env
    from hermes_constants import hermes_home_key

    fields = (
        "HERMES_SESSION_PLATFORM", "HERMES_SESSION_SOURCE",
        "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_CHAT_TYPE",
        "HERMES_SESSION_THREAD_ID", "HERMES_SESSION_USER_ID",
        "HERMES_SESSION_USER_ID_ALT", "HERMES_SESSION_SCOPE_ID",
        "HERMES_SESSION_KEY", "HERMES_SESSION_ID", "HERMES_SESSION_PROFILE",
        "HERMES_UI_SESSION_ID",
    )
    return (hermes_home_key(), *(get_session_env(name, "") for name in fields))


# Registry + lock shared-shape with code_kernel; explicit retirement paths carry proof.
_REGISTRY = KernelRegistry(lambda kernel: kernel.kill())
_REMOTE_KERNELS: Dict[Tuple, RemoteKernel] = _REGISTRY.kernels

#: How long a retirement waits for the cell's RPC poller to drain. Admitted tool
#: calls may still be in flight when retirement begins, so this is a bound on the
#: WAIT, not a promise that draining finished — a timeout here means the drain is
#: unverified, which is why it must not be treated as a clean stop.
_RPC_DRAIN_TIMEOUT = 5


@dataclass
class _RemoteLifetime:
    """Per-slot lifetime bookkeeping, parallel to ``_REMOTE_KERNELS``.

    The two are kept in step on purpose: a slot can be accounted as USED before any
    PID exists, which is what stops an ambiguous launch reply from looking like
    "nothing happened". ``users`` counts callers that hold or are WAITING for the
    slot, so a drain cannot be observed as finished while someone is still queued.
    """
    lock: Any = field(default_factory=threading.Lock)
    closed: threading.Event = field(default_factory=threading.Event)
    users: int = 0
    poller: Optional[Any] = None
    authority: Optional[_CellAuthority] = None
    draining_authority: Optional[_CellAuthority] = None
    #: Sticky: this slot ended a cell while an admitted host call was still executing.
    #: The call cannot be cancelled and its effects cannot be rewound, so the slot must
    #: keep refusing until the operation is observed to have drained. Sticky because a
    #: plain attribute can be cleared by a later caller between the check and the use —
    #: and then a quarantine that was real a moment ago silently stops protecting.
    quarantined: bool = False
    fallback_active: bool = False
    #: Set when this slot is being retired. A host tool dispatch admitted before this
    #: may still finish later; one arriving after it is refused. Ordering admission by
    #: this flag (not by the non-cancellable tool call) is the point: retirement cannot
    #: rewind a call's external effects, so it must refuse the ones it cannot stop.
    retired: bool = False


_REMOTE_LIFETIMES: Dict[Tuple, _RemoteLifetime] = {}

#: The lock guarding remote-kernel registry state. It is deliberately the SAME object
#: as the registry's own lock rather than a second one: two locks would let a caller
#: hold "the" kernel lock while a mutation ran under the other, so the serialisation
#: the name promises would not actually exist.
_REMOTE_KERNELS_LOCK = _REGISTRY.lock

def _lifetime_base(key: Tuple) -> Tuple:
    """Owner/env identity shared across authority/tool-set transition keys."""
    return key[:4]


def _slot_key(key: Tuple) -> Tuple:
    """Serialize all authorities of one owner/env under one lifetime slot."""
    return _lifetime_base(key)


def _lifetime_error(note: str) -> Dict[str, Any]:
    # Not None: caller MUST NOT fall back outside the registered lifetime while a
    # previous operation might still be running.
    return {"status": "error", "stdout": "", "stderr": "", "traceback": "",
            "tool_calls_made": 0, "error": note,
            "kernel": {"remote": True, "note": note}}


_RPC_DRAINING_NOTE = (
    "Previous remote RPC is still draining; cell was not started. "
    "An already admitted host operation cannot be cancelled. Retry after it finishes."
)


def _lifetime_retired(slot_key: Tuple, lifetime: _RemoteLifetime, active_users: int) -> bool:
    # Called under the registry lock: no kernel for this owner/env, no poller,
    # no admitted host call, and no other caller/waiter.
    draining = (lifetime.draining_authority is not None
                and lifetime.draining_authority.is_draining())
    has_kernel = any(_slot_key(kernel_key) == slot_key for kernel_key in _REMOTE_KERNELS)
    return (lifetime.users == active_users and not has_kernel
            and not lifetime.quarantined and not draining
            and (lifetime.poller is None or not lifetime.poller.is_alive())
            and _REMOTE_LIFETIMES.get(slot_key) is lifetime)


def _forget_lifetime(key: Tuple, lifetime: _RemoteLifetime) -> None:
    # Keeping a slot with waiters prevents ABA: an old waiter must never acquire a
    # different lifetime's lock after the original slot is replaced.
    if _lifetime_retired(key, lifetime, 0):
        del _REMOTE_LIFETIMES[key]


def _mark_draining(lifetime: Optional[_RemoteLifetime], authority: Optional[_CellAuthority]) -> None:
    """Record that a cell ended with an admitted host call still executing.

    Sets the sticky quarantine and keeps the authority reference, so the slot can only
    stop refusing when that operation is OBSERVED to have drained. Clearing the flag
    without observing the drain is exactly how a real quarantine silently disappears.
    """
    if lifetime is None or authority is None:
        return
    with _REMOTE_KERNELS_LOCK:
        lifetime.draining_authority = authority
        lifetime.quarantined = True
        lifetime.closed.set()


def _quarantine_block(lifetime: _RemoteLifetime) -> Optional[Dict[str, Any]]:
    """Refuse while a slot's admitted host call is outstanding, else lift.

    Returns the refusal, or None once the operation has been OBSERVED to have drained.
    The lift happens in exactly one place, so every caller agrees on when a quarantine
    ends — and it happens only on observation, never on assumption.
    """
    with _REMOTE_KERNELS_LOCK:
        if not lifetime.quarantined:
            return None
        authority = lifetime.draining_authority
        if authority is not None and not authority.is_draining():
            lifetime.draining_authority = None
            lifetime.quarantined = False
            return None
    return _lifetime_error(_RPC_DRAINING_NOTE)


def _replace_retired_lifetime(key: Tuple, lifetime: _RemoteLifetime) -> Optional[_RemoteLifetime]:
    """Replace a fully retired closed slot while its sole caller holds its lock."""
    with _REMOTE_KERNELS_LOCK:
        if not _lifetime_retired(key, lifetime, 1):
            return None
        replacement = _RemoteLifetime(users=1)
        replacement.lock.acquire()  # unpublished: cannot block
        _REMOTE_LIFETIMES[key] = replacement
        lifetime.users = 0
    lifetime.lock.release()
    return replacement


#: Bumped by every shutdown. A spawn that straddles one must not publish its kernel:
#: the shutdown was a decision about the slots that existed when it ran, and a kernel
#: launched under the old state would be adopted into the new one unannounced.
_SHUTDOWN_GENERATION = 0

#: Returned by the spawn when a shutdown landed mid-launch: the caller must REFUSE
#: (never fall open to the per-call path), because a runner may already exist.
_SPAWN_ABORTED = object()


def _owner_shutdown(owner: str) -> bool:
    """True when a shutdown already closed every lifetime of *owner*."""
    with _REGISTRY.lock:
        return any(key and key[0] == owner and lifetime.closed.is_set()
                   for key, lifetime in _REMOTE_LIFETIMES.items())


def _shutdown_remote_lifetimes(owner: Optional[str] = None) -> None:
    global _SHUTDOWN_GENERATION
    with _REMOTE_KERNELS_LOCK:
        slots = [(key, slot) for key, slot in _REMOTE_LIFETIMES.items()
                 if owner is None or (key and key[0] == owner)]
        for _, slot in slots:
            slot.users += 1
            slot.closed.set()
            if slot.authority is not None:
                slot.authority.retire()
        _SHUTDOWN_GENERATION += 1
    for key, slot in slots:
        acquired = slot.lock.acquire(timeout=_RPC_DRAIN_TIMEOUT)
        try:
            if acquired:
                with _REMOTE_KERNELS_LOCK:
                    kernel = _REMOTE_KERNELS.get(key)
                if kernel is not None:
                    _retire_verified(key, kernel)
        finally:
            if acquired:
                slot.lock.release()
            with _REMOTE_KERNELS_LOCK:
                slot.users -= 1
                # Whether this slot may be forgotten is decided in ONE place, and it
                # accounts for a live poller, an active caller, and an admitted host
                # call that is still draining. Popping it here unconditionally erased
                # the quarantine of a slot that was still refusing new cells.
                _forget_lifetime(key, slot)


def _held_by_live_work(slot: Optional[_RemoteLifetime]) -> bool:
    """True while something could still be running under this slot.

    An admitted host call that has not returned, a live RPC poller, or a caller still
    inside the slot all mean the slot must keep refusing even across a full teardown —
    none of them can be cancelled, and their effects cannot be rewound. A slot held by
    NOTHING is the other case: only an unproven retirement is keeping it, and nothing can
    still be running, so retaining it just refuses cells that were never at risk.
    """
    if slot is None:
        return False
    if slot.users > 0:
        return True
    if slot.poller is not None and slot.poller.is_alive():
        return True
    return slot.draining_authority is not None and slot.draining_authority.is_draining()


def _retire_leftovers(owner: Optional[str]) -> None:
    """Retire and forget what shutdown can safely take over.

    Retirement always runs through the verified proof. A slot is only FORGOTTEN when
    nothing is left holding it: an active caller (``users``), a live poller, or an
    admitted host call still draining all keep their closed slot, because that slot IS
    the quarantine that refuses the next cell.
    """
    with _REMOTE_KERNELS_LOCK:
        leftovers = [(key, kernel) for key, kernel in _REMOTE_KERNELS.items()
                     if owner is None or (key and key[0] == owner)]
    for key, kernel in leftovers:
        _retire_verified(key, kernel)
    if owner is None:
        # FULL process disposal: a record whose death could not be PROVEN must not survive
        # it — EXCEPT when an admitted host call is still running. That call cannot be
        # cancelled and its effects cannot be rewound, so its slot must keep refusing even
        # across a full teardown; that is the one protection the quarantine genuinely
        # provides. A placeholder with no admitted call is different: nothing can still be
        # running, and retaining it is what used to refuse cells in a LATER turn — and in
        # the next test of a shared process, since a pid-less launch can never have its
        # death proven by design.
        with _REMOTE_KERNELS_LOCK:
            for key in list(_REMOTE_KERNELS):
                if _held_by_live_work(_REMOTE_LIFETIMES.get(_slot_key(key))):
                    continue
                _REMOTE_KERNELS.pop(key, None)
    with _REMOTE_KERNELS_LOCK:
        for slot_key, slot in list(_REMOTE_LIFETIMES.items()):
            if owner is not None and not (slot_key and slot_key[0] == owner):
                continue
            if owner is None:
                if _held_by_live_work(slot):
                    continue  # something may still be running under it: keep refusing
                _REMOTE_LIFETIMES.pop(slot_key, None)
                continue
            if any(_slot_key(k) == slot_key for k in _REMOTE_KERNELS):
                continue
            if slot.users > 0:
                continue
            if slot.poller is not None and slot.poller.is_alive():
                continue
            if slot.quarantined:
                continue
            # Consult the authority objects directly as well: whichever of them is
            # still inside an admitted host call holds this slot, regardless of what
            # any single flag currently says.
            if slot.draining_authority is not None and slot.draining_authority.is_draining():
                continue
            if slot.authority is not None and slot.authority.is_draining():
                continue
            _REMOTE_LIFETIMES.pop(slot_key, None)


def shutdown_all_remote_kernels() -> None:
    _shutdown_remote_lifetimes()
    _retire_leftovers(None)


def shutdown_remote_kernels_for_owner(owner: str) -> None:
    """Session-boundary disposal — retires active, queued, and spawning calls.

    A slot whose caller is still active, or which still has an admitted host call
    draining, keeps refusing new cells after shutdown: its unverified state is exactly
    what must not be handed to the next caller.
    """
    if owner:
        _shutdown_remote_lifetimes(owner)
        _retire_leftovers(owner)


def shutdown_remote_kernels_where(owner_matches: Callable[[str], bool]) -> None:
    """Dispose every remote kernel whose owner satisfies the predicate (a finished child's kernels)."""
    _REGISTRY.shutdown(owner_matches=owner_matches)


def _reap_unlocked(idle_timeout: int) -> List["RemoteKernel"]:
    """Pop idle-expired, unattached remote kernels; caller tears them down outside the lock. The
    runner self-exits after the same idle window, so this clears the HOST-side entry — without it
    the map grew one entry per never-revisited (owner, env_type, task_env_id) for the gateway's life."""
    now = time.monotonic()
    doomed = [key for key, kernel in _REMOTE_KERNELS.items()
              if kernel.attached == 0 and now - kernel.last_used > idle_timeout]
    return [_REMOTE_KERNELS.pop(key) for key in doomed]


def _evict_over_cap_unlocked(keep: Tuple) -> List["RemoteKernel"]:
    """Pop least-recently-used unattached remote kernels beyond the process-wide cap (the same
    ``max_session_kernels`` bound as local kernels, applied independently to this map)."""
    from tools.code_kernel import _lifecycle_limits
    cap, _ = _lifecycle_limits()
    unpinned = [key for key in _REMOTE_KERNELS if not _REMOTE_KERNELS[key].pinned]
    if len(unpinned) <= cap:
        return []
    by_age = sorted((key for key in unpinned if key != keep and _REMOTE_KERNELS[key].attached == 0),
                    key=lambda key: _REMOTE_KERNELS[key].last_used)
    return [_REMOTE_KERNELS.pop(key) for key in by_age[: len(unpinned) - cap]]


atexit.register(shutdown_all_remote_kernels)


# ---- Verified retirement --------------------------------------------------
# A kill is not a retirement. SIGTERM is catchable by cell code, a dropped
# transport proves nothing, and a kernel we cannot reach may STILL be running
# with the retired cell's authority and RPC token. So retirement is reported only
# once death is proven, and until then the registry record is KEPT as the
# quarantine: the slot refuses new cells rather than being reused, and a refusal
# is never allowed to fall open to the per-call path (which would run a cell the
# caller was told was blocked, with none of the accounting it was promised).
def _retire_verified(key: Tuple, kernel: "RemoteKernel") -> bool:
    """Retire *kernel* only on PROVEN death; keep the record otherwise."""
    # Marked BEFORE the transport runs, so a concurrent caller cannot reuse the slot.
    kernel.retiring = True
    with _REGISTRY.lock:
        lifetime = _REMOTE_LIFETIMES.get(_slot_key(key))
    if lifetime is not None:
        # Retired BEFORE the transport: from this moment a host tool that has not been
        # admitted yet is refused, while one already admitted is still counted — the
        # call cannot be cancelled, so ordering admission is the only real protection.
        lifetime.retired = True
    if not kernel.kill_verified():
        return False
    # Delete only after proof. Deleting first and probing afterwards cannot work:
    # there would be nothing left to prove anything about, and an unreachable
    # kernel would be forgotten rather than refused.
    with _REGISTRY.lock:
        if _REMOTE_KERNELS.get(key) is kernel:
            del _REMOTE_KERNELS[key]
    return True


def _quarantine_refusal(key: Tuple) -> Dict[str, Any]:
    """Result for a cell refused because a slot's retirement is unproven.

    Deliberately NOT ``None``: None means "no kernel, fall open to per-call", and
    falling open is exactly what a quarantine must prevent.
    """
    note = ("The previous remote kernel's retirement is unverified, or a host tool call it "
            "admitted is still draining; the lifetime is quarantined and no new cell can "
            "start until that is established. The admitted call cannot be cancelled and its "
            "external effects cannot be rewound, so it is waited for rather than raced.")
    return {"status": "error", "stdout": "", "stderr": "", "traceback": "",
            "tool_calls_made": 0, "error": note,
            "kernel": {"remote": True, "quarantined": True, "reused": False, "note": note}}


def _launch_reply_is_definite_failure(started: Any) -> bool:
    """True when the transport never confirmed the launch command ran at all.

    A non-conforming reply — no ``returncode`` AND no output of any kind — is not
    ambiguous evidence about a running process; it is evidence the command was never
    reported as executed, which is the "host cannot sustain a background process" case
    the per-call route exists for. Every real transport adapter reports a returncode,
    and any output at all (including a malformed PID) keeps the stricter reading: the
    command ran, so an unidentifiable runner may exist and the slot must refuse.
    """
    if not isinstance(started, dict):
        return False
    out = started.get("output") or ""
    # A pid line means the runner reported one: it exists whatever the status, so the
    # strict reading stands and the slot must refuse.
    if any(line.strip().startswith("PID:") and line.strip()[4:].strip().isdigit()
           for line in out.splitlines()):
        return False
    if "returncode" in started:
        # The command was attempted. A NONZERO status with no pid means the attempt
        # failed and the background job never came up (`sh: cannot fork`), so nothing was
        # started and there is nothing to retire — the caller may fall open. A zero status
        # with no pid is the ambiguous case: the command ran and produced no identifiable
        # runner, so the stricter reading keeps the slot refusing.
        rc = started.get("returncode")
        return bool(rc) and rc != 0
    return not out.strip()


def _spawn_remote_kernel(env, env_type: str, owner: str, task_env_id: str,
                         sandbox_tools: frozenset, *, idle_exit: int) -> Any:
    """Start a detached kernel runner on the remote.

    Returns the kernel, ``None`` on failure (directory removed when nothing could
    have launched), or ``_SPAWN_ABORTED`` when a shutdown landed while the launch was
    in flight — in which case the caller must REFUSE rather than fall open, because a
    runner may already exist that we chose not to adopt.
    """
    from tools.code_execution_tool import (
        MAX_STDOUT_BYTES, _ship_file_to_remote, _env_temp_dir, generate_hermes_tools_module,
    )
    kernel_dir = f"{_env_temp_dir(env)}/hermes_rkernel_{uuid.uuid4().hex[:12]}"
    q_dir = shlex.quote(kernel_dir)
    kernel = None
    # Set the moment the launch is dispatched. Everything before it is safe to undo;
    # after it, an unidentifiable runner may exist and the record must survive.
    launch_attempted = False
    try:
        _sh(env, f"mkdir -p {q_dir}/cells {q_dir}/rpc")
        rpc_token = secrets.token_urlsafe(32)
        _ship_file_to_remote(env, f"{kernel_dir}/kernel_runner.py", REMOTE_KERNEL_RUNNER_SOURCE.format(
            cell_source=RUNNER_CELL_SOURCE, capture_limit=MAX_STDOUT_BYTES, idle_exit=idle_exit))
        _ship_file_to_remote(env, f"{kernel_dir}/hermes_tools.py",
                             generate_hermes_tools_module(list(sandbox_tools), transport="file"))
        env_prefix = (f"HERMES_KERNEL_DIR={q_dir} HERMES_RPC_DIR={shlex.quote(kernel_dir + '/rpc')} "
                      f"HERMES_RPC_TOKEN={shlex.quote(rpc_token)} PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={q_dir}")
        # Accounting BEFORE the transport. An ambiguous launch reply (lost, malformed,
        # nonzero, or no PID) may still have started a process, so the attempt must be
        # on record at the moment the launch is dispatched — otherwise a failed reply
        # looks like "nothing happened" and the next call launches a second runner
        # beside the first. The placeholder is marked retiring with no PID: it is
        # accountable, unusable, and (because no death can be proven without a PID)
        # it quarantines the slot rather than being forgotten.
        key = _kernel_key(owner, env_type, task_env_id, sandbox_tools)
        placeholder = RemoteKernel(env=env, env_identity=id(env), env_type=env_type, kernel_dir=kernel_dir,
                                   pid="", rpc_token=rpc_token, owner=owner, retiring=True)
        with _REGISTRY.lock:
            _REMOTE_LIFETIMES.setdefault(_slot_key(key), _RemoteLifetime())
            _REMOTE_KERNELS[key] = placeholder
        launch_attempted = True
        gen_before = _SHUTDOWN_GENERATION
        # Kept as the raw reply (not just its text) because whether the transport even
        # confirmed the command decides ambiguous-vs-definite failure below.
        launched = env.execute(
            f"cd {q_dir} || exit; mkdir {q_dir}/launch 2>/dev/null || exit; "
            f"nohup env {env_prefix} python3 kernel_runner.py "
            f"< /dev/null > {q_dir}/runner.log 2>&1 & pid=$!; "
            f"printf '%s\\n' \"$pid\" > {q_dir}/launch/launcher-pid.tmp; "
            f"mv {q_dir}/launch/launcher-pid.tmp {q_dir}/launch/pid; echo PID:$pid",
            cwd="/", timeout=20)
        started = (launched.get("output", "") if isinstance(launched, dict) else "") or ""
        if _launch_reply_is_definite_failure(launched):
            # The launch was never reported as executed: nothing can be running, so this
            # is a definite pre-launch failure and the caller may fall open to per-call.
            # The placeholder must go with it, or the slot would refuse forever over a
            # process that was never started.
            if gen_before == _SHUTDOWN_GENERATION:
                with _REGISTRY.lock:
                    _REMOTE_KERNELS.pop(key, None)
                    _REMOTE_LIFETIMES.pop(_slot_key(key), None)
                try:
                    _sh(env, f"rm -rf {q_dir}")
                except Exception:
                    logger.debug("remote kernel dir cleanup failed", exc_info=True)
                return None
        pid = next((line.strip()[4:].strip() for line in started.splitlines()
                    if line.strip().startswith("PID:")), "")
        if not pid.isdigit():
            logger.warning("remote kernel spawn returned no PID: %r", started)
        else:
            candidate = RemoteKernel(env=env, env_identity=id(env), env_type=env_type, kernel_dir=kernel_dir,
                                     pid=pid, rpc_token=rpc_token, owner=owner)
            if candidate.is_alive():
                kernel = candidate
                with _REGISTRY.lock:
                    _REMOTE_KERNELS[key] = candidate
            else:
                # Died instantly (missing python3 was pre-checked by the caller,
                # so this is unexpected) — surface the runner log.
                try:
                    logger.warning("remote kernel died at spawn: %s",
                                   _sh(env, f"cat {q_dir}/runner.log", timeout=10)[:500])
                except Exception:
                    pass
        if gen_before != _SHUTDOWN_GENERATION:
            # A shutdown landed while we were launching this kernel. It belongs to the
            # state that was just retired, so it is neither published nor left running:
            # retire it (verified, best effort) and drop both accounting records, which
            # also leaves the slot free for the next call to start a fresh lifetime.
            if kernel is not None:
                _retire_verified(key, kernel)
            with _REGISTRY.lock:
                _REMOTE_KERNELS.pop(key, None)
                _REMOTE_LIFETIMES.pop(_slot_key(key), None)
            return _SPAWN_ABORTED
    except Exception:
        logger.warning("remote kernel spawn failed", exc_info=True)
    if kernel is None and not launch_attempted:
        # Never reached the dispatch: nothing can be running, so the directory is safe
        # to remove. When a launch WAS attempted the directory stays — a runner we
        # cannot identify may still be writing to it, and that record is the quarantine.
        try:
            _sh(env, f"rm -rf {q_dir}")
        except Exception:
            pass
    return kernel


def _acquire_remote_kernel(env, env_type: str, owner: str, task_env_id: str,
                           sandbox_tools: frozenset, *, reset: bool,
                           idle_exit: int) -> Tuple[Optional[RemoteKernel], bool, bool, bool, Optional[Tuple]]:
    """Find/respawn the owner's kernel: (kernel|None, reused, state_reset, state_lost, refused_key);
    reaps idle-expired entries on the way in. A non-None ``refused_key`` means the slot is
    quarantined — the caller must REFUSE the cell, never fall open to per-call."""
    key = _kernel_key(owner, env_type, task_env_id, sandbox_tools)
    state_lost = state_reset = False
    with _REGISTRY.lock:
        expired = _reap_unlocked(idle_exit)
        kernel = _REMOTE_KERNELS.get(key)
    for doomed in expired:
        doomed.kill()
    # An AUTHORITY TRANSITION retires the previous namespace instead of leaving it
    # live: the old kernel still holds the retired turn's cell authority, so it must
    # be torn down (verified) and the loss reported — never reused, and never left
    # running beside the new one. "Authority" covers BOTH dimensions that shape what a
    # cell may do: the tool set its stub module was generated from, and the host-bound
    # scope it was admitted under. Either one changing makes the old kernel a
    # different authority, and its namespace must not survive.
    with _REGISTRY.lock:
        stale = [k for k in _REMOTE_KERNELS
                 if k != key and k[:4] == key[:4] and (k[4] != key[4] or k[5] != key[5])]
    for old_key in stale:
        old = _REMOTE_KERNELS.get(old_key)
        if old is None:
            continue
        if not _retire_verified(old_key, old):
            # The previous authority's kernel could not be proven dead. Refuse rather
            # than start a second one beside it.
            return None, False, False, False, old_key
        state_lost = True
    if kernel is not None and kernel.env_identity != id(env):
        if not _retire_verified(key, kernel):
            return None, False, False, False, key
        kernel, state_lost = None, True
    if kernel is not None and kernel.retiring:
        # A previous retirement was never proven. Retry the proof, and while it stays
        # unproven the slot is REFUSED: never reused, and never silently respawned
        # over the top of a process that may still hold the retired authority.
        if not _retire_verified(key, kernel):
            return None, False, False, False, key
        kernel = None
    if kernel is not None and reset:
        # An explicit reset is also a retirement, so it carries the same proof
        # requirement: a fresh kernel must not start while the old one's death is
        # unproven, or two processes would hold the same cell authority.
        if not _retire_verified(key, kernel):
            return None, False, False, False, key
        kernel, state_reset = None, True
    if kernel is not None and not kernel.is_alive():
        # Transport drop, container restart, self-reaped on idle, OOM — all
        # the same answer: report the loss, respawn fresh (kill is then only
        # best-effort dir cleanup; the process is already gone).
        _REGISTRY.discard(key, kernel)
        kernel, state_lost = None, True
    reused = kernel is not None
    if kernel is None:
        kernel = _spawn_remote_kernel(env, env_type, owner, task_env_id, sandbox_tools, idle_exit=idle_exit)
        if kernel is _SPAWN_ABORTED:
            # A shutdown landed while this kernel was launching. The call is refused
            # (never fall open to per-call) and the slot was left free, so the NEXT
            # call starts a fresh lifetime rather than inheriting the retired state.
            return None, False, False, False, key
        if kernel is not None:
            from agent.delegation_context import is_delegated_child_context
            kernel.pinned = is_delegated_child_context()
            with _REGISTRY.lock:
                _REMOTE_KERNELS[key] = kernel
        else:
            # The launch was ATTEMPTED and could not be turned into a live kernel. The
            # placeholder the spawn left behind is the quarantine, so the call is
            # refused here: returning None would fall open to per-call while a runner
            # we cannot identify may already be executing on the remote.
            with _REGISTRY.lock:
                left_behind = _REMOTE_KERNELS.get(key)
            if left_behind is not None and left_behind.retiring:
                return None, False, False, False, key
    return kernel, reused, state_reset, state_lost, None


def _run_remote_cell(kernel: RemoteKernel, code: str, timeout: int, *,
                     lifetime: Optional["_RemoteLifetime"] = None,
                     rpc_dir: str, rpc_token: str,
                     authority: _CellAuthority) -> Tuple[str, Dict[str, Any]]:
    """Ship one cell request and poll for its result: (cell status, payload)."""
    from tools.code_execution_tool import _ship_file_to_remote
    kernel.cell_seq += 1
    seq = f"{kernel.cell_seq:06d}"
    q_cells, q_res = shlex.quote(f"{kernel.kernel_dir}/cells"), shlex.quote(f"cell_res_{seq}.json")
    _ship_file_to_remote(kernel.env, f"{kernel.kernel_dir}/cells/cell_req_{seq}.json.tmp",
                         json.dumps({"id": seq, "code": code,
                                     "rpc_dir": rpc_dir, "rpc_token": rpc_token},
                                    ensure_ascii=False))
    # The request is STAGED but not yet published. If the slot closed while we were
    # shipping — shutdown, retirement — publishing now would hand a cell to a kernel
    # whose authority is retired. So the staged request is removed and the cell
    # refuses: nothing is published and nothing executes.
    if lifetime is not None and lifetime.closed.is_set():
        try:
            kernel.sh(f"rm -f {q_cells}/cell_req_{seq}.json.tmp", timeout=10)
        except Exception:
            pass
        return "error", {"error": "Remote cell authority retired while the cell was in flight."}
    kernel.sh(f"mv {q_cells}/cell_req_{seq}.json.tmp {q_cells}/cell_req_{seq}.json", timeout=10)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            body = kernel.sh(f"cat {q_cells}/{q_res} 2>/dev/null", timeout=20).strip()
        except Exception:
            # One flaky round-trip is not kernel death; liveness decides.
            body = ""
        if body:
            # Settlement precedes remote cleanup: a delayed request read must not
            # admit a callback after the cell result has been observed.
            authority.retire()
            try:
                payload = json.loads(body)
                status = payload.get("status", "error")
            except ValueError:
                payload, status = {}, "protocol-error"
            kernel.sh(f"rm -f {q_cells}/{q_res}", timeout=10)
            return status, payload
        time.sleep(_CELL_POLL_INTERVAL)
    return "timeout", {}


def execute_in_remote_kernel(
    code: str, *, env, env_type: str, task_env_id: str, sandbox_tools: frozenset,
    timeout: int, max_tool_calls: int, reset: bool, idle_exit: int = 1800,
    fallback: Optional[Callable[[_RemoteLifetime, _CellAuthority, str, Tuple], Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Serialize the whole remote lifetime, including spawn and accounted fallback."""
    from tools.code_kernel import _resolve_owner

    owner = _resolve_owner(task_env_id)
    key = _kernel_key(owner, env_type, task_env_id, sandbox_tools)
    slot_key = _slot_key(key)
    with _REMOTE_KERNELS_LOCK:
        lifetime = _REMOTE_LIFETIMES.setdefault(slot_key, _RemoteLifetime())
        if lifetime.fallback_active:
            return _lifetime_error("Remote fallback is busy; cell was not started.")
        lifetime.users += 1

    acquired = False
    cancelled = False
    try:
        acquired = lifetime.lock.acquire(timeout=max(0, timeout))
        if not acquired:
            return _lifetime_error("Remote kernel is busy; cell was not started.")
        if lifetime.closed.is_set():
            blocked = _quarantine_block(lifetime)
            if blocked is not None:
                return blocked
            replacement = _replace_retired_lifetime(slot_key, lifetime)
            if replacement is None:
                if lifetime.poller is not None and lifetime.poller.is_alive():
                    return _lifetime_error(_RPC_DRAINING_NOTE)
                # A quarantined kernel needs one retry to verify retirement. If that
                # succeeds, this closed generation is refused once and the following
                # caller gets a fresh lifetime; never start in the same call that proved
                # the previous generation dead.
                with _REMOTE_KERNELS_LOCK:
                    quarantined = [
                        (old_key, old) for old_key, old in _REMOTE_KERNELS.items()
                        if _slot_key(old_key) == slot_key and old.retiring
                    ]
                retired = bool(quarantined)
                for old_key, old in quarantined:
                    retired = _retire_verified(old_key, old) and retired
                if retired:
                    with _REMOTE_KERNELS_LOCK:
                        if not any(_slot_key(k) == slot_key for k in _REMOTE_KERNELS):
                            _REMOTE_LIFETIMES.pop(slot_key, None)
                return _lifetime_error("Remote kernel lifetime was shut down; cell was not started.")
            lifetime = replacement
        if lifetime.poller is not None and lifetime.poller.is_alive():
            return _lifetime_error(_RPC_DRAINING_NOTE)
        blocked = _quarantine_block(lifetime)
        if blocked is not None:
            return blocked

        kernel, reused, state_reset, state_lost, refused_key = _acquire_remote_kernel(
            env, env_type, owner, task_env_id, sandbox_tools, reset=reset, idle_exit=idle_exit)
        if refused_key is not None:
            result: Optional[Dict[str, Any]] = _quarantine_refusal(refused_key)
        elif kernel is None:
            result = None
        else:
            kernel.last_used = time.monotonic()
            with _REGISTRY.lock:
                kernel.attached += 1
                evicted = _evict_over_cap_unlocked(keep=key)
            for doomed in evicted:
                doomed.kill()
            try:
                result = _run_attached_cell(
                    kernel, key, code, env=env, task_env_id=task_env_id,
                    sandbox_tools=sandbox_tools, timeout=timeout,
                    max_tool_calls=max_tool_calls, reused=reused,
                    state_reset=state_reset, state_lost=state_lost)
            finally:
                with _REGISTRY.lock:
                    kernel.attached -= 1
                    kernel.last_used = time.monotonic()

        if result is None and fallback is not None:
            authority = _CellAuthority(task_env_id)
            with _REMOTE_KERNELS_LOCK:
                lifetime.fallback_active = True
                lifetime.authority = authority
                if lifetime.closed.is_set():
                    authority.retire()
            try:
                if authority.stop.is_set():
                    return _lifetime_error("Remote lifetime shut down; no fallback started.")
                return fallback(lifetime, authority, owner, key)
            finally:
                authority.retire()
                with _REMOTE_KERNELS_LOCK:
                    lifetime.fallback_active = False
        return result
    except BaseException as exc:
        if acquired:
            lifetime.closed.set()
            if lifetime.authority is not None:
                lifetime.authority.retire()
        cancelled = not isinstance(exc, Exception)
        if cancelled:
            raise
        logger.warning("Remote kernel lifetime failed", exc_info=True)
        return _lifetime_error("Remote kernel lifetime failed; no fallback execution was started.")
    finally:
        try:
            if acquired and lifetime.closed.is_set():
                try:
                    with _REMOTE_KERNELS_LOCK:
                        kernel = _REMOTE_KERNELS.get(key)
                    if kernel is not None:
                        _retire_verified(key, kernel)
                except BaseException:
                    if not cancelled:
                        raise
                    logger.debug("Remote cancellation retirement unverified", exc_info=True)
        finally:
            if acquired:
                lifetime.lock.release()
            with _REMOTE_KERNELS_LOCK:
                lifetime.users -= 1
                if lifetime is not _REMOTE_LIFETIMES.get(slot_key):
                    # _replace_retired_lifetime transferred this caller to a fresh slot
                    # and already zeroed the old one.
                    pass
                else:
                    _forget_lifetime(slot_key, lifetime)


def _run_attached_cell(kernel: RemoteKernel, key: Tuple, code: str, *, env, task_env_id: str,
                       sandbox_tools: frozenset, timeout: int, max_tool_calls: int,
                       reused: bool, state_reset: bool, state_lost: bool) -> Dict[str, Any]:
    from tools.code_execution_tool import _rpc_poll_loop
    from tools.thread_context import propagate_context_to_thread

    # A unique capability namespace per cell: delayed pollers and stale imported
    # wrappers can name only this cell's files/token, never those of the next cell.
    rpc_dir = f"{kernel.kernel_dir}/rpc/{uuid.uuid4().hex}"
    rpc_token = secrets.token_urlsafe(32)
    authority = _CellAuthority(task_env_id)
    tool_call_counter = [0]
    lifetime = _REMOTE_LIFETIMES.get(_slot_key(key))
    if lifetime is not None:
        lifetime.authority = authority
        if lifetime.closed.is_set():
            authority.retire()

    rpc_thread = threading.Thread(
        target=propagate_context_to_thread(_rpc_poll_loop),
        args=(env, rpc_dir, task_env_id, [], tool_call_counter,
              max_tool_calls, sandbox_tools, authority.stop, rpc_token),
        kwargs={"dispatch": authority.dispatch},
        daemon=True,
    )
    if lifetime is not None:
        lifetime.poller = rpc_thread
    cell_status, cell_payload = "no-result", {}
    try:
        kernel.sh(f"mkdir -p {shlex.quote(rpc_dir)}", timeout=10)
        rpc_thread.start()
        cell_status, cell_payload = _run_remote_cell(
            kernel, code, timeout, lifetime=lifetime,
            rpc_dir=rpc_dir, rpc_token=rpc_token, authority=authority)
    finally:
        authority.retire()
        if authority.is_draining() and lifetime is not None:
            # The poll loop may return as soon as retirement is signalled while the
            # admitted host operation is still executing outside it. Keep this slot
            # closed and registered until that operation has really drained.
            _mark_draining(lifetime, authority)
            if kernel is not None:
                kernel.retiring = True
        if rpc_thread.ident is not None:
            rpc_thread.join(timeout=_RPC_DRAIN_TIMEOUT)
        if rpc_thread.is_alive() and cell_status not in ("timeout", "shutdown"):
            cell_status = "rpc-draining"
        if not rpc_thread.is_alive():
            try:
                kernel.sh(f"rm -rf {shlex.quote(rpc_dir)}", timeout=10)
            except Exception:
                logger.debug("Remote cell RPC cleanup failed", exc_info=True)
    kernel_info: Dict[str, Any] = {"reused": reused, "remote": True}
    result: Dict[str, Any] = {
        "status": "error", "stdout": cell_payload.get("stdout", ""), "stderr": cell_payload.get("stderr", ""),
        "traceback": cell_payload.get("traceback", ""), "tool_calls_made": tool_call_counter[0], "kernel": kernel_info,
    }
    if cell_status in ("timeout", "protocol-error", "no-result"):
        # No safe way to interrupt one cell in place (same contract as local): retire,
        # report, respawn. The report must match what is actually KNOWN — a retirement
        # that could not be proven is not "killed", it is quarantined, and the slot
        # refuses the next cell until the process is shown dead.
        if cell_status == "timeout":
            result["status"] = "timeout"
        if _retire_verified(key, kernel):
            kernel_info.update(ended=True, state_lost=True, note=(
                "Cell timed out; the remote session kernel was killed and its state was lost. The next call "
                "starts a fresh kernel." if cell_status == "timeout"
                else "Remote kernel protocol failure; kernel killed, state lost."))
        else:
            kernel_info.update(ended=True, state_lost=True, quarantined=True, note=(
                "Cell timed out and the remote session kernel's retirement is unverified; the lifetime is "
                "quarantined and no new cell can start until the process is proven dead."
                if cell_status == "timeout" else
                "Remote kernel protocol failure and its retirement is unverified; the lifetime is "
                "quarantined until the process is proven dead."))
        return result
    if authority.is_draining():
        _mark_draining(lifetime, authority)
        # An admitted host call outlived this cell. Its external effects cannot be
        # rewound, so the cell is reported as an error and the slot stays quarantined
        # until that call returns: the next call must WAIT for it, not race it.
        kernel_info.update(state_lost=True, draining=True, note=(
            "A host tool call admitted before this cell ended is still draining; the "
            "lifetime is quarantined and no new cell can start until it returns."))
        result["status"] = "error"
        result["error"] = kernel_info["note"]
        return result
    if cell_status == "exit":
        _REGISTRY.discard(key, kernel)
        kernel_info["ended"] = True
    kernel.execution_count = kernel_info["execution_count"] = int(cell_payload.get("execution_count", 0) or 0)
    if cell_status in ("ok", "exit"):
        result["status"] = "success"
    result["stdout_clipped"] = bool(cell_payload.get("stdout_clipped"))
    result["stderr_clipped"] = bool(cell_payload.get("stderr_clipped"))
    if state_reset:
        kernel_info["state_reset"] = True
    if state_lost:
        kernel_info.update(state_lost=True, note=(
            "The previous remote kernel was gone (transport drop, container "
            "restart, or idle self-exit); state from earlier calls was lost and a fresh kernel was started."))
    if cell_status == "error" and result["traceback"]:
        result["error"] = result["traceback"].strip().splitlines()[-1]
    return result


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import base64  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
