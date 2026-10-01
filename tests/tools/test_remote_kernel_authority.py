"""Real remote bootstrap + file RPC on a local shell, not a canned kernel.

Run in the canonical OS-confined test runner. No SSH/Docker/provider access.
These are API concurrency regressions, not claims about model dispatch timing.
"""
import concurrent.futures
import contextvars
import json
import subprocess
import tempfile
import threading
from pathlib import Path

import pytest

from tools import code_kernel_remote as remote

pytestmark = pytest.mark.linux_only
TURN = contextvars.ContextVar("remote_test_turn", default="unset")


class ShellEnv:
    def __init__(self, root):
        self.root = root
        self.commands = []
        from model_tools import handle_function_call
        self.real_dispatch = handle_function_call
        self.before = lambda command: None
        self.after = lambda command, result: None

    def get_temp_dir(self):
        return str(self.root)

    def execute(self, command, cwd=None, timeout=None):
        self.commands.append(command)
        self.before(command)
        # Detached grandchildren can retain PIPEs. Collect command output
        # without waiting for EOF from every grandchild (as real envs do).
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
            result = subprocess.run(
                ["/bin/bash", "-c", command], cwd=cwd, timeout=timeout,
                stdout=output, stderr=subprocess.DEVNULL,
            )
            output.seek(0)
            payload = {"output": output.read(), "returncode": result.returncode}
        self.after(command, payload)
        return payload


@pytest.fixture
def rig(tmp_path, monkeypatch):
    remote.shutdown_all_remote_kernels()
    env = ShellEnv(tmp_path)
    calls = []

    def dispatch(name, args, task_id=None):
        calls.append((name, args, task_id, TURN.get()))
        return json.dumps({"tool": name, "args": args, "turn": TURN.get()})

    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    yield env, calls
    env.before = lambda command: None
    env.after = lambda command, result: None
    remote.shutdown_all_remote_kernels()


def run(env, code, tools=("read_file",), *, task="authority", reset=False,
        timeout=12, turn="current", env_type="ssh"):
    token = TURN.set(turn)
    try:
        result = remote.execute_in_remote_kernel(
            code, env=env, env_type=env_type, task_env_id=task,
            sandbox_tools=frozenset(tools), timeout=timeout,
            max_tool_calls=5, reset=reset,
        )
        assert result is not None, "synthetic transport must start the real kernel"
        return result
    finally:
        TURN.reset(token)


def test_unchanged_authority_preserves_variables_imported_wrappers_and_rpc(rig):
    env, calls = rig
    first = run(env, "from hermes_tools import read_file\nx = 41\nprint(read_file('first'))")
    second = run(env, "x += 1\nprint(x, read_file('second'))", turn="second")
    assert first["status"] == second["status"] == "success"
    assert second["kernel"]["reused"]
    assert "42" in second["stdout"]
    assert [(c[1]["path"], c[3]) for c in calls] == [("first", "current"), ("second", "second")]


@pytest.mark.parametrize("next_tools", [(), ("write_file",)])
def test_authority_transition_retires_namespace_and_does_not_revive_old_kernel(rig, next_tools):
    env, calls = rig
    first = run(env, "from hermes_tools import read_file\nx = 'old'\nprint(read_file('approved'))")
    assert first["status"] == "success"
    old_dir = next(iter(remote._REMOTE_KERNELS.values())).kernel_dir
    second = run(env, "print('read_file' in globals(), 'x' in globals())", next_tools)
    assert second["stdout"].strip() == "False False"
    assert second["kernel"]["state_lost"]
    assert not Path(old_dir).exists()
    third = run(env, "print('x' in globals())\nfrom hermes_tools import read_file\nprint(read_file('fresh'))")
    assert third["stdout"].startswith("False\n")
    assert [c[1]["path"] for c in calls] == ["approved", "fresh"]


def test_previous_cell_background_wrapper_cannot_borrow_next_cell_authority(rig):
    env, calls = rig
    first = run(env, """
import threading
from hermes_tools import read_file
release = threading.Event()
finished = threading.Event()
def stale():
    release.wait()
    try:
        read_file('stale')
    except RuntimeError:
        pass
    finally:
        finished.set()
threading.Thread(target=stale, daemon=True).start()
""")
    assert first["status"] == "success"
    second = run(env, "release.set()\nassert finished.wait(5)\nprint(read_file('fresh'))", turn="later")
    assert second["status"] == "success", second
    assert [c[1]["path"] for c in calls] == ["fresh"]


def test_delayed_request_read_is_not_dispatched_after_cell_settles(rig):
    env, calls = rig
    read = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    pollers = []
    marker = env.root / 'request-read'

    def delay(command, payload):
        if command.startswith("cat ") and "/req_" in command and payload["output"]:
            pollers.append(threading.current_thread())
            read.set()
            marker.touch()
            assert release.wait(15)
            returned.set()

    env.after = delay
    try:
        result = run(env, f"""
import threading, time
from pathlib import Path
from hermes_tools import read_file
def delayed():
    try:
        read_file('delayed')
    except RuntimeError:
        pass
threading.Thread(target=delayed, daemon=True).start()
while not Path({str(marker)!r}).exists():
    time.sleep(0.02)
""")
        assert read.is_set(), "test must actually capture a real RPC request"
        assert result is not None
    finally:
        release.set()
    assert returned.wait(5)
    for thread in pollers:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert calls == []


def test_overlapping_cells_do_not_delete_each_others_rpc_and_keep_own_context(rig):
    env, calls = rig
    assert run(env, "x = 7")["status"] == "success"
    entered = threading.Event()
    release = threading.Event()

    def gate(command):
        if "/cell_req_000002.json.tmp" in command and command.startswith("mv "):
            entered.set()
            assert release.wait(10)

    env.before = gate
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        first = pool.submit(run, env, "from hermes_tools import read_file\nprint(read_file('one'))", turn="one")
        assert entered.wait(5)
        second = pool.submit(run, env, "import hermes_tools\nprint(hasattr(hermes_tools, 'read_file'))", (), turn="empty")
        release.set()
        a, b = first.result(20), second.result(20)
    assert a["status"] == b["status"] == "success", (a, b)
    assert b["stdout"].strip() == "False"
    assert [(c[1]["path"], c[3]) for c in calls] == [("one", "one")]


def test_authority_transition_kills_a_runner_that_ignores_term(rig):
    import psutil
    env, _ = rig
    first = run(env, "import os, signal, threading\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nthreading.Thread(target=lambda: threading.Event().wait()).start()\nprint(os.getpid())")
    assert first["status"] == "success"
    pid = int(first["stdout"])
    second = run(env, "print('fresh')", ())
    assert second["status"] == "success"
    # A killed orphan may be a zombie until the test namespace init reaps it.
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_simultaneous_spawn_is_single_flight_and_busy_call_never_runs(rig):
    env, calls = rig
    entered, release = threading.Event(), threading.Event()

    def hold_spawn(command):
        if "nohup" in command and not entered.is_set():
            entered.set()
            assert release.wait(10)

    env.before = hold_spawn
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        first = pool.submit(run, env, "x = 'first'")
        assert entered.wait(5)
        try:
            second = run(env, "x = 'second'", (), timeout=0)
            assert second["status"] == "error", second
            assert "busy" in second.get("error", "").lower()
        finally:
            release.set()
        assert first.result(15)["status"] == "success"
    assert len([c for c in env.commands if "nohup" in c]) == 1
    assert run(env, "print(x)")["stdout"].strip() == "first"
    assert calls == []


@pytest.mark.parametrize("all_owners", [False, True])
def test_shutdown_during_spawn_cannot_publish_or_execute_after_shutdown(rig, all_owners):
    env, calls = rig
    entered, release = threading.Event(), threading.Event()

    def hold_spawn(command):
        if "nohup" in command:
            entered.set()
            assert release.wait(15)

    env.before = hold_spawn
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        first = pool.submit(run, env, "from hermes_tools import read_file\nread_file('must-not-run')")
        assert entered.wait(5)
        try:
            if all_owners:
                remote.shutdown_all_remote_kernels()
            else:
                remote.shutdown_remote_kernels_for_owner("authority")
        finally:
            release.set()
        result = first.result(15)
    assert result["status"] == "error", result
    assert remote._REMOTE_KERNELS == {}
    assert calls == []
    assert run(env, "print('new-lifetime')")["status"] == "success"


def test_admitted_uncancellable_host_operation_quarantines_reset_and_empty_calls(rig, monkeypatch):
    env, calls = rig
    entered, release = threading.Event(), threading.Event()
    pollers = []
    marker = env.root / "admitted"

    def blocked_dispatch(name, args, task_id=None):
        pollers.append(threading.current_thread())
        calls.append((name, args, task_id, TURN.get()))
        marker.touch()
        entered.set()
        assert release.wait(20)
        return json.dumps({"finished": True})

    monkeypatch.setattr("model_tools.handle_function_call", blocked_dispatch)
    try:
        first = run(env, f"""
import threading, time
from pathlib import Path
from hermes_tools import read_file
def request():
    try:
        read_file('admitted-before-retirement')
    except RuntimeError:
        pass
threading.Thread(target=request, daemon=True).start()
while not Path({str(marker)!r}).exists():
    time.sleep(0.02)
""", turn="old")
        assert entered.is_set()
        assert first["status"] == "error", first
        assert first["kernel"]["state_lost"]
        before = list(env.commands)
        second = run(env, "print('must-not-run')", (), reset=True)
        assert second["status"] == "error", second
        assert "draining" in second["error"]
        assert env.commands == before, "quarantine must not clean/spawn/ship a new cell"
    finally:
        release.set()
    for poller in pollers:
        poller.join(5)
        assert not poller.is_alive()
    fresh = run(env, "print('fresh')", ())
    assert fresh["status"] == "success", fresh
    assert [(c[1]["path"], c[3]) for c in calls] == [("admitted-before-retirement", "old")]


def test_thread_pool_rpc_is_allowed_in_the_cell_that_started_it(rig):
    env, calls = rig
    result = run(env, """
from concurrent.futures import ThreadPoolExecutor
from hermes_tools import read_file
with ThreadPoolExecutor(2) as pool:
    print(list(pool.map(read_file, ['a', 'b'])))
""")
    assert result["status"] == "success", result
    assert sorted(c[1]["path"] for c in calls) == ["a", "b"]
    assert result["tool_calls_made"] == 2


def test_reset_waits_for_current_cell_without_revoking_its_rpc(rig):
    env, calls = rig
    entered, release = threading.Event(), threading.Event()
    assert run(env, "x = 'old'")["status"] == "success"

    def hold_cell(command):
        if command.startswith("mv ") and "cell_req_000002.json.tmp" in command:
            entered.set()
            assert release.wait(10)

    env.before = hold_cell
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        first = pool.submit(run, env, "from hermes_tools import read_file\nprint(read_file('current'))")
        assert entered.wait(5)
        try:
            refused = run(env, "print('not-run')", reset=True, timeout=0)
            assert refused["status"] == "error", refused
            assert "busy" in refused.get("error", "").lower()
        finally:
            release.set()
        assert first.result(15)["status"] == "success"
    reset = run(env, "print('x' in globals())", reset=True)
    assert reset["status"] == "success"
    assert reset["stdout"].strip() == "False"
    assert reset["kernel"]["state_reset"]
    assert [c[1]["path"] for c in calls] == ["current"]


@pytest.mark.parametrize("field", [
    "platform", "source", "chat_id", "chat_type", "thread_id", "user_id",
    "user_id_alt", "scope_id", "session_key", "session_id", "profile",
    "ui_session_id", "home",
])
def test_authenticated_scope_transition_cannot_reuse_or_revive_namespace(rig, field):
    from gateway.session_context import set_session_vars
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    env, calls = rig
    identity = dict(platform="synthetic", source="gateway", chat_id="room-a",
                    chat_type="group", thread_id="thread-a", user_id="actor-a",
                    user_id_alt="alias-a", scope_id="tenant-a", session_key="key-a",
                    session_id="conversation-a", profile="profile-a", ui_session_id="ui-a")

    def invoke(values, code, home="a"):
        def bound():
            from tools.approval_context import set_current_session_key
            set_current_session_key("stable-kernel-owner")
            set_session_vars(**values)
            token = set_hermes_home_override(str(env.root / home))
            try:
                return run(env, code)
            finally:
                reset_hermes_home_override(token)
        return contextvars.copy_context().run(bound)

    assert invoke(identity, "x = 'private'\nfrom hermes_tools import read_file")["status"] == "success"
    changed = dict(identity)
    if field != "home":
        changed[field] += "-b"
    second = invoke(changed, "print('x' in globals())\nfrom hermes_tools import read_file\nprint(read_file('new'))",
                    home="b" if field == "home" else "a")
    assert second["status"] == "success", second
    assert second["stdout"].startswith("False\n"), field
    assert not second["kernel"]["reused"], field
    third = invoke(identity, "print('x' in globals())")
    assert third["stdout"].strip() == "False", field
    assert [c[1]["path"] for c in calls] == ["new"]


def test_message_metadata_change_preserves_same_principal_namespace(rig):
    from gateway.session_context import set_session_vars
    env, _ = rig

    def invoke(message, code):
        def bound():
            set_session_vars(platform="synthetic", user_id="actor", chat_id="room",
                             session_id="conversation", message_id=message, user_name=message)
            return run(env, code)
        return contextvars.copy_context().run(bound)

    assert invoke("one", "x = 42")["status"] == "success"
    result = invoke("two", "print(x)")
    assert result["stdout"].strip() == "42"
    assert result["kernel"]["reused"]


def test_shutdown_while_shipping_never_publishes_the_cell(rig, monkeypatch):
    env, calls = rig
    assert run(env, "x = 1")["status"] == "success"
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(remote, "_RPC_DRAIN_TIMEOUT", 0.1)

    def hold_ship(command, payload):
        if command.startswith("echo ") and "cell_req_000002.json.tmp" in command:
            entered.set()
            assert release.wait(10)

    env.after = hold_ship
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        cell = pool.submit(run, env, "print('must-not-execute')")
        assert entered.wait(5)
        try:
            remote.shutdown_remote_kernels_for_owner("authority")
        finally:
            release.set()
        result = cell.result(15)
    assert result["status"] == "error", result
    assert not any(c.startswith("mv ") and "cell_req_000002.json.tmp" in c for c in env.commands)
    assert calls == []


@pytest.mark.parametrize("operation", ["reset", "transition", "timeout", "shutdown"])
def test_failed_remote_retirement_quarantines_until_death_is_verified(rig, operation):
    env, _ = rig
    assert run(env, "x = 'old'")["status"] == "success"

    def refuse_kill(command):
        if "kill -KILL" in command:
            raise OSError("synthetic transport disconnected during retirement")

    env.before = refuse_kill
    if operation == "shutdown":
        remote.shutdown_remote_kernels_for_owner("authority")
    else:
        result = run(env, "import time; time.sleep(30)" if operation == "timeout" else "print('forbidden')",
                     tools=() if operation == "transition" else ("read_file",),
                     reset=operation == "reset", timeout=1 if operation == "timeout" else 12)
        assert result["status"] in ("error", "timeout"), result
    count = sum("nohup" in c for c in env.commands)
    refused = run(env, "print('must-not-run')", ())
    assert refused["status"] == "error", refused
    assert sum("nohup" in c for c in env.commands) == count
    env.before = lambda command: None
    # A closed shutdown generation rejects one caller while retiring itself.
    retry = run(env, "print('fresh')", ())
    if operation == "shutdown":
        retry = run(env, "print('fresh')", ())
    assert retry["status"] == "success", retry
    assert not retry["kernel"]["reused"]


def test_cell_result_retires_rpc_before_slow_result_cleanup(rig, monkeypatch):
    env, calls = rig
    read, release, result_read = threading.Event(), threading.Event(), threading.Event()
    marker = env.root / "rpc-read"
    pollers = []

    def delay(command, payload):
        if command.startswith("cat ") and "/req_" in command and payload["output"]:
            pollers.append(threading.current_thread())
            marker.touch()
            read.set()
            assert release.wait(15)

    def cleanup(command):
        if command.startswith("rm -f ") and "/cell_res_" in command:
            result_read.set()
            release.set()
            for poller in pollers:
                poller.join(5)

    env.after, env.before = delay, cleanup
    result = run(env, f"""
import threading, time
from pathlib import Path
from hermes_tools import read_file
def pending():
    try:
        read_file('after-settlement')
    except RuntimeError:
        pass
threading.Thread(target=pending, daemon=True).start()
while not Path({str(marker)!r}).exists():
    time.sleep(0.02)
""")
    assert read.is_set() and result_read.is_set()
    assert result["status"] == "success", result
    assert calls == []


def test_remote_caller_does_not_fall_back_after_unexpected_kernel_error(rig, monkeypatch):
    from tools.code_execution_tool import _execute_remote
    env, _ = rig
    monkeypatch.setattr("tools.code_execution_tool._get_or_create_env", lambda task: (env, "ssh"))

    def uncertain(*args, **kwargs):
        raise RuntimeError("uncertain lifecycle")

    monkeypatch.setattr(remote, "execute_in_remote_kernel", uncertain)
    result = json.loads(_execute_remote("print('must-not-run')", "authority", []))
    assert result["status"] == "error", result
    assert not any("python3 script.py" in c for c in env.commands)


def test_timeout_caller_does_not_claim_unverified_kernel_death(rig):
    import time
    from tools.code_execution_tool import _finish_remote_kernel_result
    env, _ = rig
    assert run(env, "x = 1")["status"] == "success"

    def disconnected(command):
        if "kill -KILL" in command:
            raise OSError("retirement disconnected")

    env.before = disconnected
    raw = run(env, "import time; time.sleep(30)", timeout=1)
    result = json.loads(_finish_remote_kernel_result(raw, timeout=1, exec_start=time.monotonic()))
    assert result["status"] == "timeout"
    assert "quarantined" in result["error"]
    assert "killed" not in result["error"]


@pytest.mark.parametrize("env_type", ["ssh", "docker", "modal"])
def test_approved_real_dispatch_and_explicit_empty_caller(rig, monkeypatch, env_type):
    from tools.code_execution_tool import _execute_remote
    env, _ = rig
    sample = env.root / "approved.txt"
    sample.write_text("approved local fixture", encoding="utf-8")
    monkeypatch.setattr("model_tools.handle_function_call", env.real_dispatch)
    monkeypatch.setattr("tools.code_execution_tool._get_or_create_env", lambda task: (env, env_type))
    first = json.loads(_execute_remote(
        f"from hermes_tools import read_file\nx = 41\nprint(read_file({str(sample)!r}))",
        "authority", ["execute_code", "read_file"]))
    assert first["status"] == "success", first
    assert "approved local fixture" in first["output"]
    assert first["tool_calls_made"] == 1
    same = json.loads(_execute_remote("print(x + 1)", "authority", ["execute_code", "read_file"]))
    assert same["status"] == "success" and same["kernel"]["reused"], same
    assert same["output"].strip() == "42"
    denied = json.loads(_execute_remote("print('x' in globals())\nfrom hermes_tools import read_file",
                                        "authority", []))
    assert denied["status"] == "error", denied
    assert denied["tool_calls_made"] == 0
    assert denied["kernel"]["state_lost"] and not denied["kernel"]["reused"]
    assert "False" in denied["output"]


def test_reset_retires_subprocess_group_as_well_as_runner(rig):
    import psutil
    env, _ = rig
    first = run(env, "import subprocess\np = subprocess.Popen(['python3', '-c', 'import time; time.sleep(60)'])\nprint(p.pid)")
    assert first["status"] == "success", first
    pid = int(first["stdout"])
    result = run(env, "print('p' in globals())", reset=True)
    assert result["status"] == "success", result
    assert result["kernel"]["state_reset"] and not result["kernel"]["reused"]
    assert result["stdout"].strip() == "False"
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_replaced_transport_cannot_reuse_old_kernel(rig):
    env, _ = rig
    assert run(env, "x = 'old'")["status"] == "success"
    replacement = ShellEnv(env.root)
    result = run(replacement, "print('x' in globals())")
    assert result["status"] == "success", result
    assert result["stdout"].strip() == "False"
    assert result["kernel"]["state_lost"] and not result["kernel"]["reused"]


def test_shutdown_during_failed_spawn_cannot_fall_back_to_per_call(rig, monkeypatch):
    from tools.code_execution_tool import _execute_remote
    env, _ = rig
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(remote, "_RPC_DRAIN_TIMEOUT", 0.1)
    monkeypatch.setattr("tools.code_execution_tool._get_or_create_env", lambda task: (env, "ssh"))

    def fail_spawn(command):
        if "nohup" in command:
            entered.set()
            assert release.wait(10)
            raise OSError("disconnected during spawn")

    env.before = fail_spawn
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        cell = pool.submit(_execute_remote, "print('must-not-run')", "authority", [])
        assert entered.wait(5)
        try:
            remote.shutdown_remote_kernels_for_owner("authority")
        finally:
            release.set()
        result = json.loads(cell.result(15))
    assert result["status"] == "error", result
    assert not any("python3 script.py" in c for c in env.commands)


def test_admitted_callback_retains_authenticated_context_after_caller_changes(rig, monkeypatch):
    from gateway.session_context import get_session_env, set_session_vars
    from tools.approval_context import set_current_session_key
    from tools.terminal_tool import _get_approval_callback, set_approval_callback
    env, _ = rig
    release, admitted = threading.Event(), threading.Event()
    marker = env.root / "host-admitted"
    pollers, seen = [], []
    callbacks = {actor: (lambda: None) for actor in ("actor-a", "actor-b")}
    monkeypatch.setattr(remote, "_RPC_DRAIN_TIMEOUT", 0.1)

    def snapshot():
        return (get_session_env("HERMES_SESSION_USER_ID"),
                get_session_env("HERMES_SESSION_CHAT_ID"),
                get_session_env("HERMES_SESSION_PROFILE"), _get_approval_callback())

    def blocked(name, args, task_id=None):
        pollers.append(threading.current_thread())
        seen.append(snapshot())
        marker.touch()
        admitted.set()
        assert release.wait(15)
        seen.append(snapshot())
        return json.dumps({"ok": True})

    def invoke(actor, code):
        def bound():
            set_current_session_key("stable-owner")
            set_session_vars(user_id=actor, chat_id="room-" + actor, profile="profile-" + actor)
            prior = _get_approval_callback()
            set_approval_callback(callbacks[actor])
            try:
                return run(env, code)
            finally:
                set_approval_callback(prior)
        return contextvars.copy_context().run(bound)

    monkeypatch.setattr("model_tools.handle_function_call", blocked)
    try:
        first = invoke("actor-a", f"""
import threading, time
from pathlib import Path
from hermes_tools import read_file
def pending():
    try:
        read_file('admitted')
    except RuntimeError:
        pass
threading.Thread(target=pending, daemon=True).start()
while not Path({str(marker)!r}).exists():
    time.sleep(0.02)
""")
        assert admitted.is_set() and first["status"] == "error", first
        second = invoke("actor-b", "print('must-not-run')")
        assert second["status"] == "error" and "draining" in second["error"], second
    finally:
        release.set()
    for poller in pollers:
        poller.join(5)
        assert not poller.is_alive()
    assert seen == [("actor-a", "room-actor-a", "profile-actor-a", callbacks["actor-a"])] * 2
    assert invoke("actor-b", "print('fresh')")["status"] == "success"


def live(pid):
    import psutil
    return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE


@pytest.mark.parametrize('failure', ['liveness', 'startup_reply'])
def test_uncertain_partial_spawn_must_not_forget_process_and_fall_back(rig, monkeypatch, failure):
    import time
    import psutil
    from tools.code_execution_tool import _execute_remote
    env, calls = rig
    monkeypatch.setattr('tools.code_execution_tool._get_or_create_env', lambda task: (env, 'ssh'))
    spawned = []
    broken = [False]

    def after(command, payload):
        if 'nohup' in command:
            pid = int(next(line[4:] for line in payload['output'].splitlines() if line.startswith('PID:')))
            spawned.append(pid)
            # Prove the detached process entered its actual Python runner before
            # simulating lost transport reads/cleanup, without changing bootstrap.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if live(pid) and 'kernel_runner.py' in psutil.Process(pid).cmdline():
                    break
                time.sleep(.01)
            assert live(pid)
            broken[0] = True
            if failure == 'startup_reply':
                raise OSError('synthetic lost successful startup reply')

    def before(command):
        if broken[0] and ((command.startswith('kill -0 ')) or
                         (command.startswith('rm -rf ') and '/hermes_rkernel_' in command)):
            raise OSError('synthetic transport cannot probe or remove partial runner')

    env.before, env.after = before, after
    try:
        first = json.loads(_execute_remote("print('fallback-ran')", 'authority', []))
        original = spawned[0]
        observed = {'first_result': first, 'original_alive': live(original),
                    'tracked_kernels': len(remote._REMOTE_KERNELS),
                    'tracked_lifetimes': len(remote._REMOTE_LIFETIMES),
                    'fallback_executed': any('python3 script.py' in c for c in env.commands)}
        env.before = lambda command: None
        env.after = lambda command, payload: None
        second = run(env, "print('replacement-ran')", ())
        observed.update(second_result=second, original_alive_with_replacement=live(original),
                        replacement_pids=[k.pid for k in remote._REMOTE_KERNELS.values()])
        print('PARTIAL_SPAWN_OBSERVED', json.dumps(observed, sort_keys=True))
        assert first['status'] == 'error', observed
        assert not observed['fallback_executed'], observed
        assert not (observed['original_alive_with_replacement'] and second['status'] == 'success'), observed
    finally:
        env.before = lambda command: None
        env.after = lambda command, payload: None
        # Unknown orphan lifetimes are deliberately retained as evidence until
        # the prescribed bwrap PID namespace exits and reaps its processes.
        # Do not disable the repository's live-process guard to clean them up.
        pass


def test_definite_prelaunch_failure_preserves_initial_legacy_fallback(rig, monkeypatch):
    from tools.code_execution_tool import _execute_remote
    env, calls = rig
    monkeypatch.setattr('tools.code_execution_tool._get_or_create_env', lambda task: (env, 'ssh'))
    def before(command):
        if command.startswith('echo ') and '/kernel_runner.py' in command:
            raise OSError('synthetic bootstrap shipment rejection before launch')
    env.before = before
    result = json.loads(_execute_remote("from hermes_tools import read_file\nprint(read_file('initial-approved'))", 'authority', ['read_file']))
    print('INITIAL_FALLBACK_OBSERVED', json.dumps(result))
    assert result['status'] == 'success', result
    assert [c[1]['path'] for c in calls] == ['initial-approved']
    assert not remote._REMOTE_KERNELS


def test_publication_inflight_at_shutdown_is_accounted_and_host_rpc_revoked(rig, monkeypatch):
    import time
    env, calls = rig
    assert run(env, 'x = 1')['status'] == 'success'
    entered, release = threading.Event(), threading.Event()
    marker = env.root / 'publication-executed'
    monkeypatch.setattr(remote, '_RPC_DRAIN_TIMEOUT', .1)

    def before(command):
        if command.startswith('mv ') and 'cell_req_000002.json.tmp' in command:
            entered.set()
            assert release.wait(10)

    def after(command, payload):
        if command.startswith('mv ') and 'cell_req_000002.json.tmp' in command:
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert marker.exists(), 'prove runner executed published cell before transport acknowledgement'

    env.before, env.after = before, after
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        first = pool.submit(run, env, f"from pathlib import Path\nPath({str(marker)!r}).touch()\nfrom hermes_tools import read_file\nread_file('revoked')")
        assert entered.wait(5)
        try:
            remote.shutdown_remote_kernels_for_owner('authority')
            waiting = run(env, "print('must-not-run')", (), timeout=0)
            assert waiting['status'] == 'error'
            assert remote._REMOTE_LIFETIMES
        finally:
            release.set()
        result = first.result(15)
    print('PUBLICATION_OBSERVED', json.dumps({'executed_after_shutdown_return': marker.exists(), 'result': result, 'host_calls': calls}))
    assert result['status'] == 'error'
    assert calls == []
    assert not remote._REMOTE_KERNELS
    env.before = lambda command: None
    env.after = lambda command, payload: None
    assert run(env, "print('fresh')", ())['status'] == 'success'


def test_saved_async_task_cannot_borrow_next_cell_but_fresh_async_call_works(rig):
    env, calls = rig
    first = run(env, """
import asyncio
from hermes_tools import read_file
loop = asyncio.new_event_loop()
release = asyncio.Event()
async def stale():
    await release.wait()
    try:
        read_file('stale-async')
    except RuntimeError:
        return 'retired'
task = loop.create_task(stale())
loop.run_until_complete(asyncio.sleep(0))
""")
    assert first['status'] == 'success', first
    second = run(env, """
release.set()
print(loop.run_until_complete(task))
async def fresh():
    return read_file('fresh-async')
print(loop.run_until_complete(fresh()))
loop.close()
""")
    assert second['status'] == 'success', second
    assert 'retired' in second['stdout']
    assert [c[1]['path'] for c in calls] == ['fresh-async']
    assert second['kernel']['reused']


@pytest.mark.parametrize("reply", ["lost", "missing", "malformed", "zero", "unicode", "nonzero", "liveness"])
def test_partial_launch_quarantines_failed_retirement_then_recovers(rig, monkeypatch, reply):
    import time
    from tools.code_execution_tool import _execute_remote
    env, calls = rig
    monkeypatch.setattr("tools.code_execution_tool._get_or_create_env", lambda task: (env, "ssh"))
    spawned = []
    broken = [False]
    tracked_before_dispatch = []

    def before(command):
        if "nohup" in command:
            tracked_before_dispatch.append(bool(remote._REMOTE_KERNELS and remote._REMOTE_LIFETIMES))
        if broken[0] and (command.startswith("kill -0 ") or "kill -KILL" in command
                          or "launch" in command or command.startswith("rm -rf ")):
            raise OSError("synthetic retirement transport unavailable")

    def after(command, payload):
        if "nohup" not in command:
            return
        pid = int(next(line[4:] for line in payload["output"].splitlines() if line.startswith("PID:")))
        spawned.append(pid)
        deadline = time.monotonic() + 5
        # Runner receipt, when supported, is independent of the startup reply.
        while not live(pid) and time.monotonic() < deadline:
            time.sleep(.01)
        assert live(pid)
        broken[0] = True
        if reply == "lost":
            raise OSError("lost startup response")
        if reply in ("missing", "malformed", "zero", "unicode"):
            payload["output"] = {"missing": "", "malformed": "PID:bogus", "zero": "PID:0", "unicode": "PID:²"}[reply]
        if reply == "nonzero":
            payload["returncode"] = 1

    env.before, env.after = before, after
    first = json.loads(_execute_remote("print('must-not-fallback')", "authority", []))
    original = spawned[0]
    try:
        assert first["status"] == "error", first
        assert tracked_before_dispatch == [True], "accounting must precede ambiguous launch dispatch"
        assert live(original)
        assert len(remote._REMOTE_KERNELS) == len(remote._REMOTE_LIFETIMES) == 1
        for reset in (False, True):
            refused = run(env, "print('must-not-run')", (), reset=reset)
            assert refused["status"] == "error", refused
            assert len(spawned) == 1 and live(original)
        assert not any("python3 script.py" in c for c in env.commands)
    finally:
        env.before = lambda command: None
        env.after = lambda command, payload: None
    recovered = run(env, "x = 41\nfrom hermes_tools import read_file\nprint(read_file('recovered'))")
    assert recovered["status"] == "success", recovered
    assert not live(original), "no replacement until targeted retirement is verified"
    same = run(env, "print(x + 1, read_file('same'))")
    assert same["status"] == "success" and same["kernel"]["reused"], same
    assert [c[1]["path"] for c in calls] == ["recovered", "same"]


def test_lost_launch_with_delayed_dispatch_is_fenced_before_replacement(rig, monkeypatch):
    from tools.code_execution_tool import _execute_remote
    env, calls = rig
    monkeypatch.setattr("tools.code_execution_tool._get_or_create_env", lambda task: (env, "ssh"))
    delayed = []

    def before(command):
        if "nohup" in command:
            delayed.append(command)
            raise OSError("transport timed out but command delivery is still pending")

    env.before = before
    result = json.loads(_execute_remote("print('not-fallback')", "authority", []))
    assert result["status"] == "error", result
    assert remote._REMOTE_KERNELS and remote._REMOTE_LIFETIMES
    env.before = lambda command: None
    recovered = run(env, "x = 'replacement'", ())
    assert recovered["status"] == "success", recovered
    # Deliver the ORIGINAL command after recovery, not a newly invented launch.
    late = env.execute(delayed[0], cwd="/", timeout=10)
    assert "PID:" not in late["output"], "retirement must fence delayed launch, not assume it vanished"
    assert run(env, "print(x)", ())["stdout"].strip() == "replacement"
    assert calls == []
