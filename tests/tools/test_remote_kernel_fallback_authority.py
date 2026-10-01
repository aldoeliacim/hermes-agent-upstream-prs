"""Actual per-call scripts and file RPC share the remote lifetime ledger.

Run only with the canonical OS-confined runner. Transport and host tools are
synthetic; neither the lifecycle nor generated RPC implementation is mocked.
"""
import concurrent.futures
import json
import threading
import time

import pytest

from tools import code_execution_tool as execution
from tools import code_kernel_remote as remote
from tests.tools.test_remote_kernel_authority import ShellEnv

pytestmark = pytest.mark.linux_only


@pytest.fixture
def fallback_rig(tmp_path, monkeypatch):
    remote.shutdown_all_remote_kernels()
    env = ShellEnv(tmp_path)
    config = {"timeout": 12, "max_tool_calls": 5}
    monkeypatch.setattr(execution, "_get_or_create_env", lambda task: (env, "ssh"))
    monkeypatch.setattr(execution, "_load_config", lambda: config.copy())
    admitted = threading.Event()
    release = threading.Event()
    drained = threading.Event()
    calls = []
    failed = False

    def before(command):
        nonlocal failed
        if not failed and command.startswith("mkdir -p ") and "hermes_rkernel_" in command:
            failed = True
            raise OSError("definite prelaunch preparation failure")

    def dispatch(name, args, task_id=None):
        calls.append((task_id, args["path"]))
        if args["path"] == "old":
            admitted.set()
            assert release.wait(35), "test must release admitted host operation"
            drained.set()
        return json.dumps({"path": args["path"]})

    env.before = before
    monkeypatch.setattr("model_tools.handle_function_call", dispatch)
    yield env, config, admitted, release, drained, calls
    release.set()
    env.before = lambda command: None
    remote.shutdown_all_remote_kernels()


def execute(code, owner="fallback", tools=("read_file",), reset=False):
    return json.loads(execution._execute_remote(code, owner, list(tools), reset=reset))


OLD = "from hermes_tools import read_file\nprint(read_file('old'))"
NEW = "from hermes_tools import read_file\nprint(read_file('new'))"


@pytest.mark.parametrize("transition", ["reuse", "reset", "authority", "owner", "all"])
def test_fallback_admission_shutdown_and_useful_recovery(fallback_rig, transition):
    env, config, admitted, release, drained, calls = fallback_rig
    code = OLD + "\nprint(read_file('after-shutdown'))" if transition in ("owner", "all") else OLD
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        old = pool.submit(execute, code)
        assert admitted.wait(10), "useful actual fallback RPC must execute"
        try:
            if transition == "owner":
                remote.shutdown_remote_kernels_for_owner("fallback")
            elif transition == "all":
                remote.shutdown_all_remote_kernels()
            denied = execute(NEW, tools=() if transition == "authority" else ("read_file",),
                             reset=transition == "reset")
            assert denied["status"] == "error", denied
            assert calls == [("fallback", "old")], calls
            # A disjoint owner remains useful while old host RPC is blocked.
            other = execute(NEW, owner="unrelated")
            assert other["status"] == "success", other
        finally:
            release.set()
        original = old.result(20)
    assert drained.is_set()
    assert ("fallback", "after-shutdown") not in calls
    if transition not in ("owner", "all"):
        assert original["status"] == "success" and original["tool_calls_made"] == 1, original
    fresh = execute("from hermes_tools import read_file\nx=41\nprint(read_file('fresh'))")
    assert fresh["status"] == "success", fresh
    reused = execute("x+=1\nprint(x, read_file('reused'))")
    assert reused["status"] == "success" and "42" in reused["output"], reused
    assert reused["kernel"]["reused"]
    empty = execute("print('x' in globals())\nfrom hermes_tools import read_file", tools=())
    assert empty["status"] == "error" and "False" in empty["output"], empty


def test_caller_timeout_keeps_admitted_host_rpc_quarantined(fallback_rig):
    env, config, admitted, release, drained, calls = fallback_rig
    config["timeout"] = 2
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        old = pool.submit(execute, OLD)
        assert admitted.wait(10)
        try:
            timed_out = old.result(15)
            assert timed_out["status"] in ("error", "timeout"), timed_out
            assert not drained.is_set()
            blocked = execute(NEW, reset=True)
            assert blocked["status"] == "error", blocked
            assert calls == [("fallback", "old")]
            remote.shutdown_remote_kernels_for_owner("fallback")
            blocked = execute(NEW)
            assert blocked["status"] == "error", blocked
        finally:
            release.set()
    assert drained.wait(10)
    # Explicit shutdown waits for the actual registered poller to finish.
    slot = next(iter(remote._REMOTE_LIFETIMES.values()), None)
    if slot is not None and slot.poller is not None:
        slot.poller.join(10)
    remote.shutdown_remote_kernels_for_owner("fallback")
    config["timeout"] = 12
    fresh = execute(NEW)
    assert fresh["status"] == "success", fresh


@pytest.mark.parametrize("reset,tools", [(True, ("read_file",)), (False, ())])
def test_queued_before_fallback_choice_cannot_overlap(fallback_rig, reset, tools):
    env, config, admitted, release, drained, calls = fallback_rig
    preparing = threading.Event()
    choose_fallback = threading.Event()
    before = env.before

    def held_preparation(command):
        if command.startswith("mkdir -p ") and "hermes_rkernel_" in command and not preparing.is_set():
            preparing.set()
            assert choose_fallback.wait(15)
        before(command)

    env.before = held_preparation
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        old = pool.submit(execute, OLD)
        assert preparing.wait(10)
        newer = pool.submit(execute, "print('fresh')", "fallback", tools, reset)
        deadline = time.monotonic() + 5
        queued = False
        while time.monotonic() < deadline:
            with remote._REMOTE_KERNELS_LOCK:
                queued = any(slot.users == 2 for slot in remote._REMOTE_LIFETIMES.values())
            if queued:
                break
            threading.Event().wait(0.01)
        assert queued, "second caller must actually queue before fallback choice"
        choose_fallback.set()
        assert admitted.wait(10)
        try:
            other = execute(NEW, owner="unrelated")
            assert other["status"] == "success", other
            assert calls == [("fallback", "old"), ("unrelated", "new")]
        finally:
            release.set()
        assert old.result(20)["status"] == "success"
        result = newer.result(20)
        assert result["status"] == "success", result
    assert drained.is_set()
    fresh = execute("print('fresh')", reset=reset, tools=tools)
    assert fresh["status"] == "success", fresh


@pytest.mark.parametrize("route", ["persistent", "fallback"])
@pytest.mark.parametrize("capability", ["supported", "missing", "error", "ineffective", "noncallable", "missing_getpgid"])
def test_generated_target_requires_remote_posix_session(fallback_rig, monkeypatch, route, capability):
    # Fault injection is ONLY in a spawned target Python, never the host OS.
    import shlex
    import sys
    from pathlib import Path

    env, config, admitted, release, drained, calls = fallback_rig
    config["timeout"] = 3
    if route == "persistent":
        env.before = lambda command: None
    gate = env.root / "capability"
    gate.mkdir()
    marker = gate / "user-ran"
    diagnostic = gate / "error"
    target = gate / "target.py"
    target.write_text(
        "import os, pathlib, runpy, sys\n"
        "fault, diagnostic, script = sys.argv[1:]\n"
        "if fault == 'missing': del os.setsid\n"  # windows-footgun: ok — generated remote-POSIX target
        "if fault == 'missing_getpgid': del os.getpgid\n"
        "if fault == 'noncallable': os.setsid = None\n"  # windows-footgun: ok — generated remote-POSIX target
        "if fault == 'ineffective': os.setsid = lambda: None\n"  # windows-footgun: ok — generated remote-POSIX target
        "if fault == 'error':\n"
        "    def denied(): raise OSError('synthetic setsid refusal')\n"
        "    os.setsid = denied\n"  # windows-footgun: ok — generated remote-POSIX target
        "sys.path.insert(0, os.getcwd())\n"
        "try: runpy.run_path(script, run_name='__main__')\n"
        "except BaseException as exc:\n"
        "    pathlib.Path(diagnostic).write_text(str(exc), encoding='utf-8')\n"
        "    raise\n", encoding="utf-8",
    )
    directory = gate / "bin"
    directory.mkdir()
    shim = directory / "python3"
    shim.write_text(
        "#!/bin/bash\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(target))
        + " " + shlex.quote(capability) + " " + shlex.quote(str(diagnostic)) + ' "$1"\n',
        encoding="utf-8",
    )
    shim.chmod(0o700)
    original_execute = env.execute
    def launch_with_capability(command, cwd=None, timeout=None):
        if "nohup" in command or "python3 script.py" in command:
            command = "export PATH=" + shlex.quote(str(directory)) + ':"$PATH"; ' + command
        return original_execute(command, cwd=cwd, timeout=timeout)
    monkeypatch.setattr(env, "execute", launch_with_capability)
    code = (
        "import os\nfrom pathlib import Path\nfrom hermes_tools import read_file\n"
        + f"Path({str(marker)!r}).touch()\n"
        + "print(os.getpid() == os.getpgid(0))\nx=73\nwrapper_id=id(read_file)\nprint(read_file('capability'))"
    )
    result = execute(code)
    if capability == "supported":
        assert result["status"] == "success" and "True" in result["output"], result
        assert marker.exists() and calls == [("fallback", "capability")]
        if route == "persistent":
            reused = execute("assert x==73 and id(read_file)==wrapper_id\nprint(read_file('reused'))")
            assert reused["status"] == "success" and reused["kernel"]["reused"], reused
            assert calls == [("fallback", "capability"), ("fallback", "reused")]
    else:
        assert result["status"] in ("error", "timeout"), result
        assert not marker.exists() and calls == [], (result, calls)
        assert diagnostic.exists(), result
        assert "Remote POSIX process-group capability" in diagnostic.read_text(encoding="utf-8")
    remote.shutdown_remote_kernels_for_owner("fallback")
    assert not remote._REMOTE_KERNELS and not remote._REMOTE_LIFETIMES
