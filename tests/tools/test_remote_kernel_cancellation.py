"""Cancellation/retirement boundaries using real generated code and local carriers.

Transport substitution is not production SSH proof. Event/receipt barriers
establish when user code is running; control-flow exceptions must propagate.
"""
import json
import os
import signal
import subprocess
import threading
import time

import pytest

from tests.tools.test_remote_kernel_authority import ShellEnv
from tools import code_execution_tool as execution
from tools import code_kernel_remote as remote
from tools.environments.base import BaseEnvironment

pytestmark = pytest.mark.linux_only


def wait_marker(path):
    deadline = time.monotonic() + 6
    while not path.exists() and time.monotonic() < deadline:
        threading.Event().wait(.01)
    assert path.exists(), 'real generated user code must reach the named barrier'

def invoke(code):
    return json.loads(execution._execute_remote(code, 'quality-owner', ['read_file']))

@pytest.fixture
def rig(tmp_path, monkeypatch):
    remote.shutdown_all_remote_kernels()
    env = ShellEnv(tmp_path)
    monkeypatch.setattr(execution, '_get_or_create_env', lambda task: (env, 'ssh'))
    monkeypatch.setattr(execution, '_load_config', lambda: {'timeout': 5, 'max_tool_calls': 5})
    calls = []
    gate: dict = {'block': None, 'started': None}  # optional Events: dispatch blocks until set
    def dispatch(name, args, task_id=None, **kw):
        calls.append(args['path'])
        if gate['block'] is not None:
            gate['started'].set()
            gate['block'].wait(20)
        return json.dumps({'path': args['path']})
    monkeypatch.setattr('model_tools.handle_function_call', dispatch)
    yield env, calls, gate
    env.before = lambda command: None
    env.after = lambda command, result: None
    if gate['block'] is not None:
        gate['block'].set()
    remote.shutdown_all_remote_kernels()

@pytest.mark.parametrize('failure', ['OSError', 'KeyboardInterrupt', 'SystemExit'])
def test_cancelled_persistent_cell_cannot_be_reused(rig, failure):
    env, calls, _ = rig
    started = env.root / 'started'
    release = env.root / 'release'
    fault = {'OSError': OSError, 'KeyboardInterrupt': KeyboardInterrupt, 'SystemExit': SystemExit}[failure]
    def interrupt(command, result):
        if command.startswith('mv ') and 'cell_req_000001.json.tmp' in command:
            wait_marker(started)
            raise fault('synthetic interrupted transport after real cell started')
    env.after = interrupt
    code = (
        'from pathlib import Path\nimport time\n'
        f'Path({str(started)!r}).touch()\n'
        f'while not Path({str(release)!r}).exists(): time.sleep(.01)\n'
        "tainted_by_cancelled_cell = True\n"
    )
    caught = None
    try:
        first = invoke(code)
    except (KeyboardInterrupt, SystemExit) as exc:
        caught = type(exc).__name__
        first = None
    old = next(iter(remote._REMOTE_KERNELS.values()), None)
    observation = {'failure': failure, 'caught': caught, 'first': first,
                   'retained_kernel': old is not None,
                   'retiring': old.retiring if old else None,
                   'alive_after_cancel': remote._is_alive(old) if old else False}
    env.after = lambda command, result: None
    release.touch()
    second = invoke("from hermes_tools import read_file\nprint('tainted_by_cancelled_cell' in globals(), read_file('fresh'))")
    observation.update(second=second, calls=calls)
    assert second['status'] == 'success', observation
    assert calls == ['fresh'], observation
    assert second['kernel']['reused'] is False and second['output'].startswith('False'), observation


def test_positive_useful_persistence_and_empty_authority(rig):
    env, calls, _ = rig
    first = invoke("from hermes_tools import read_file\nx=41\nwrapper_id=id(read_file)\nprint(read_file('first'))")
    second = invoke("x+=1\nassert id(read_file)==wrapper_id\nprint(x, read_file('second'))")
    assert first['status'] == second['status'] == 'success', (first, second)
    assert second['kernel']['reused'] is True and second['output'].startswith('42'), second
    assert calls == ['first', 'second']
    empty = json.loads(execution._execute_remote("print('x' in globals())\nfrom hermes_tools import read_file", 'quality-owner', []))
    assert empty['status'] == 'error' and empty['output'].startswith('False'), empty
    assert calls == ['first', 'second']


@pytest.mark.parametrize('verified', [False, True])
def test_fallback_unverified_timeout_must_not_claim_killed(rig, monkeypatch, verified):
    env, calls, _ = rig
    started = env.root / 'fallback-started'
    release = env.root / 'fallback-release'
    original_execute = env.execute
    launched = None
    refuse_retirement = not verified
    failed_prelaunch = False
    def transport(command, cwd=None, timeout=None):
        nonlocal launched, failed_prelaunch
        if not failed_prelaunch and command.startswith('mkdir -p ') and 'hermes_rkernel_' in command:
            failed_prelaunch = True
            raise OSError('synthetic definite prelaunch failure')
        if 'python3 script.py' in command:
            launched = subprocess.Popen(['/bin/bash', '-c', command], cwd=cwd,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wait_marker(started)
            return {'output':'', 'returncode':124}
        if refuse_retirement and 'kill -KILL' in command:
            raise OSError('synthetic retirement transport unavailable')
        return original_execute(command, cwd=cwd, timeout=timeout)
    monkeypatch.setattr(env, 'execute', transport)
    code = ('from pathlib import Path\nimport time\n' + f'Path({str(started)!r}).touch()\n'
            + f'while not Path({str(release)!r}).exists(): time.sleep(.01)\n')
    try:
        result = invoke(code)
        old = next(iter(remote._REMOTE_KERNELS.values()), None)
        alive = remote._is_alive(old) if old else False
        blocked = invoke("print('must not run')")
        observation = {'result':result,'alive':alive,'retiring':old.retiring if old else None,'blocked':blocked}
        if verified:
            assert old is None and blocked['status'] == 'success', observation
            assert 'was killed' in result.get('error', ''), observation
        else:
            assert old is not None and alive and old.retiring and blocked['status']=='error', observation
            assert 'was killed' not in result.get('error',''), observation
            assert 'was killed' not in result['output'], observation
    finally:
        refuse_retirement = False
        release.touch()
        remote.shutdown_all_remote_kernels()
        if launched is not None:
            launched.wait(8)


@pytest.mark.parametrize('route', ['persistent', 'fallback'])
@pytest.mark.parametrize('failure', [KeyboardInterrupt, SystemExit])
def test_cancel_during_retirement_releases_lease_and_preserves_exception(rig, monkeypatch, route, failure):
    env, calls, _ = rig
    started = env.root / 'cleanup-started'
    release = env.root / 'cleanup-release'
    original_execute = env.execute
    launched = None
    failed_prelaunch = False
    refuse = True
    cleanup_interrupted = False
    original = failure('original cancellation')

    def transport(command, cwd=None, timeout=None):
        nonlocal launched, failed_prelaunch, cleanup_interrupted
        if route == 'fallback' and not failed_prelaunch and command.startswith('mkdir -p ') and 'hermes_rkernel_' in command:
            failed_prelaunch = True
            raise OSError('definite prelaunch failure')
        if refuse and 'kill -KILL' in command:
            if not cleanup_interrupted:
                cleanup_interrupted = True
                raise SystemExit('secondary interrupted cleanup')
            raise OSError('retirement transport still unavailable')
        if route == 'fallback' and 'python3 script.py' in command:
            launched = subprocess.Popen(['/bin/bash', '-c', command], cwd=cwd,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wait_marker(started)
            raise original
        result = original_execute(command, cwd=cwd, timeout=timeout)
        if refuse and route == 'persistent' and command.startswith('mv ') and 'cell_req_000001.json.tmp' in command:
            wait_marker(started)
            raise original
        return result

    monkeypatch.setattr(env, 'execute', transport)
    try:
        with pytest.raises(failure) as caught:
            invoke('from pathlib import Path\nimport time\n'
                   + f'Path({str(started)!r}).touch()\n'
                   + f'while not Path({str(release)!r}).exists(): time.sleep(.01)\n')
        assert caught.value is original
        old = next(iter(remote._REMOTE_KERNELS.values()))
        slot = next(iter(remote._REMOTE_LIFETIMES.values()))
        assert old.retiring and remote._is_alive(old)
        assert slot.users == 0 and not slot.lock.locked()
        assert slot.authority is not None and slot.authority.stop.is_set()
        assert invoke("print('must not run')")['status'] == 'error'
        assert calls == []
    finally:
        refuse = False
        release.touch()
        remote.shutdown_all_remote_kernels()
        if launched is not None:
            launched.wait(8)
    assert invoke("from hermes_tools import read_file\nprint(read_file('recovered'))")['status'] == 'success'
    assert calls == ['recovered']


@pytest.mark.parametrize('route', ['persistent', 'fallback'])
@pytest.mark.parametrize('failure', [KeyboardInterrupt, SystemExit])
def test_cancel_during_rpc_dispatch_first_call_after_drain_is_admitted(rig, monkeypatch, route, failure):
    """K-I1: an admitted host tool keeps the slot (poller alive) -> refuse with the
    draining message; once the poller drains, the FIRST same-owner call must be
    admitted with fresh state. No 'was shut down' refusal from the stale slot."""
    env, calls, gate = rig
    gate['block'] = threading.Event()
    gate['started'] = threading.Event()
    original_execute = env.execute
    launched = None
    failed_prelaunch = False
    fired = False

    def transport(command, cwd=None, timeout=None):
        nonlocal launched, failed_prelaunch, fired
        if route == 'fallback' and not failed_prelaunch and command.startswith('mkdir -p ') and 'hermes_rkernel_' in command:
            failed_prelaunch = True
            raise OSError('definite prelaunch failure')
        if route == 'fallback' and 'python3 script.py' in command:
            launched = subprocess.Popen(['/bin/bash', '-c', command], cwd=cwd,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            assert gate['started'].wait(10), 'dispatch must be in flight'
            raise failure('cancelled while RPC dispatch in flight')
        result = original_execute(command, cwd=cwd, timeout=timeout)
        if (route == 'persistent' and not fired and command.startswith('cat ')
                and 'cell_res_000001.json' in command and gate['started'].is_set()):
            fired = True
            raise failure('cancelled while RPC dispatch in flight')
        return result

    monkeypatch.setattr(env, 'execute', transport)
    code = "from hermes_tools import read_file\nread_file('during')\nafter_dispatch = True\n"
    poller = None
    try:
        with pytest.raises(failure):
            invoke(code)
        slot = next(iter(remote._REMOTE_LIFETIMES.values()))
        poller = slot.poller
        obs = {'route': route, 'poller_alive': poller is not None and poller.is_alive(),
               'users': slot.users, 'locked': slot.lock.locked(), 'closed': slot.closed.is_set()}
        assert poller is not None and poller.is_alive(), obs
        assert slot.users == 0 and not slot.lock.locked(), obs
        refused = invoke("print('overlap')")
        obs['refused'] = refused
        assert refused['status'] == 'error' and 'overlap' not in refused.get('output', ''), obs
    finally:
        gate['block'].set()
        if launched is not None:
            launched.wait(8)
    assert poller is not None
    poller.join(10)
    assert not poller.is_alive()
    fresh = invoke("print('after_dispatch' in globals())")
    obs['first_after_drain'] = fresh
    assert fresh['status'] == 'success', (fresh.get('error'), obs)
    assert fresh['kernel']['reused'] is False, obs
    assert fresh['output'].startswith('False'), obs
    assert calls == ['during'], calls
    # The drain-window refusal must name the real reason, not the stale slot.
    assert 'still draining' in refused['error'], obs


class NamespaceCarrier(BaseEnvironment):
    """Only the remote carrier is replaced by local Bash; execute/wait are real."""
    def __init__(self, root):
        self.root=root
        self.held=threading.Event()
        self.hold=True
        self.killed=[]
        super().__init__(str(root),10)
        self._prefer_nonlogin=True
    def get_temp_dir(self): return str(self.root)
    def cleanup(self): pass
    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        held=self.hold and 'cell_res_000001.json' in cmd_string and 'cat ' in cmd_string
        if held: cmd_string += '; sleep 30'
        proc=subprocess.Popen(['/bin/bash','-c',cmd_string],stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,text=True,encoding='utf-8',
            start_new_session=True)
        if held: self.held.set()
        return proc
    def _kill_process(self,proc):
        self.killed.append(getattr(proc, 'pid', None))
        super()._kill_process(proc)


def test_real_sigint_propagates_public_remote_path_and_retires_cell(tmp_path,monkeypatch):
    remote.shutdown_all_remote_kernels()
    env=NamespaceCarrier(tmp_path)
    monkeypatch.setattr(execution,'_get_or_create_env',lambda task:(env,'ssh'))
    monkeypatch.setattr(execution,'_load_config',lambda:{'timeout':8,'max_tool_calls':5})
    monkeypatch.setattr('tools.terminal_tool._get_env_config',lambda:{'env_type':'ssh'})
    monkeypatch.setattr('tools.terminal_tool._docker_has_host_access',lambda cfg:False)
    monkeypatch.setattr('tools.approval.check_execute_code_guard',lambda *a,**kw:{'approved':True})
    monkeypatch.setattr('tools.process_registry._is_supervised_gateway_process',lambda:False)
    calls=[]
    def dispatch(name,args,**kw):
        calls.append(args['path']);return json.dumps({'path':args['path']})
    monkeypatch.setattr('model_tools.handle_function_call',dispatch)
    started=tmp_path/'started';release=tmp_path/'release'
    delivered=threading.Event(); finished=threading.Event()
    def sender():
        deadline=time.monotonic()+10
        while not (started.exists() and env.held.is_set()) and time.monotonic()<deadline:
            threading.Event().wait(.01)
        if started.exists() and env.held.is_set():
            delivered.set()
            os.kill(os.getpid(),signal.SIGINT)
        finished.set()
    signaler=threading.Thread(target=sender,daemon=True)
    original=signal.signal(signal.SIGINT,signal.default_int_handler)
    signaler.start()
    caught=False
    try:
        code=('from pathlib import Path\nimport time\n'+f'Path({str(started)!r}).touch()\n'
            +f'while not Path({str(release)!r}).exists(): time.sleep(.01)\n'
            +'cancelled_state=True\n')
        try: execution.execute_code(code,task_id='sigint-owner',enabled_tools=['read_file'])
        except KeyboardInterrupt: caught=True
        assert delivered.is_set() and finished.wait(5) and caught
        old=next(iter(remote._REMOTE_KERNELS.values()),None)
        env.hold=False
        observation={'caught_real_sigint':caught,'base_execute_used':env.execute.__func__ is BaseEnvironment.execute,
            'base_wait_used':env._wait_for_process.__func__ is BaseEnvironment._wait_for_process,
            'transport_process_kill_called':bool(env.killed),'kernel_retained':old is not None,
            'retiring':old.retiring if old else None,'old_alive':remote._is_alive(old) if old else False}
        release.touch()
        result=json.loads(execution.execute_code("from hermes_tools import read_file\nprint('cancelled_state' in globals(),read_file('fresh'))",
            task_id='sigint-owner',enabled_tools=['read_file']))
        observation.update(result=result,calls=calls)
        assert result['status']=='success' and calls==['fresh'],observation
        assert not result['kernel']['reused'] and result['output'].startswith('False'),observation
    finally:
        env.hold=False;release.touch();signaler.join(12)
        signal.signal(signal.SIGINT,original)
        remote.shutdown_all_remote_kernels()
