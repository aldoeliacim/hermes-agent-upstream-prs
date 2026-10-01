"""Process handles are not bearer tokens: test the real model-tool boundary."""
import contextvars
import json
import time

import pytest

from gateway import session_context as sc
from tools import approval
from tools import process_registry as pr


ACTIONS = ("poll", "log", "wait", "kill", "write", "submit", "close")


class SyntheticPty:
    """A byte sink instead of a real user's terminal or a killable PID."""
    def __init__(self):
        self.events = []

    def write(self, data):
        self.events.append(("write", data))

    def sendeof(self):
        self.events.append(("eof",))

    def terminate(self, force=False):
        self.events.append(("terminate", force))


def context(contact="a", *, profile="default", source="", platform="whatsapp",
            key=None, session_id=None):
    ctx = contextvars.Context()
    ctx.run(sc.set_session_vars, platform=platform, source=source,
            chat_id=f"contact-{contact}", user_id=f"contact-{contact}",
            session_key=key if key is not None else f"chat-{contact}",
            session_id=session_id or f"conversation-{contact}", profile=profile)
    return ctx


@pytest.fixture
def registry(monkeypatch, tmp_path):
    monkeypatch.setattr(sc, "_session_context_engaged", False)
    for key in tuple(sc._VAR_MAP) + ("_HERMES_GATEWAY", "HERMES_GATEWAY_SESSION", "HERMES_PLATFORM"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(pr, "CHECKPOINT_PATH", tmp_path / "processes.json")
    reg = pr.ProcessRegistry()
    monkeypatch.setattr(pr, "process_registry", reg)
    return reg


def add(registry, ctx, contact="a", *, finished=False, task="default", sid=None):
    session = ctx.run(pr.ProcessSession,
        id=sid or ("proc_aaaa11112222" if contact == "a" else "proc_bbbb11112222"),
        command=f"synthetic-command-{contact}", task_id=task,
        owner_task_id=f"sa-{contact}",
        session_key=ctx.run(sc.get_session_env, "HERMES_SESSION_KEY"),
        parent_session_id=ctx.run(sc.get_session_env, "HERMES_SESSION_ID"),
        watcher_platform=ctx.run(sc.get_session_env, "HERMES_SESSION_PLATFORM"),
        watcher_chat_id=f"contact-{contact}",
        started_at=time.time(), exited=finished, exit_code=0 if finished else None,
        output_buffer=f"private-output-{contact}\n")
    session._pty = SyntheticPty()
    (registry._finished if finished else registry._running)[session.id] = session
    if finished:
        session._completion_event.set()
    return session


def call(ctx, action, sid="", **kwargs):
    # Model-supplied identity-looking fields must never become authority.
    args = dict(action=action, session_id=sid, timeout=0.001, data="test-input",
                task_id="default", session_key="chat-a", profile="default")
    return json.loads(ctx.run(pr._handle_process, args, **kwargs))


def handle(session, form):
    return {"full": session.id, "prefix": session.id[:9], "bare": session.id[5:9]}[form]


@pytest.mark.parametrize("owner", ["a", "b"])
@pytest.mark.parametrize("finished", [False, True], ids=["running", "finished"])
@pytest.mark.parametrize("form", ["full", "prefix", "bare"])
@pytest.mark.parametrize("action", ACTIONS)
def test_foreign_handle_is_nonexistent_without_effects(registry, owner, finished, form, action):
    own = context(owner)
    other = context("b" if owner == "a" else "a")
    session = add(registry, own, owner, finished=finished)
    sid = handle(session, form)
    # Even a trusted executor task key collision cannot override foreign origin.
    result = call(other, action, sid, task_id="default")
    missing = call(other, action, "proc_eeee11112222", task_id="default")
    assert result == {**missing, "error": f"No process with ID {sid}"}
    assert result["status"] == "not_found"
    assert session._pty.events == []
    assert session.exited is finished
    assert not registry._completion_consumed
    assert not registry._poll_observed


@pytest.mark.parametrize("owner", ["a", "b"])
@pytest.mark.parametrize("finished", [False, True], ids=["running", "finished"])
@pytest.mark.parametrize("form", ["full", "prefix", "bare"])
@pytest.mark.parametrize("action", ACTIONS)
def test_owner_can_use_every_handle_operation(registry, owner, finished, form, action):
    ctx = context(owner)
    session = add(registry, ctx, owner, finished=finished, task="background-other-task")
    result = call(ctx, action, handle(session, form), task_id="new-turn-task")
    if action in {"poll", "log", "wait"}:
        assert result["status"] == ("exited" if finished else "timeout" if action == "wait" else "running")
        assert f"private-output-{owner}" in json.dumps(result)
    elif action == "kill":
        assert result["status"] == ("already_exited" if finished else "killed")
        assert session._pty.events == ([] if finished else [("terminate", True)])
    elif finished:
        assert result["status"] == "already_exited"
    else:
        assert result["status"] == "ok"
        assert session._pty.events


@pytest.mark.parametrize("owner", ["a", "b"])
def test_list_only_owned_running_and_finished_even_with_shared_container_key(registry, owner):
    contexts = {c: context(c) for c in ("a", "b")}
    expected = set()
    for c, ctx in contexts.items():
        for i, finished in enumerate((False, True)):
            s = add(registry, ctx, c, finished=finished, sid=f"proc_{c*4}{i:08x}")
            if c == owner:
                expected.add(s.id)
    result = call(contexts[owner], "list", task_id="default")
    assert {s["session_id"] for s in result["processes"]} == expected
    assert f"private-output-{'b' if owner == 'a' else 'a'}" not in json.dumps(result)


@pytest.mark.parametrize("action", ("list",) + ACTIONS)
@pytest.mark.parametrize("missing", ["cleared", "unbound", "platform-only", "gateway-marker"])
def test_missing_gateway_context_never_borrows_environment_authority(registry, monkeypatch, action, missing):
    owner = context()
    session = add(registry, owner)
    monkeypatch.setenv("HERMES_SESSION_KEY", "chat-a")
    monkeypatch.setenv("HERMES_SESSION_PROFILE", "default")
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "cli")
    ctx = contextvars.Context()
    if missing == "cleared":
        ctx.run(sc.clear_session_vars, [])
    elif missing == "platform-only":
        ctx = context(key="")
    elif missing == "gateway-marker":
        monkeypatch.setattr(sc, "_session_context_engaged", False)
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
    result = call(ctx, action, session.id, task_id="default")
    assert result == {"processes": []} if action == "list" else result["status"] == "not_found"
    assert session._pty.events == []


@pytest.mark.parametrize("action", ("list",) + ACTIONS)
@pytest.mark.parametrize("difference", ["profile", "platform"])
def test_matching_key_cannot_cross_profile_or_source(registry, action, difference):
    owner = context()
    session = add(registry, owner)
    other = context(profile="other" if difference == "profile" else "default",
                    platform="telegram" if difference == "platform" else "whatsapp")
    result = call(other, action, session.id, task_id="default")
    assert result == {"processes": []} if action == "list" else result["status"] == "not_found"
    assert session._pty.events == []


def test_foreign_prefix_collision_does_not_hide_owned_process(registry):
    owner = context()
    s = add(registry, owner)
    add(registry, context("b"), "b", sid="proc_aaaa99998888")
    result = call(owner, "poll", "aaaa")
    assert result["session_id"] == s.id


def test_denial_does_not_refresh_foreign_detached_handle(registry):
    s = add(registry, context())
    s._pty = None
    s.detached = True
    s.pid = None  # refresh would mark it exited without touching a host PID
    result = call(context("b"), "poll", s.id)
    assert result["status"] == "not_found"
    assert not s.exited
    assert s.id in registry._running


def test_internal_cleanup_is_not_subject_to_model_contact_gate(registry):
    s = add(registry, context())
    result = context("b").run(registry.kill_all, "default")
    assert result == 1
    assert s._pty.events == [("terminate", True)]


@pytest.mark.parametrize("action", ("list",) + ACTIONS)
def test_explicit_owner_cli_retains_administrative_access(registry, action):
    s = add(registry, context(), finished=True)
    cli = context(source="cli", platform="", key="")
    result = call(cli, action, s.id)
    assert result["processes"] if action == "list" else result["status"] != "not_found"


def test_delegated_child_inherits_contact_ownership_not_container_authority(registry):
    from agent.delegation_context import delegated_child_context
    owner = context()
    own = add(registry, owner)
    add(registry, context("b"), "b")
    def child():
        with delegated_child_context("child-conversation"):
            return json.loads(pr._handle_process({"action": "list"}, task_id="sa-child"))
    result = owner.run(child)
    assert {s["session_id"] for s in result["processes"]} == {own.id}


@pytest.mark.parametrize("action", ("list",) + ACTIONS)
@pytest.mark.parametrize("difference", ["home", "source", "legacy"])
def test_additional_owner_boundaries(registry, tmp_path, action, difference):
    from hermes_constants import set_hermes_home_override
    owner = context()
    session = add(registry, owner)
    caller = context(source="api" if difference == "source" else "")
    if difference == "home":
        caller.run(set_hermes_home_override, str(tmp_path / "foreign-home"))
    if difference == "legacy":
        session.owner_context = {}
    result = call(caller, action, session.id)
    assert result == {"processes": []} if action == "list" else result["status"] == "not_found"
    assert not session._pty.events


@pytest.mark.parametrize("action", ("list",) + ACTIONS)
@pytest.mark.parametrize("form", ["full", "prefix", "bare"])
def test_all_denials_precede_detached_refresh(registry, action, form):
    session = add(registry, context())
    session._pty = None
    session.detached = True
    session.pid = None  # A refresh would finish/move this record.
    result = call(context("b"), action, handle(session, form))
    assert result == {"processes": []} if action == "list" else result["status"] == "not_found"
    assert not session.exited
    assert session.id in registry._running
    assert not registry._poll_observed
    assert not registry._completion_consumed


@pytest.mark.parametrize("difference", ["platform", "chat", "cli-with-message"])
def test_bound_message_source_cannot_conflict_with_owner(registry, difference):
    from types import SimpleNamespace
    ctx = context()
    session = add(registry, ctx)
    caller = context(source="cli", platform="", key="") if difference == "cli-with-message" else context()
    caller.run(approval.set_current_message_source, SimpleNamespace(
        platform="telegram" if difference == "platform" else "whatsapp",
        chat_id="contact-b" if difference == "chat" else "contact-a"))
    assert call(caller, "poll", session.id)["status"] == "not_found"


def test_matching_bound_message_source_keeps_owner_access(registry):
    from types import SimpleNamespace
    ctx = context()
    session = add(registry, ctx, finished=True)
    ctx.run(approval.set_current_message_source, SimpleNamespace(platform="whatsapp", chat_id="contact-a"))
    assert call(ctx, "poll", session.id)["status"] == "exited"


@pytest.mark.parametrize("difference", ["profile", "home"])
def test_cli_administration_cannot_cross_profile_home(registry, tmp_path, difference):
    from hermes_constants import set_hermes_home_override
    session = add(registry, context(), finished=True)
    cli = context(source="cli", platform="", key="", profile="other" if difference == "profile" else "default")
    if difference == "home":
        cli.run(set_hermes_home_override, str(tmp_path / "other-home"))
    assert call(cli, "list") == {"processes": []}
    assert call(cli, "poll", session.id)["status"] == "not_found"


def test_standalone_cli_controls_same_home_processes(registry, monkeypatch):
    # Real legacy CLI never binds session vars; there is no multiplex host.
    ctx = contextvars.Context()
    session = add(registry, ctx, finished=True)
    monkeypatch.setattr(sc, "_session_context_engaged", False)
    assert call(ctx, "poll", session.id)["status"] == "exited"
    assert {p["session_id"] for p in call(ctx, "list")["processes"]} == {session.id}


def test_compression_rotation_preserves_conversation_ownership(registry, monkeypatch):
    ctx = context()
    session = add(registry, ctx, finished=True)
    # This is the session identity setter used by conversation_compression.
    # Monkeypatch records the env key so teardown restores the test environment.
    monkeypatch.setenv("HERMES_SESSION_ID", "conversation-a")
    ctx.run(sc.set_current_session_id, "compressed-continuation-a")
    assert ctx.run(sc.get_session_env, "HERMES_SESSION_ID") != session.parent_session_id
    assert call(ctx, "poll", session.id)["status"] == "exited"
    assert call(context("b", session_id="compressed-continuation-a"), "poll", session.id)["status"] == "not_found"


def test_delegated_background_child_process_is_owned_by_parent(registry):
    from concurrent.futures import ThreadPoolExecutor
    from agent.delegation_context import delegated_child_context
    ctx = context()
    def spawn_child():
        with delegated_child_context("child-durable-id"):
            return add(registry, contextvars.copy_context(), finished=True, task="bg-child")
    with ThreadPoolExecutor(max_workers=1) as pool:
        session = pool.submit(ctx.run, spawn_child).result(timeout=10)
    assert session.parent_session_id == "child-durable-id"
    assert call(ctx, "poll", session.id, task_id="sa-next-child")["status"] == "exited"
    assert call(context("b"), "poll", session.id, task_id="bg-child")["status"] == "not_found"


def test_model_dispatch_scope_is_restored_on_exception(registry, monkeypatch):
    session = add(registry, context())
    def fail(*args, **kwargs):
        # Upstream dispatches list through list_sessions(task_id=..., session_key=...);
        # tolerate its call shape — this test asserts scope restoration, not arity.
        raise RuntimeError("synthetic dispatch fault")
    monkeypatch.setattr(registry, "list_sessions", fail)
    with pytest.raises(RuntimeError, match="synthetic dispatch fault"):
        call(context("b"), "list")
    assert context("b").run(registry.get, session.id) is session


@pytest.mark.parametrize("action", ("list",) + ACTIONS)
def test_legacy_missing_owner_cannot_prove_cli_same_profile(registry, action):
    session = add(registry, context(), finished=True)
    session.owner_context = {}
    cli = context(source="cli", platform="", key="")
    result = call(cli, action, session.id)
    assert result == {"processes": []} if action == "list" else result["status"] == "not_found"


def test_process_session_keeps_positional_task_id_compatibility(registry):
    session = pr.ProcessSession("proc_positional", "synthetic", "task-positional")
    assert session.task_id == "task-positional"


@pytest.mark.parametrize("use_pty", [False, True], ids=["pipe", "pty"])
def test_real_spawn_through_registered_handler_is_owner_scoped(registry, tmp_path, use_pty):
    import shlex
    import sys
    ctx = context()
    command = shlex.join([sys.executable, "-u", "-c", "print('owned-real-output', flush=True)"])
    session = ctx.run(registry.spawn_local, command, cwd=str(tmp_path), task_id="default", session_key="chat-a", use_pty=use_pty)
    try:
        assert session._completion_event.wait(timeout=15)
        assert bool(session._pty) is use_pty
        # Upstream names the registered tool process_manage; the handle scoping
        # asserted below is unchanged by the rename.
        entry = pr.registry.get_entry("process_manage")
        assert entry is not None
        def dispatch(caller, action, sid):
            return json.loads(caller.run(entry.handler, {"action": action, "session_id": sid}, task_id="default"))
        for form in ("full", "prefix", "bare"):
            sid = handle(session, form)
            assert dispatch(context("b"), "poll", sid)["status"] == "not_found"
            result = dispatch(ctx, "poll", sid)
            assert result["status"] == "exited"
            assert "owned-real-output" in result["output_preview"]
    finally:
        registry.kill_process(session.id)


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_recovery_does_not_attribute_old_resource_to_recovering_caller(registry, tmp_path, legacy):
    import shlex
    import sys
    owner = context()
    command = shlex.join([sys.executable, "-u", "-c", "import time; time.sleep(60)"])
    session = owner.run(registry.spawn_local, command, cwd=str(tmp_path), task_id="default", session_key="chat-a")
    try:
        registry._write_checkpoint()
        saved = json.loads(pr.CHECKPOINT_PATH.read_text(encoding="utf-8"))
        assert len(saved) == 1
        if legacy:
            saved[0].pop("owner_context")
            pr.CHECKPOINT_PATH.write_text(json.dumps(saved), encoding="utf-8")
        # Recovery really checks the live synthetic child's PID/start time.
        other = context("b")
        restored = other.run(pr.ProcessRegistry)
        assert other.run(restored.recover_from_checkpoint) == 1
        recovered = restored.get(session.id)
        assert recovered is not None
        assert recovered.detached
        old_registry = pr.process_registry
        pr.process_registry = restored
        try:
            assert call(other, "list") == {"processes": []}
            assert call(other, "poll", session.id)["status"] == "not_found"
            result = call(owner, "poll", session.id)
            assert result["status"] == ("not_found" if legacy else "running")
        finally:
            pr.process_registry = old_registry
    finally:
        registry.kill_process(session.id)
