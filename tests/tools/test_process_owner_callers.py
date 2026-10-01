"""Replay reviewer F1–F3 through callers, real child processes, and the UI sink."""
import contextvars
import json
import os
import shlex
import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway import session_context as sc
from tools import approval, process_registry as pr
from tests.tools.test_process_ownership import registry, context, add, call


def spawn_and_manage(reg, tmp_path, task_id="default", before_control=None):
    from tools.terminal_tool import terminal_tool
    command = shlex.join([sys.executable, "-u", "-c",
                          "import sys; print('ready', flush=True); print(sys.stdin.readline(), flush=True)"])
    started = json.loads(terminal_tool(command, background=True, pty=True, workdir=str(tmp_path),
                                       task_id=task_id, force=True, _host_local=True))
    assert started.get("session_id"), started
    sid = started["session_id"]
    try:
        # Upstream registers the process tool as process_manage (renamed from
        # process); the ownership assertions below are unchanged by the rename.
        entry = pr.registry.get_entry("process_manage")
        assert entry is not None
        handler = entry.handler
        def invoke(action, **args):
            return json.loads(handler(dict(action=action, session_id=sid, **args), task_id=task_id))
        assert sid in {p["session_id"] for p in invoke("list")["processes"]}
        if before_control is not None:
            assert invoke("poll")["status"] == "running"
            before_control(sid)
            assert sid in {p["session_id"] for p in invoke("list")["processes"]}
        results = [invoke("poll"), invoke("write", data="owned-"), invoke("submit", data="input"),
                   invoke("close"), invoke("wait", timeout=10), invoke("log"), invoke("kill")]
        assert all(r.get("status") != "not_found" for r in results), results
        assert results[1]["status"] == results[2]["status"] == "ok", results
        assert results[4]["status"] == "exited", results
        assert "owned-input" in json.dumps(results), results
        # Exercise termination of a live owned child too, not just the
        # already-exited response after a successful wait.
        second = json.loads(terminal_tool(command, background=True, pty=True,
                                         workdir=str(tmp_path), task_id=task_id,
                                         force=True, _host_local=True))
        assert second.get("session_id"), second
        try:
            killed = json.loads(handler({"action": "kill", "session_id": second["session_id"]}, task_id=task_id))
            assert killed["status"] == "killed", killed
        finally:
            reg.kill_process(second["session_id"])
        return sid
    finally:
        reg.kill_process(sid)


def test_real_cron_run_can_manage_own_process_not_delivery_chat_or_other_run(registry, tmp_path, monkeypatch):
    from cron import scheduler
    foreign_ctx = context()
    foreign = add(registry, foreign_ctx, finished=True)
    seen = []
    agent = MagicMock()
    def conversation(*args, task_id=None, **kwargs):
        ctx = contextvars.copy_context()
        # Real run_job binding and worker-context propagation, not a fixture
        # impersonating cron. The provider itself is replaced to avoid inference.
        seen.append((ctx, spawn_and_manage(registry, tmp_path, task_id)))
        assert call(ctx, "poll", foreign.id)["status"] == "not_found"
        for name in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_KEY"):
            assert sc.get_session_env(name) == ""  # no delivery impersonation
        return {"final_response": "verified"}
    agent.run_conversation.side_effect = conversation
    db = MagicMock()
    db.get_compression_tip.side_effect = lambda sid: sid
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
             "api_key": "synthetic", "base_url": "https://example.invalid/v1",
             "provider": "openrouter", "api_mode": "chat_completions"}), \
         patch("run_agent.AIAgent", return_value=agent):
        for _ in range(2):
            result = contextvars.Context().run(scheduler.run_job, {
                "id": "same-job", "name": "ownership replay", "prompt": "Run the local check", "model": "synthetic-model",
                "deliver": "origin", "origin": {"platform": "whatsapp", "chat_id": "contact-a"}})
            assert result[0] is True, result
    assert len(seen) == 2
    assert call(seen[0][0], "poll", seen[1][1])["status"] == "not_found"
    assert call(foreign_ctx, "poll", seen[0][1])["status"] == "not_found"
    assert call(contextvars.Context(), "poll", seen[0][1])["status"] == "not_found"


def test_real_classic_cli_source_tag_does_not_revoke_process_access(registry, tmp_path, monkeypatch):
    import hermes_cli.main as main
    monkeypatch.setattr(main, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(main, "_termux_should_prefetch_update_check", lambda: False)
    monkeypatch.setattr(main, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main, "_confirm_startup_expensive_model_override", lambda args: None)
    called = []
    def turn(**kwargs):
        assert os.environ["HERMES_SESSION_SOURCE"] == "tool"
        called.append(spawn_and_manage(registry, tmp_path))
    monkeypatch.setitem(sys.modules, "cli", SimpleNamespace(main=turn))
    args = Namespace(cli=True, tui=False, continue_last=None, source="tool", model=None,
                     toolsets=None, query="check", safe_mode=False)
    contextvars.Context().run(main.cmd_chat, args)
    assert len(called) == 1


def test_notification_routing_alias_does_not_change_canonical_owner(registry, tmp_path):
    from tools.terminal_tool import terminal_tool
    phone, lid = "5215550001111@s.whatsapp.net", "123456789@lid"
    key = "agent:main:whatsapp:dm:5215550001111"
    owner = contextvars.Context()
    owner.run(sc.set_session_vars, platform="whatsapp", session_key=key, chat_id=phone)
    command = shlex.join([sys.executable, "-u", "-c", "print('alias-owned')"])
    started = json.loads(owner.run(terminal_tool, command, background=True,
                                   notify_on_complete=True, force=True,
                                   _host_local=True, workdir=str(tmp_path)))
    assert started.get("session_id"), started
    session = registry.get(started["session_id"])
    assert session is not None
    assert session._completion_event.wait(10)
    assert session.watcher_chat_id == phone
    assert session.watcher_platform == "whatsapp"
    caller = contextvars.Context()
    caller.run(sc.set_session_vars, platform="whatsapp", session_key=key, chat_id=lid)
    caller.run(approval.set_current_message_source, SimpleNamespace(platform="whatsapp", chat_id=lid))
    assert call(caller, "poll", session.id)["status"] == "exited"
    foreign = contextvars.Context()
    foreign.run(sc.set_session_vars, platform="whatsapp", session_key="foreign", chat_id=phone)
    foreign.run(approval.set_current_message_source, SimpleNamespace(platform="whatsapp", chat_id=phone))
    assert call(foreign, "poll", session.id)["status"] == "not_found"


@pytest.fixture
def desktop(registry, tmp_path, monkeypatch):
    from tui_gateway import server
    from tools import close_terminal_tool
    monkeypatch.setattr(close_terminal_tool, "process_registry", registry)
    monkeypatch.setattr(server, "_sessions", {
        "window-a": {"session_key": "desktop-a", "source": "desktop"},
        "window-b": {"session_key": "desktop-b", "source": "desktop"},
    })
    events = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload: events.append((event, sid, payload)))
    server._wire_agent_terminal_output()
    def bind(which):
        ctx = contextvars.Context()
        ctx.run(server._set_session_context, f"desktop-{which}", str(tmp_path), ui_session_id=f"window-{which}")
        return ctx
    def close(ctx, pid):
        return json.loads(ctx.run(pr.registry.get_entry("close_terminal").handler, {"process_id": pid}))
    return bind, close, events, server


@pytest.mark.parametrize("form", ["full", "prefix", "bare"])
def test_desktop_close_cannot_emit_foreign_process_to_sink(registry, desktop, form):
    bind, close, events, _ = desktop
    a, b = bind("a"), bind("b")
    foreign = add(registry, a)
    pid = {"full": foreign.id, "prefix": foreign.id[:9], "bare": foreign.id[5:9]}[form]
    assert close(b, pid)["status"] == "not_found"
    assert events == []
    assert foreign._pty.events == []
    assert close(a, pid)["status"] == "ok"
    assert events == [("terminal.close", "window-a", {"process_id": foreign.id})]
    assert foreign._pty.events == []


def test_pruned_tab_close_targets_only_positively_bound_current_window(registry, desktop):
    bind, close, events, _ = desktop
    assert close(bind("b"), "proc_pruned12345")["status"] == "ok"
    assert events == [("terminal.close", "window-b", {"process_id": "proc_pruned12345"})]


@pytest.mark.parametrize("missing", ["unbound", "key", "ui", "messaging"])
def test_pruned_close_never_broadcasts_without_ui_provenance(registry, desktop, monkeypatch, missing):
    bind, close, events, _ = desktop
    ctx = contextvars.Context()
    if missing != "unbound":
        ctx.run(sc.set_session_vars, session_key="" if missing == "key" else "desktop-a",
                ui_session_id="" if missing == "ui" else "window-a",
                source="desktop", platform="whatsapp" if missing == "messaging" else "")
    monkeypatch.setenv("HERMES_UI_SESSION_ID", "window-a")
    monkeypatch.setenv("HERMES_SESSION_KEY", "desktop-a")
    assert close(ctx, "proc_pruned12345")["status"] == "not_found"
    assert events == []


def test_close_sink_does_not_broadcast_or_target_unmatched_ui_identity(registry, desktop):
    _, _, events, _ = desktop
    registry.on_close(None, "proc_pruned12345")
    assert events == []
    orphan = pr.ProcessSession(id="proc_orphan", command="synthetic", session_key="unknown")
    registry.on_close(orphan, orphan.id)
    assert events == []


@pytest.mark.parametrize("version", [None, 999, "1"])
def test_unknown_checkpoint_owner_version_cannot_gain_cli_authority(registry, version):
    session = add(registry, context(), finished=True)
    if version is None:
        session.owner_context.pop("version", None)
    else:
        session.owner_context["version"] = version
    assert call(context(source="cli", platform="", key=""), "poll", session.id)["status"] == "not_found"


@pytest.mark.parametrize("clear", [sc.reset_session_vars, lambda: sc.clear_session_vars([])])
def test_execution_principal_is_cleared_and_never_borrowed_from_environment(registry, monkeypatch, clear):
    ctx = contextvars.Context()
    ctx.run(sc.set_session_vars, execution_principal="owned-run")
    session = ctx.run(pr.ProcessSession, id="proc_internal1234", command="synthetic")
    registry._running[session.id] = session
    inherited = ctx.copy()
    monkeypatch.setenv("execution_principal", "owned-run")
    monkeypatch.setenv("HERMES_EXECUTION_PRINCIPAL", "owned-run")
    inherited.run(clear)
    assert inherited.run(sc.bound_session_snapshot)["execution_principal"] == ""
    assert call(inherited, "poll", session.id)["status"] == "not_found"
    assert call(ctx, "poll", session.id)["status"] == "running"


def test_execution_principal_cannot_fall_back_to_matching_container_key(registry):
    owner = contextvars.Context()
    owner.run(sc.set_session_vars, execution_principal="host-created-run")
    session = owner.run(pr.ProcessSession, id="proc_cron123456", command="synthetic",
                        session_key="default", exited=True, exit_code=0)
    registry._finished[session.id] = session
    caller = contextvars.Context()
    caller.run(sc.set_session_vars, session_key="default")
    assert call(caller, "poll", session.id)["status"] == "not_found"
    assert call(owner, "poll", session.id)["status"] == "exited"


@pytest.mark.parametrize("marker", ["_HERMES_GATEWAY", "HERMES_GATEWAY_SESSION"])
def test_missing_key_gateway_cannot_gain_admin_from_cli_label(registry, monkeypatch, marker):
    owned = add(registry, context(), finished=True)
    monkeypatch.setenv(marker, "1")
    caller = context(source="cli", platform="", key="")
    assert call(caller, "poll", owned.id)["status"] == "not_found"
    assert call(caller, "list") == {"processes": []}


def test_pruned_close_reports_denial_when_bound_window_no_longer_matches(registry, desktop):
    bind, close, events, server = desktop
    caller = bind("a")
    server._sessions["window-a"]["session_key"] = "new-conversation"
    assert close(caller, "proc_pruned12345")["status"] == "not_found"
    assert events == []


def test_bound_snapshot_never_reads_environment_or_returns_live_state(registry, monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_KEY", "ambient-foreign")
    ctx = contextvars.Context()
    assert ctx.run(sc.bound_session_snapshot).get("HERMES_SESSION_KEY", "") == ""
    ctx.run(sc.set_session_vars, session_key="bound-own")
    snapshot = ctx.run(sc.bound_session_snapshot)
    assert snapshot["HERMES_SESSION_KEY"] == "bound-own"
    snapshot["HERMES_SESSION_KEY"] = "tampered-copy"
    assert ctx.run(sc.bound_session_snapshot)["HERMES_SESSION_KEY"] == "bound-own"


@pytest.fixture
def classic_cli(monkeypatch):
    """Real cmd_chat entry/exit; replace only startup services and conversation."""
    import hermes_cli.main as main
    monkeypatch.setattr(main, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(main, "_termux_should_prefetch_update_check", lambda: False)
    monkeypatch.setattr(main, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main, "_confirm_startup_expensive_model_override", lambda args: None)

    def run(turn, source="cli", ctx=None):
        monkeypatch.setitem(sys.modules, "cli", SimpleNamespace(main=turn))
        args = Namespace(cli=True, tui=False, continue_last=None, source=source,
                         model=None, toolsets=None, query="check", safe_mode=False)
        (ctx if ctx is not None else contextvars.Context()).run(main.cmd_chat, args)
    return run


@pytest.mark.parametrize("source,profile", [
    ("cli", ""), ("tool", ""), ("custom-integration", ""), ("tool", "named-profile"),
])
@pytest.mark.parametrize("cron_fails", [False, True], ids=["cron-success", "cron-failure"])
def test_classic_cli_controls_existing_process_after_real_cron_worker_returns(
        registry, tmp_path, monkeypatch, classic_cli, source, profile, cron_fails):
    from concurrent.futures import ThreadPoolExecutor
    from cron import scheduler
    from tools.thread_context import propagate_context_to_thread

    monkeypatch.setenv("HERMES_SESSION_PROFILE", profile)
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "api_key": "synthetic", "base_url": "https://example.invalid/v1",
        "provider": "openrouter", "api_mode": "chat_completions"})
    observed = []

    def turn(**kwargs):
        assert os.environ["HERMES_SESSION_SOURCE"] == source

        def run_cron(cli_sid):
            agent = MagicMock()

            def conversation(*args, task_id=None, **kwargs):
                cron_ctx = contextvars.copy_context()
                assert call(cron_ctx, "poll", cli_sid)["status"] == "not_found"
                cron_sid = spawn_and_manage(registry, tmp_path, task_id)
                observed.append((cron_ctx, cron_sid))
                # The cron worker must not revive inherited CLI administration
                # after either form of gateway/session cleanup.
                for clear in (sc.reset_session_vars, lambda: sc.clear_session_vars([])):
                    cleared = cron_ctx.copy()
                    cleared.run(clear)
                    assert call(cleared, "poll", cli_sid)["status"] == "not_found"
                if cron_fails:
                    raise RuntimeError("synthetic provider failure")
                return {"final_response": "verified"}

            agent.run_conversation.side_effect = conversation

            def cron_tool_worker():
                # This is the same context bridge used by tool_executor for
                # sequential AND parallel cronjob tools. No fresh Context() to
                # hide inherited CLI authority; run_job and its worker are real.
                result = scheduler.run_job({
                    "id": "cli-lifecycle", "name": "ownership replay",
                    "prompt": "Run the local check", "model": "synthetic-model",
                    "deliver": "local"})
                assert result[0] is not cron_fails, result
                if cron_fails:
                    assert result[3] == "RuntimeError: synthetic provider failure", result
                cleared = contextvars.copy_context()
                assert call(cleared, "poll", cli_sid)["status"] == "not_found"
                assert sc.bound_session_snapshot()["execution_principal"] == ""

            with patch("run_agent.AIAgent", return_value=agent), \
                 ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(propagate_context_to_thread(cron_tool_worker)).result(timeout=30)
            assert sc.session_context_engaged()  # deliberately never reset the latch
            assert sc.bound_session_snapshot()["execution_principal"] == ""

        spawn_and_manage(registry, tmp_path, before_control=run_cron)
        # A second turn/worker still inherits the entrypoint's provenance after
        # cron's worker has engaged the process-global context machinery.
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(propagate_context_to_thread(
                lambda: spawn_and_manage(registry, tmp_path))).result(timeout=30)

    classic_cli(turn, source)
    assert len(observed) == 1
    assert call(contextvars.Context(), "poll", observed[0][1])["status"] == "not_found"


@pytest.mark.parametrize("boundary", ["set", "reset", "clear"])
def test_inherited_classic_cli_authority_does_not_survive_gateway_scope(
        registry, tmp_path, monkeypatch, classic_cli, boundary):
    def turn(**kwargs):
        def check_gateway(cli_sid):
            gateway = contextvars.copy_context()
            if boundary == "set":
                gateway.run(sc.set_session_vars, platform="whatsapp", chat_id="other", session_key="")
            elif boundary == "reset":
                gateway.run(sc.reset_session_vars)
            else:
                gateway.run(sc.clear_session_vars, [])
            # A lost-context gateway may have an ambient CLI tag from its parent.
            monkeypatch.setenv("HERMES_SESSION_SOURCE", "cli")
            assert call(gateway, "poll", cli_sid)["status"] == "not_found"
            assert call(gateway, "list") == {"processes": []}
            gateway.run(sc.reset_session_vars)
            assert call(gateway, "poll", cli_sid)["status"] == "not_found"
        spawn_and_manage(registry, tmp_path, before_control=check_gateway)
    classic_cli(turn, source="tool")


@pytest.mark.parametrize("raises", [False, True], ids=["return", "exception"])
def test_classic_cli_entrypoint_releases_authority_on_exit(
        registry, tmp_path, classic_cli, raises):
    caller = contextvars.Context()
    owned = []
    def turn(**kwargs):
        owned.append(spawn_and_manage(registry, tmp_path))
        # Engage the latch in a different context, as a worker would.
        contextvars.Context().run(sc.set_session_vars)
        if raises:
            raise RuntimeError("synthetic CLI failure")
    if raises:
        with pytest.raises(RuntimeError, match="synthetic CLI failure"):
            classic_cli(turn, source="tool", ctx=caller)
    else:
        classic_cli(turn, source="tool", ctx=caller)
    assert call(caller, "poll", owned[0])["status"] == "not_found"
