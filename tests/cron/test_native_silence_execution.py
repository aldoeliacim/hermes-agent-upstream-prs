#!/usr/bin/env python3
"""E2E: the cronjob_silent native-silence tool actually suppresses cron delivery.

This suite exercises the REAL scheduler execution seam that the API+toolset
landing left inert. The prior port made the ``cronjob_silent`` tool *visible*
(schema registered, toolset always injected, handler + ``CronSilenceState``
present) but never bound the turn-flag / silence-state around the agent run and
never read the decision back to suppress delivery — so a cron agent could call
the tool and still have its response delivered. These tests drive
``cron.scheduler.run_job`` / ``run_one_job`` down through the genuine
``_run_agent_with_watchdog`` worker-thread seam (``contextvars.copy_context()``)
with a fake agent whose ``run_conversation`` calls the REAL
``cronjob_silent_tool`` handler. Nothing about the binding, the tool
authorization, the ``CronSilenceState`` mutation, or the delivery-suppression
decision is mocked — only bookkeeping and the final platform sender are stubbed.

Coverage:
  * native silence prevents ALL deliver/mirror/ledger-delivered outcome, while
    the run is still saved + marked (the ledger records the outcome, nothing
    leaves the process);
  * an ordinary (non-silent) response DOES deliver through the same path;
  * the string-marker fallback ("SILENCIO"-class magic words) is preserved;
  * calling the tool OUTSIDE a cron turn refuses (never silently no-ops), and a
    bound turn with no ``CronSilenceState`` surfaces the wiring gap loudly;
  * the turn-flag and silence-state never leak past the run (no stale state
    across jobs / concurrent contexts).
"""
import json

import pytest

import cron.scheduler as s
from cron.agent_turn_context import is_cron_agent_turn
from tools.approval import get_cron_silence_state
from tools.cronjob_silent_tool import cronjob_silent_tool


# --------------------------------------------------------------------------- #
# Fake agent: its run_conversation runs inside the worker's copied context,
# exactly where a real cron agent would call the tool.
# --------------------------------------------------------------------------- #
class _FakeAgent:
    """Minimal agent standing in for run_agent.AIAgent.

    ``run_conversation`` optionally invokes the REAL cronjob_silent tool handler
    (proving the turn-flag + CronSilenceState propagated across the
    copy_context() seam and that the handler's authorization + state mutation
    fire), then returns a normal successful result dict.
    """

    def __init__(self, *, call_silent=False, reason="nothing to report",
                 final_response="here is the update", tool_results=None):
        self._call_silent = call_silent
        self._reason = reason
        self._final_response = final_response
        self.tool_results = tool_results if tool_results is not None else []

    def run_conversation(self, prompt, task_id=None):
        if self._call_silent:
            # The genuine handler: checks is_cron_agent_turn(), calls
            # note_cron_silence_decided() against the bound CronSilenceState.
            self.tool_results.append(cronjob_silent_tool({"reason": self._reason}))
        return {"final_response": self._final_response, "completed": True}


def _install_agent_seam(monkeypatch, agent):
    """Replace only agent-construction/config resolution so the REAL binding,
    watchdog worker seam, tool authorization and delivery decision all run."""
    class _Jc:
        cfg = {}
        model = "fake-model"

    class _Setup:
        blocked = None
        model = "fake-model"
        fallback_notice = None
        runtime = {}

    monkeypatch.setattr(s, "_prepare_job_prompt",
                        lambda *a, **k: (None, "do the scan"))
    monkeypatch.setattr(s, "_load_cron_job_config", lambda *a, **k: _Jc())
    monkeypatch.setattr(s, "_resolve_cron_agent_setup", lambda *a, **k: _Setup())
    monkeypatch.setattr(s, "_open_cron_session_db", lambda *a, **k: None)
    monkeypatch.setattr(s, "_construct_cron_agent", lambda *a, **k: agent)
    monkeypatch.setattr(s, "_FireAudit", lambda *a, **k: _DummyAudit())
    monkeypatch.setattr(s, "_reload_dotenv_and_publish_delivery_target",
                        lambda *a, **k: None)
    # Keep the inactivity watchdog thread out of the way — 0 == unlimited.
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")


class _DummyAudit:
    def note(self, *a, **k):
        pass

    def __getattr__(self, _name):
        return lambda *a, **k: None


# --------------------------------------------------------------------------- #
# Group 1 — real run_job seam: the binding stamps native silence, resets clean.
# --------------------------------------------------------------------------- #
class TestRunJobExecutionBinding:
    def test_native_silence_call_stamps_marker_and_resets_state(self, monkeypatch):
        """A cron agent that calls cronjob_silent leaves the per-fire native
        marker set, AND both ContextVars are reset after the run."""
        agent = _FakeAgent(call_silent=True, reason="quiet week")
        _install_agent_seam(monkeypatch, agent)

        # Pre-state: not in a cron turn, no silence state bound.
        assert is_cron_agent_turn() is False
        assert get_cron_silence_state() is None

        job = {"id": "j-silent", "name": "scout"}
        success, output, final_response, error = s.run_job(job)

        # The tool ran under a live cron turn and recorded the decision.
        assert agent.tool_results, "fake agent never reached the tool"
        assert json.loads(agent.tool_results[0]) == {
            "success": True, "decision": "silent"}
        # run_job read the decision back and stamped the fire-local marker.
        assert job.get("_cron_native_silence") is True
        assert success is True and error is None
        # No leak: both bindings reset in reverse order in the finally.
        assert is_cron_agent_turn() is False
        assert get_cron_silence_state() is None

    def test_ordinary_run_leaves_no_native_marker(self, monkeypatch):
        agent = _FakeAgent(call_silent=False, final_response="the brief")
        _install_agent_seam(monkeypatch, agent)

        job = {"id": "j-normal", "name": "scout"}
        success, output, final_response, error = s.run_job(job)

        assert success is True and error is None
        assert final_response == "the brief"
        assert "_cron_native_silence" not in job
        assert is_cron_agent_turn() is False
        assert get_cron_silence_state() is None

    def test_no_stale_state_across_sequential_jobs(self, monkeypatch):
        """A silent job followed by an ordinary job: the second must NOT inherit
        the first's silence decision (fresh CronSilenceState per fire)."""
        silent_agent = _FakeAgent(call_silent=True)
        _install_agent_seam(monkeypatch, silent_agent)
        job1 = {"id": "j1", "name": "a"}
        s.run_job(job1)
        assert job1.get("_cron_native_silence") is True

        normal_agent = _FakeAgent(call_silent=False, final_response="report")
        _install_agent_seam(monkeypatch, normal_agent)
        job2 = {"id": "j2", "name": "b"}
        s.run_job(job2)
        assert "_cron_native_silence" not in job2
        assert get_cron_silence_state() is None
        assert is_cron_agent_turn() is False

    def test_no_cross_leak_between_concurrent_jobs(self, monkeypatch):
        """Two cron fires running on SEPARATE threads at the SAME instant: the
        silent one's decision must never bleed into the loud one. Because the
        bindings are ContextVars set inside run_job (and the watchdog worker
        captures them via copy_context per thread), each fire sees only its own
        CronSilenceState. A Barrier forces both agents to be *inside* their turn
        simultaneously so a process-global would be caught red-handed."""
        import threading

        # Both threads run through the same monkeypatched seam; the agent
        # dispatches on job id so one fire calls the tool and the other does not.
        both_inside = threading.Barrier(2, timeout=10)

        class _DispatchAgent:
            def __init__(self, job):
                self._job = job
                self.tool_results = []

            def run_conversation(self, prompt, task_id=None):
                # Rendezvous so both fires hold their turn binding at once.
                both_inside.wait()
                if self._job["id"] == "c-silent":
                    self.tool_results.append(
                        cronjob_silent_tool({"reason": "quiet"}))
                    return {"final_response": "muted", "completed": True}
                # The loud fire must observe a FRESH (undecided) state despite
                # the silent fire having decided concurrently.
                assert get_cron_silence_state() is not None
                assert get_cron_silence_state().decided is False
                return {"final_response": "loud brief", "completed": True}

        _install_agent_seam(monkeypatch, None)
        monkeypatch.setattr(
            s, "_construct_cron_agent",
            lambda _AIAgent, job, *a, **k: _DispatchAgent(job))

        results = {}

        def _fire(job):
            results[job["id"]] = s.run_job(job)

        jobs = [{"id": "c-silent", "name": "a"}, {"id": "c-loud", "name": "b"}]
        threads = [threading.Thread(target=_fire, args=(j,)) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        silent_job, loud_job = jobs
        # The silent fire stamped its marker; the loud fire did NOT inherit it.
        assert silent_job.get("_cron_native_silence") is True
        assert "_cron_native_silence" not in loud_job
        # Both runs succeeded and the main thread is left fully unbound.
        assert results["c-silent"][0] is True
        assert results["c-loud"][0] is True
        assert results["c-loud"][2] == "loud brief"
        assert get_cron_silence_state() is None
        assert is_cron_agent_turn() is False


# --------------------------------------------------------------------------- #
# Group 2 — full run_one_job path: native silence blocks the real delivery.
# --------------------------------------------------------------------------- #
@pytest.fixture
def deliver_env(monkeypatch, tmp_path):
    """run_one_job with the REAL delivery path down to a captured sender.

    Only bookkeeping + the platform sender are stubbed; the native-silence
    binding, tool call and _save_compose_deliver suppression are genuine.
    """
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  slack:\n    enabled: true\n    token: xoxb-test\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    send_calls = []

    async def fake_sender(pconfig, chat_id, message, *, thread_id=None,
                          media_files=None, force_document=False, caption=None):
        send_calls.append({"chat_id": chat_id, "message": message})
        return {"success": True, "chat_id": chat_id, "message_id": "1.2"}

    import gateway.platform_registry as reg
    import hermes_cli.plugins as hp

    entry = reg.platform_registry.get("slack")
    if entry is None:
        hp.discover_plugins()
        entry = reg.platform_registry.get("slack")
    if entry is None:
        pytest.skip("slack platform entry not registered")
    monkeypatch.setattr(entry, "standalone_sender_fn", fake_sender)
    monkeypatch.setattr(hp, "discover_plugins", lambda *a, **k: None)

    state = {"send": send_calls, "marked": [], "saved": []}
    monkeypatch.setattr(s, "create_execution", lambda *_a, **_kw: {"id": "exec-t"})
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: {})
    monkeypatch.setattr(
        s, "save_job_output",
        lambda jid, out: state["saved"].append(jid) or f"/tmp/{jid}.txt")
    monkeypatch.setattr(
        s, "mark_job_run",
        lambda *a, **kw: state["marked"].append((a, kw)) or True)
    monkeypatch.setattr(s, "finish_execution", lambda *a, **kw: None)
    monkeypatch.setattr(s, "_upsert_incident_for_failure",
                        lambda *_a, **_kw: (False, None))
    monkeypatch.setattr(s, "load_config", lambda: {})
    return state


class TestNativeSilenceSuppressesDelivery:
    def test_tool_call_suppresses_all_delivery(self, deliver_env, monkeypatch):
        """(a) The agent calls cronjob_silent -> NOTHING is delivered/mirrored,
        yet the run is still saved + marked (ledger records the outcome)."""
        agent = _FakeAgent(call_silent=True,
                           final_response="I would have said lots here")
        _install_agent_seam(monkeypatch, agent)

        s.run_one_job({"id": "j-quiet", "name": "scout",
                       "deliver": "slack:D0MAIN"})

        assert agent.tool_results, "native tool was never reached"
        assert deliver_env["send"] == [], (
            "native silence must prevent ALL delivery")
        # The outcome is still recorded — silence is not a failure.
        assert deliver_env["saved"] == ["j-quiet"]
        assert deliver_env["marked"], "run should still be marked"

    def test_ordinary_response_delivers(self, deliver_env, monkeypatch):
        """(c) Same path, no tool call -> the response is delivered."""
        agent = _FakeAgent(call_silent=False,
                           final_response="all good, here is the brief")
        _install_agent_seam(monkeypatch, agent)

        s.run_one_job({"id": "j-loud", "name": "scout",
                       "deliver": "slack:D0MAIN"})

        assert [c["chat_id"] for c in deliver_env["send"]] == ["D0MAIN"]
        assert "here is the brief" in deliver_env["send"][0]["message"]

    def test_string_marker_fallback_still_suppresses(self, deliver_env, monkeypatch):
        """The legacy lexical silence contract is preserved: a bare [SILENT]
        response still suppresses delivery even without a tool call."""
        agent = _FakeAgent(call_silent=False, final_response="[SILENT]")
        _install_agent_seam(monkeypatch, agent)

        s.run_one_job({"id": "j-marker", "name": "scout",
                       "deliver": "slack:D0MAIN"})

        assert deliver_env["send"] == [], (
            "string-marker silence fallback must remain intact")


# --------------------------------------------------------------------------- #
# Group 3 — the tool refuses outside a cron turn (independent of scheduler).
# --------------------------------------------------------------------------- #
class TestOutsideTurnRefusal:
    def test_refuses_when_no_cron_turn_active(self):
        assert is_cron_agent_turn() is False
        out = cronjob_silent_tool({"reason": "whatever"})
        assert "only valid inside a running cron job" in out
        # And it did NOT silently succeed.
        assert '"decision": "silent"' not in out
        # State remains unbound: no accidental global effect.
        assert get_cron_silence_state() is None

    def test_bound_turn_without_state_surfaces_wiring_gap(self):
        """If the turn flag is set but no CronSilenceState is bound (a wiring
        bug), the tool loudly reports it rather than pretending to succeed."""
        from cron.agent_turn_context import (
            reset_cron_agent_turn, set_cron_agent_turn)
        tok = set_cron_agent_turn(True)
        try:
            out = cronjob_silent_tool({"reason": "x"})
            assert "no" in out.lower() and "cronsilencestate" in out.lower().replace(" ", "")
            assert '"decision": "silent"' not in out
        finally:
            reset_cron_agent_turn(tok)
        assert is_cron_agent_turn() is False
