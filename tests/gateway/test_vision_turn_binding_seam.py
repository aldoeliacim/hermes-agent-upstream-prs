"""E2E: the gateway per-turn seam must ACTIVATE the vision stale-cache guard.

The guard in ``tools.vision_tools`` is inert unless the gateway binds the
current turn's inbound image paths via ``set_current_turn_image_paths`` before
tools run, and resets the binding in a ``finally`` on success/error/cancel.

Unlike ``tests/tools/test_vision_stale_cache_guard.py`` (which sets the
contextvar directly), these tests drive the REAL boundary —
``TurnRunner._run_conversation_with_approval`` — and observe the guard's live
value from *inside* the model turn (``run_conversation``), exactly where the
native vision tool executes. That is the only place that proves the seam is no
longer inert.

Property under test:
  * during the turn, ``_is_stale_inbound_cache_path`` refuses a cache path from
    an EARLIER turn but allows the path the gateway attached THIS turn;
  * after the turn the binding is reset to ``None`` (inert) — on success, on
    error, and on cancellation — with no cross-turn leak;
  * a queued follow-up turn observes ITS OWN attached paths, never the prior
    turn's.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.vision_tools as vt
from gateway.turn_context import TurnContext


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = (tmp_path / "image_cache").resolve()
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(vt, "_inbound_image_cache_dir", lambda: d)
    return d


def _touch(d: Path, name: str) -> Path:
    p = d / name
    p.write_bytes(b"\x00")
    return p.resolve()


class _StubRunner:
    """Minimal GatewayRunner stand-in for the seam under test."""

    def __init__(self, native_by_session):
        # session_key -> list[str] of one-shot native image paths (consumed once)
        self._native = dict(native_by_session)

    def _consume_pending_native_image_paths(self, session_key):
        paths = list(self._native.get(session_key) or [])
        if paths:
            self._native[session_key] = []
        return paths

    # The seam wraps the message with observed group context via a run.py import;
    # keep that import-time symbol trivially callable through monkeypatch below.


class _ProbeAgent:
    """Agent whose run_conversation probes the LIVE guard state mid-turn.

    Whatever ``probe`` returns is recorded so the test can assert what the vision
    fast path would have seen at the exact moment it runs.
    """

    def __init__(self, probe, raise_exc=None):
        self._probe = probe
        self._raise = raise_exc
        self.observed = None

    def run_conversation(self, message, **kwargs):
        # This runs at the same nesting the native vision tool runs at.
        self.observed = self._probe()
        if self._raise is not None:
            raise self._raise
        return "ok"


def _make_ctx(session_key, message="look at this"):
    source = SimpleNamespace(user_id="u1", user_name="User", is_bot=False)
    return TurnContext(
        source=source,
        message=message,
        session_id=session_key,
        session_key=session_key,
    )


def _run_seam(monkeypatch, runner, ctx, agent):
    """Invoke the real _run_conversation_with_approval seam with light stubs."""
    from gateway.run_turn_runner import TurnRunner

    # Neutralize the observed-group-context wrapper and notification_turn ctxmgr
    # so the seam runs without a full gateway; we only care about the binding.
    import gateway.run as gwrun
    monkeypatch.setattr(gwrun, "_wrap_current_message_with_observed_context",
                        lambda msg, _obs: msg, raising=False)

    import agent.notification_presentation as np
    from contextlib import contextmanager

    @contextmanager
    def _noop_turn(*a, **k):
        yield
    monkeypatch.setattr(np, "notification_turn", _noop_turn, raising=False)

    tr = TurnRunner(runner, ctx)
    return tr._run_conversation_with_approval(
        agent, agent_history=[], observed_group_context=None,
        persist_user_message_override=None, persist_user_timestamp_override=None,
    )


def test_seam_binds_current_turn_paths_and_blocks_stale(cache_dir, monkeypatch):
    """Inside the turn: this turn's image is fresh; a prior-turn cache path is stale."""
    fresh = _touch(cache_dir, "img_fresh.jpg")
    stale = _touch(cache_dir, "img_stale.jpg")

    def probe():
        return {
            "bound": vt._current_turn_image_paths.get(),
            "fresh_stale": vt._is_stale_inbound_cache_path(fresh),
            "stale_stale": vt._is_stale_inbound_cache_path(stale),
        }

    runner = _StubRunner({"s1": [str(fresh)]})
    agent = _ProbeAgent(probe)
    ctx = _make_ctx("s1")

    result = _run_seam(monkeypatch, runner, ctx, agent)

    assert result == "ok"
    # The seam must have bound SOMETHING (not None) during the turn.
    assert agent.observed["bound"] is not None, "seam left the guard inert (binding is None)"
    # This turn's attached image is allowed; a prior-turn cache path is refused.
    assert agent.observed["fresh_stale"] is False
    assert agent.observed["stale_stale"] is True
    # After the turn: reset to inert (no leak).
    assert vt._current_turn_image_paths.get() is None


def test_seam_empty_turn_blocks_any_cache_path(cache_dir, monkeypatch):
    """A turn that attached NO inbound images => any inbound-cache path is stale."""
    some = _touch(cache_dir, "img_any.jpg")

    def probe():
        return {
            "bound": vt._current_turn_image_paths.get(),
            "stale": vt._is_stale_inbound_cache_path(some),
        }

    runner = _StubRunner({"s2": []})
    agent = _ProbeAgent(probe)
    ctx = _make_ctx("s2")

    _run_seam(monkeypatch, runner, ctx, agent)

    assert agent.observed["bound"] == [], "empty turn must bind an empty list, not None"
    assert agent.observed["stale"] is True
    assert vt._current_turn_image_paths.get() is None


def test_seam_resets_binding_on_error(cache_dir, monkeypatch):
    """An exception mid-turn still resets the binding in finally (no leak)."""
    img = _touch(cache_dir, "img_err.jpg")

    runner = _StubRunner({"s3": [str(img)]})
    agent = _ProbeAgent(lambda: vt._current_turn_image_paths.get(),
                        raise_exc=RuntimeError("boom"))
    ctx = _make_ctx("s3")

    with pytest.raises(RuntimeError, match="boom"):
        _run_seam(monkeypatch, runner, ctx, agent)

    # Bound during the turn, reset after despite the error.
    assert agent.observed is not None
    assert vt._current_turn_image_paths.get() is None


def test_seam_resets_binding_on_cancel(cache_dir, monkeypatch):
    """A cancellation mid-turn still resets the binding in finally (no leak)."""
    img = _touch(cache_dir, "img_cancel.jpg")

    runner = _StubRunner({"s4": [str(img)]})
    agent = _ProbeAgent(lambda: vt._current_turn_image_paths.get(),
                        raise_exc=KeyboardInterrupt())
    ctx = _make_ctx("s4")

    with pytest.raises(KeyboardInterrupt):
        _run_seam(monkeypatch, runner, ctx, agent)

    assert vt._current_turn_image_paths.get() is None


def test_queued_followup_gets_its_own_paths(cache_dir, monkeypatch):
    """A follow-up turn observes ITS OWN attached image, never the prior turn's."""
    first = _touch(cache_dir, "img_first.jpg")
    second = _touch(cache_dir, "img_second.jpg")

    seen = {}

    def probe_key(key):
        return lambda: {
            "bound": list(vt._current_turn_image_paths.get() or []),
            "first_stale": vt._is_stale_inbound_cache_path(first),
            "second_stale": vt._is_stale_inbound_cache_path(second),
        }

    # Same runner instance, two turns on the same session_key (queued follow-up).
    runner = _StubRunner({"s5": [str(first)]})

    a1 = _ProbeAgent(probe_key("t1"))
    _run_seam(monkeypatch, runner, _make_ctx("s5"), a1)
    seen["t1"] = a1.observed
    assert vt._current_turn_image_paths.get() is None

    # Follow-up arrives with a DIFFERENT image; the one-shot consume was cleared,
    # so we re-arm the session's native paths for the second turn.
    runner._native["s5"] = [str(second)]
    a2 = _ProbeAgent(probe_key("t2"))
    _run_seam(monkeypatch, runner, _make_ctx("s5"), a2)
    seen["t2"] = a2.observed

    # Turn 1 saw first fresh, second stale.
    assert seen["t1"]["first_stale"] is False
    assert seen["t1"]["second_stale"] is True
    # Turn 2 saw second fresh, first stale — no leak of prior-turn paths.
    assert seen["t2"]["second_stale"] is False
    assert seen["t2"]["first_stale"] is True
    assert vt._current_turn_image_paths.get() is None
