"""Integration coverage for the reply-decision forced-decision stop gate (#921).

Exercises the REAL ``agent.turn_stop_gates.apply_stop_gates`` seam — the same
entry the conversation loop calls when the model stops with a text answer — to
prove the tool-gated free-response turn is:

  * nudged EXACTLY ONCE when it stops undecided (no delivery, no explicit
    silence), and that the nudge row is request-copy-only scaffolding — flagged
    so it never enters durable session persistence or compaction anchoring;
  * resolved WITHOUT reaching the fail-loud ``undecided`` fallback once the
    model then makes an explicit decision (a current-target send OR explicit
    silence), because the same gate goes inert the moment the turn is decided.

Only the reply-decision gate must be able to fire here, so the sibling stop
gates are pinned inert: ``verify_on_stop`` off, no kanban task, no file
mutations. ``HERMES_REPLY_DECISION_NUDGE`` is set explicitly (env wins over any
ambient config) so the run is hermetic regardless of the developer's config.
"""

from __future__ import annotations

import pytest

import tools.approval as approval
from agent.conversation_compression import _is_real_user_message
from agent.reply_decision_stop import _NUDGE_TEXT, DEFAULT_MAX_REPLY_DECISION_NUDGES
from agent.session_persistence import _is_ephemeral_scaffolding
from agent.turn_stop_gates import apply_stop_gates


class _FakeAgent:
    """Minimal stand-in exposing only the surface ``apply_stop_gates`` touches.

    ``_append_interim_answer`` persists+emits the assistant candidate and
    ``_continue`` appends the synthetic nudge row; both are recorded here so the
    test can inspect exactly what landed in the transcript.
    """

    def __init__(self):
        self.session_id = "s-reply-decision"
        self.platform = "telegram"
        self.model = "test"
        self._reply_decision_nudges = 0
        self._session_messages = None
        self.emitted = []
        self.flushed = 0

    # --- surface consumed by _append_interim_answer / _continue --- #
    def _emit_interim_assistant_message(self, final_msg):
        self.emitted.append(final_msg)

    def _flush_messages_to_session_db(self, messages, conversation_history):
        self.flushed += 1

    def _interim_content_was_streamed(self, _text):
        return False


@pytest.fixture
def hermetic_stop_gates(monkeypatch):
    """Pin every sibling stop gate inert so only the reply-decision gate can fire."""
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_REPLY_DECISION_NUDGE", "1")
    yield


@pytest.fixture
def tool_gated_turn():
    """Arm the turn-scoped reply-gate context: tool-gated + an undecided ledger."""
    state = approval.TurnDeliveryState()
    gate_tok = approval.set_reply_tool_gated_turn(True)
    state_tok = approval.set_turn_delivery_state(state)
    try:
        yield state
    finally:
        approval.reset_turn_delivery_state(state_tok)
        approval.reset_reply_tool_gated_turn(gate_tok)


def _run_gate(agent, messages, final_response):
    final_msg = {"role": "assistant", "content": final_response}
    return apply_stop_gates(
        agent,
        final_msg,
        final_response=final_response,
        messages=messages,
        conversation_history=[],
        pending_verification_response=None,
        pending_verification_response_previewed=False,
    )


def test_undecided_turn_is_nudged_exactly_once(
    hermetic_stop_gates, tool_gated_turn
):
    agent = _FakeAgent()
    messages = []

    verdict = _run_gate(agent, messages, "Here is my complete answer.")

    # The gate held the turn open for one more decision round.
    assert verdict.continue_turn is True
    # final_response is cleared (kept only as a budget-exhaustion fallback), so
    # the finalizer can tell this gate from an error exit.
    assert verdict.final_response is None
    assert verdict.pending_verification_response == "Here is my complete answer."
    # Exactly one nudge issued.
    assert agent._reply_decision_nudges == 1


def test_nudge_row_is_request_copy_only(hermetic_stop_gates, tool_gated_turn):
    """The nudge lands as flagged synthetic scaffolding: it must never enter
    durable persistence or compaction anchoring."""
    agent = _FakeAgent()
    messages = []

    _run_gate(agent, messages, "answer body")

    # transcript now holds: the assistant candidate, then the synthetic nudge row.
    nudge_rows = [
        m for m in messages
        if m.get("role") == "user" and m.get("_reply_decision_synthetic")
    ]
    assert len(nudge_rows) == 1
    nudge = nudge_rows[0]
    assert nudge["content"] == _NUDGE_TEXT

    # Persistence treats it as ephemeral scaffolding (never written durably).
    assert _is_ephemeral_scaffolding(nudge) is True
    # Compaction does NOT count it as a real user turn (no anchor short-circuit).
    assert _is_real_user_message(nudge) is False

    # The assistant candidate itself is NOT synthetic — it is real content the
    # user should see, persisted+emitted as an interim answer.
    assert agent.emitted and agent.emitted[0]["content"] == "answer body"
    assert agent.flushed == 1


def test_explicit_delivery_decision_resolves_without_fail_loud(
    hermetic_stop_gates, tool_gated_turn
):
    """After the nudge, an explicit current-target send (tool_sends>0) makes the
    gate go inert: the turn RESOLVES (continue_turn=False) instead of nudging
    again or reaching the fail-loud undecided path."""
    agent = _FakeAgent()
    messages = []

    # Round 1: undecided → nudged once.
    v1 = _run_gate(agent, messages, "answer")
    assert v1.continue_turn is True
    assert agent._reply_decision_nudges == 1

    # The model now actually delivers this turn.
    tool_gated_turn.tool_sends = 1
    assert tool_gated_turn.delivery_decided is True

    # Round 2: gate is inert → the turn resolves, no second nudge, no fail-loud.
    v2 = _run_gate(agent, messages, "answer")
    assert v2.continue_turn is False
    assert v2.final_response == "answer"
    assert agent._reply_decision_nudges == 1  # still exactly one


def test_explicit_silence_decision_resolves_without_fail_loud(
    hermetic_stop_gates, tool_gated_turn
):
    """Explicit silence (decided_silent>0) is equally a decision: it resolves
    the turn without another nudge or the fail-loud undecided path."""
    agent = _FakeAgent()
    messages = []

    _run_gate(agent, messages, "answer")  # round 1: nudged
    assert agent._reply_decision_nudges == 1

    tool_gated_turn.decided_silent = 1
    assert tool_gated_turn.delivery_decided is True

    v2 = _run_gate(agent, messages, "answer")
    assert v2.continue_turn is False
    assert agent._reply_decision_nudges == 1


def test_nudge_is_capped_and_fails_open_to_undecided(
    hermetic_stop_gates, tool_gated_turn
):
    """A model that keeps stopping undecided is nudged at most ``cap`` times,
    then the gate fails open to the ORIGINAL undecided suppression (the loop
    resolves rather than nudging forever)."""
    agent = _FakeAgent()
    messages = []

    cap = DEFAULT_MAX_REPLY_DECISION_NUDGES
    # Each undecided stop below the cap issues a nudge.
    for i in range(cap):
        v = _run_gate(agent, messages, "answer")
        assert v.continue_turn is True
        assert agent._reply_decision_nudges == i + 1

    # Cap reached: the next undecided stop no longer nudges — turn resolves.
    v_final = _run_gate(agent, messages, "answer")
    assert v_final.continue_turn is False
    assert agent._reply_decision_nudges == cap


def test_gate_inert_when_not_tool_gated(hermetic_stop_gates):
    """Outside a tool-gated turn the gate never arms — a plain DM/mention turn
    resolves immediately with no nudge."""
    agent = _FakeAgent()
    messages = []
    # No tool_gated context set → is_reply_tool_gated_turn() is False.
    v = _run_gate(agent, messages, "answer")
    assert v.continue_turn is False
    assert agent._reply_decision_nudges == 0
