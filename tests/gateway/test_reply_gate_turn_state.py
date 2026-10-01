"""Reply-gate v2: TurnDeliveryState is the identity-stable per-inbound-message
delivery ledger that fixes the ~44% counter-miss rate of the v1 per-source
counters.

ROOT CAUSE (measured 2026-07-13, WhatsApp group "TAMHAL Y JVic"): a single
inbound message's turn can span MULTIPLE _run_agent_inner invocations (the
re-entrant queued-follow-up chain), and each recursive run (a) reset the
source counters and (b) bound a DIFFERENT `next_source` object than the outer
post-turn block reads. A send_message(target="current") firing in a follow-up
run incremented a source object the gateway never inspected, so the reply-gate
saw tool_sends=0 and either double-sent a free-text tail or (before the
transcript-scan safety net) dropped a decision. The forced-decision nudge, which
also read the per-source counters, never fired once in the entire burn-in.

FIX: one TurnDeliveryState created at the OUTER per-inbound-message scope,
bound to a contextvar for the whole (re-entrant) chain, mutated in place by the
send_message tool (attribute writes survive the executor-thread -> asyncio-loop
boundary), read by identity by both the post-turn dedup block and the nudge.

These tests lock the invariant that made v1 fail: a send under a DIFFERENT
source object, on a worker thread, in a follow-up run, must still be visible to
the outer scope via the single shared TurnDeliveryState.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import tools.approval as ap
from tools.thread_context import propagate_context_to_thread


class _Src:
    """Minimal stand-in for a gateway message source (the object that diverges
    across the re-entrant follow-up chain in production)."""

    def __init__(self):
        self.reply_gate_tool_sends = 0
        self.reply_gate_decided_silent = 0


def test_send_under_diverged_source_on_worker_thread_is_seen_by_outer_state():
    """THE regression: outer scope binds one TurnDeliveryState; a follow-up
    run binds a DIFFERENT source object and its send fires on a worker thread
    (copy_context). The outer TurnDeliveryState must still count it — the exact
    scenario v1's per-source counter missed ~44% of the time."""
    state = ap.TurnDeliveryState()
    tok_state = ap.set_turn_delivery_state(state)
    try:
        # First run: source A, no send.
        src_a = _Src()
        tok_a = ap.set_current_message_source(src_a)
        ap.reset_current_message_source(tok_a)

        # Follow-up run: DIFFERENT source B; send fires on a worker thread.
        src_b = _Src()
        tok_b = ap.set_current_message_source(src_b)

        def worker():
            ap.note_current_message_delivered()

        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(propagate_context_to_thread(worker)).result()
        ap.reset_current_message_source(tok_b)

        # v1 would read src_a (the outer object) -> 0 (the bug).
        assert src_a.reply_gate_tool_sends == 0
        # v2 reads the single shared state -> 1 (fixed).
        assert state.tool_sends == 1
    finally:
        ap.reset_turn_delivery_state(tok_state)


def test_silent_decision_accumulates_on_shared_state():
    state = ap.TurnDeliveryState()
    tok_state = ap.set_turn_delivery_state(state)
    try:
        src = _Src()
        tok = ap.set_current_message_source(src)
        ap.note_current_message_silent()
        ap.reset_current_message_source(tok)
        assert state.decided_silent == 1
        assert state.tool_sends == 0
    finally:
        ap.reset_turn_delivery_state(tok_state)


def test_counts_accumulate_across_multiple_runs_without_reset():
    """The state is NOT reset per _run_agent_inner, so sends across several
    follow-up runs of one inbound message accumulate into the same ledger."""
    state = ap.TurnDeliveryState()
    tok_state = ap.set_turn_delivery_state(state)
    try:
        for _ in range(3):
            src = _Src()  # a fresh (diverged) source each run
            tok = ap.set_current_message_source(src)
            ap.note_current_message_delivered()
            ap.reset_current_message_source(tok)
        assert state.tool_sends == 3
    finally:
        ap.reset_turn_delivery_state(tok_state)


def test_note_functions_are_noops_outside_a_gateway_turn():
    """No TurnDeliveryState and no source bound (CLI/cron/subagent) -> the note
    functions must not raise."""
    # Ensure nothing bound.
    assert ap.get_turn_delivery_state() is None
    ap.note_current_message_delivered()  # must not raise
    ap.note_current_message_silent()  # must not raise


def test_legacy_source_mirror_still_updated_for_backcompat():
    """During migration, the note functions also mirror onto the source-object
    counters so any lingering reader keeps working."""
    state = ap.TurnDeliveryState()
    tok_state = ap.set_turn_delivery_state(state)
    try:
        src = _Src()
        tok = ap.set_current_message_source(src)
        ap.note_current_message_delivered()
        ap.reset_current_message_source(tok)
        # Both the new ledger and the legacy mirror reflect the send.
        assert state.tool_sends == 1
        assert src.reply_gate_tool_sends == 1
    finally:
        ap.reset_turn_delivery_state(tok_state)
