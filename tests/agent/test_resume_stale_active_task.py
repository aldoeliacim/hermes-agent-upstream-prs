"""Regression coverage for #35344: a resumed session must not let a stale
historical task snapshot from an inherited compaction handoff hijack the reply to a
new, unrelated user message.

The failure mode (real report): a lineage was compacted, producing a handoff
whose historical task snapshot described task A. The lineage was resumed later and
the user asked about an unrelated task B. The model answered with A because
the handoff's resume directive outranked the fresh ask.

On a resumed lineage the inherited handoff must be detected as a summary
(state to rehydrate), never re-serialized as a fresh user turn.
"""

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    ContextCompressor,
)




def test_latest_message_wins_over_inherited_active_task():
    """The handoff must explicitly privilege the latest user message over a
    stale historical task snapshot — the core #35344 contract."""
    """The handoff must explicitly privilege the latest user message over a
    stale historical task snapshot — the core #35344 contract."""
    lower = SUMMARY_PREFIX.lower()
    assert "latest real user message" in lower
    assert HISTORICAL_TASK_HEADING.lower() in lower
    # Conflict-resolution must be explicit, not implied.
    assert "precedence immediately" in lower
    assert "never revive cancelled or completed tasks" in lower
    # The "consistent -> use as background" carveout licensed stale-task
    # resumption on topic overlap (#41607, #38364) — it must stay gone.
    assert "you may use the summary as background" not in lower
    assert "topic overlap" in lower
    # #80622: empty-after-handoff must not resume historical work.
    assert "if no user message appears after this summary" in lower











def test_inherited_handoff_detected_in_resumed_protected_head():
    """On a resumed lineage the handoff commonly sits right after the system
    prompt (in the protected head). ``_find_latest_context_summary`` must
    detect it there so re-compaction rehydrates state from it rather than
    serializing it as a fresh user turn (which is what let the stale Active
    Task read as live intent)."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nUser asked: 'task A'"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Unrelated task B: what's the capital of France?"},
    ]
    # Search the whole post-system range.
    idx, body = ContextCompressor._find_latest_context_summary(
        messages, 1, len(messages)
    )
    assert idx == 1, "handoff in protected head must be found"
    assert "task A" in body
    # The detected body is stripped of the prefix (treated as state, not a
    # standalone instruction message).
    assert not body.startswith(SUMMARY_PREFIX)


