"""Regression contracts for Codex Responses completion integrity.

Three failure modes where a truncated / incomplete Codex response was accepted
as a finished answer:

1. The SSE stream ends (EOF) after a partial final-answer delta but before the
   terminal ``response.completed`` frame — the synthesized message must be
   ``incomplete``, never silently ``completed``.
2. The response envelope reports ``status="incomplete"`` (e.g. max_output_tokens)
   while its final message item still says ``status="completed"`` — the envelope
   owns completion, so ``_normalize_codex_response`` must return
   ``finish_reason="incomplete"``.
3. ``looks_like_codex_intermediate_ack`` must scan only the CURRENT turn for a
   prior tool result; a completed earlier turn's tool row must not suppress
   continuation-ack detection for a fresh request.
"""

from types import SimpleNamespace

from agent.agent_runtime_helpers import looks_like_codex_intermediate_ack
from agent.codex_responses_adapter import _normalize_codex_response
from agent.codex_runtime import _consume_codex_event_stream


def test_eof_after_final_delta_is_incomplete_not_completed():
    """A lost terminal SSE frame must never turn a partial final into success."""
    response = _consume_codex_event_stream(
        iter([
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(type="message", phase="final_answer"),
            ),
            SimpleNamespace(type="response.output_text.delta", delta="Partial result"),
        ]),
        model="gpt-test",
    )

    assert response.status == "incomplete"
    assert response.output[0].status == "incomplete"
    assert response.output_text == "Partial result"


def test_terminal_completed_frame_stays_completed():
    """A real response.completed frame must still resolve as completed."""
    response = _consume_codex_event_stream(
        iter([
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(type="message", phase="final_answer"),
            ),
            SimpleNamespace(type="response.output_text.delta", delta="Full answer"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", usage=None, id="resp_1"),
            ),
        ]),
        model="gpt-test",
    )

    assert response.status == "completed"
    assert response.output[0].status == "completed"
    assert response.output_text == "Full answer"


def test_response_incomplete_overrides_completed_final_item():
    """The response envelope, not an individual item, owns completion status."""
    response = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=[SimpleNamespace(
            type="message",
            role="assistant",
            phase="final_answer",
            status="completed",
            content=[SimpleNamespace(type="output_text", text="Partial result")],
        )],
        output_text="Partial result",
    )

    _message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "incomplete"


def test_completed_envelope_still_stops():
    """A genuinely completed response must not be misclassified as incomplete."""
    response = SimpleNamespace(
        status="completed",
        output=[SimpleNamespace(
            type="message",
            role="assistant",
            phase="final_answer",
            status="completed",
            content=[SimpleNamespace(type="output_text", text="All done")],
        )],
        output_text="All done",
    )

    _message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "stop"


def test_old_tool_result_does_not_disable_current_turn_intent_ack():
    """A completed earlier turn cannot suppress ack detection for this request."""
    agent = SimpleNamespace(_strip_think_blocks=lambda text: text)
    current_user = "Review the repository in /app"
    messages = [
        {"role": "user", "content": "Earlier task"},
        {"role": "assistant", "content": "Done"},
        {"role": "tool", "content": "old result"},
        {"role": "user", "content": current_user},
    ]

    assert looks_like_codex_intermediate_ack(
        agent, current_user, "I'll inspect the repository first.", messages
    )


def test_current_turn_tool_result_still_disables_intent_ack():
    """A tool result WITHIN this turn still disables the ack (real work started)."""
    agent = SimpleNamespace(_strip_think_blocks=lambda text: text)
    current_user = "Review the repository in /app"
    messages = [
        {"role": "user", "content": current_user},
        {"role": "tool", "content": "current result"},
    ]

    assert not looks_like_codex_intermediate_ack(
        agent, current_user, "I'll inspect the repository first.", messages
    )
