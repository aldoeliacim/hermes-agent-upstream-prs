"""Regression: the reply-gate free-text tail must be deduped when a
send_message(target="current") already delivered this turn, even if the
turn-scoped reply_gate_tool_sends counter reads 0.

CONFIRMED production leak (2026-07-12, WhatsApp family group "TAMHAL Y JVic",
JID 120363555000000002@g.us): the model correctly replied to a batch of
photos by calling send_message(target="current", message="¡Ayyy, quedaron
buenísimas...") which delivered successfully (DB msg 199272 tool_call ->
199274 result {"success": true, ..., "mirrored": true}), then emitted a
trailing status narration "Enviado. ✅ Comenté las fotos de las escaleras..."
The reply-gate logged `tool_sends=0 outcome=undecided_delivered` and
double-sent that "Enviado ✅" tail into the family group as a second message.

The dedup guard (gateway/run.py: `if _delivered_via_tool: suppress`) exists
and its unit tests pass — but it keyed SOLELY on the turn-scoped
`source.reply_gate_tool_sends` counter, incremented by
tools/approval.py::note_current_message_delivered() inside the tool-executor
worker thread. That counter read back as 0 at the post-turn delivery
decision despite the transcript proving the send succeeded (a
thread/context-boundary fragility that reproduces only under the real
concurrent/coalesced-image turn conditions, not in the isolated unit test).

Fix: gateway/run.py::_transcript_has_current_chat_send() scans the actual
turn transcript for a successful send_message tool result as a
defense-in-depth dedup signal, OR-ed with the counter. The transcript is
ground truth; a live turn can only send with target="current" (the
live-turn scope guard refuses explicit cross-target sends mid-turn), so any
successful send_message result in a live turn is necessarily a current-chat
delivery and safe to dedup against.

These tests lock the transcript-scan helper's behavior so the double-send
class cannot silently regress.
"""

from __future__ import annotations

import gateway.run as gateway_run

_scan = gateway_run.GatewayRunner._transcript_has_current_chat_send


# The exact leak transcript shape (mirrors DB msgs 199272-199275).
_LEAK_TRANSCRIPT = [
    {"role": "user", "content": "[image received]"},
    {
        "role": "assistant",
        "content": "¡Ayyy, quedaron buenísimas estas fotos!",
        "tool_calls": [
            {
                "id": "t1",
                "function": {
                    "name": "send_message",
                    "arguments": '{"target": "current", "message": "¡Ayyy..."}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_name": "send_message",
        "name": "send_message",
        "tool_call_id": "t1",
        "content": '{"success": true, "platform": "whatsapp", '
                   '"chat_id": "120363555000000002@g.us", "mirrored": true}',
    },
    {"role": "assistant", "content": "Enviado. ✅ Comenté las fotos..."},
]


def test_detects_successful_current_chat_send_in_transcript():
    """The exact 2026-07-12 leak transcript must be recognized as a delivery
    so the trailing 'Enviado ✅' tail is deduped."""
    assert _scan(_LEAK_TRANSCRIPT) is True


def test_empty_transcript_is_not_a_delivery():
    assert _scan([]) is False
    assert _scan(None) is False


def test_non_send_tool_is_not_a_delivery():
    """A successful DIFFERENT tool (terminal, etc.) must not trigger the
    send dedup — only send_message counts."""
    msgs = [{"role": "tool", "tool_name": "terminal",
             "content": '{"success": true, "output": "ok"}'}]
    assert _scan(msgs) is False


def test_failed_send_is_not_a_delivery():
    """A send_message that FAILED (success:false) must not dedup the tail —
    the reply never actually went out, so the free-text tail is the only
    delivery and must be allowed through."""
    msgs = [{"role": "tool", "tool_name": "send_message",
             "content": '{"success": false, "error": "bridge down"}'}]
    assert _scan(msgs) is False


def test_accepts_name_only_tool_message_shape():
    """Some synthetic tool-result paths set only 'name', not 'tool_name'
    (agent/conversation_loop.py line ~5314). The scan must accept either
    key so those shapes still dedup correctly."""
    msgs = [{"role": "tool", "name": "send_message",
             "content": '{"success": true, "chat_id": "x@g.us"}'}]
    assert _scan(msgs) is True


def test_malformed_tool_content_is_skipped_not_raised():
    """A tool message with non-JSON content must be skipped silently, never
    raise — a bookkeeping scan must never break delivery."""
    msgs = [
        {"role": "tool", "tool_name": "send_message", "content": "not json {{{"},
        {"role": "tool", "tool_name": "send_message",
         "content": '{"success": true}'},
    ]
    # The malformed one is skipped; the valid one still triggers True.
    assert _scan(msgs) is True


def test_non_dict_messages_are_skipped():
    """Defensive: a transcript containing non-dict entries (strings, None)
    must not raise."""
    msgs = ["a string", None, 42,
            {"role": "tool", "tool_name": "send_message",
             "content": '{"success": true}'}]
    assert _scan(msgs) is True


def test_send_message_target_guidance_discourages_post_send_narration():
    """The send_message tool 'target' description must explicitly tell the
    model NOT to add a confirmation/status tail after target='current'. This
    is the source-level fix for the 'Enviado ✅' narration the model emitted
    on ~26% of group turns (2026-07-13 2-day analysis). If this guidance is
    removed the model reverts to double-message narration, so lock it."""
    from tools.send_message_tool import SEND_MESSAGE_SCHEMA
    target_desc = SEND_MESSAGE_SCHEMA["parameters"]["properties"]["target"]["description"]
    lowered = target_desc.lower()
    # Must mention the anti-narration rule and at least one exemplar token.
    assert "do not" in lowered
    assert "confirmation" in lowered or "status line" in lowered
    assert "enviado" in lowered or "sent" in lowered
    assert "duplicate" in lowered


# --- Current-turn scoping (2026-07-17 poisoned-history regression) ---------
#
# CONFIRMED production bug (2026-07-17, WhatsApp DM
# agent:main:whatsapp:dm:5215551000001, session 20260101_000000_abc123):
# a long-lived DM had 3,455 retained messages including 32 historical
# send_message success:true results from prior turns. Because the scan
# received the FULL retained conversation (not just this turn's tail), it
# returned True on EVERY turn, so _delivered_via_tool was permanently set
# and every real plain-text reply was suppressed as a phantom duplicate —
# the user got total silence in their primary channel, mode-independent
# (reply_gate_mode had no effect because tool_gated was False for the DM;
# suppression came solely through the dedup guard). Fix: scope the scan to
# messages after the LAST user-role message (a gateway turn always begins
# with the inbound user message).


def test_stale_prior_turn_send_does_not_dedup_current_turn():
    """A send_message success from an EARLIER turn (before the last user
    message) must NOT be treated as this turn's delivery. Otherwise a
    long-lived session is permanently silenced after its first tool send."""
    transcript = [
        # --- prior turn: a real successful send, now stale history ---
        {"role": "user", "content": "earlier question"},
        {
            "role": "tool",
            "tool_name": "send_message",
            "content": '{"success": true, "chat_id": "5215551000001@s.whatsapp.net"}',
        },
        {"role": "assistant", "content": "earlier reply"},
        # --- THIS turn: inbound user msg, then only a plain-text answer,
        #     NO send_message call. Must be delivered, not deduped. ---
        {"role": "user", "content": "Ya?"},
        {"role": "assistant", "content": "sí, aquí estoy"},
    ]
    assert _scan(transcript) is False


def test_current_turn_send_after_last_user_still_dedups():
    """The legitimate same-turn dedup must still fire: a send in THIS turn
    (after the last user message) counts, even when stale prior-turn sends
    also exist in the retained history."""
    transcript = [
        {"role": "user", "content": "earlier question"},
        {
            "role": "tool",
            "tool_name": "send_message",
            "content": '{"success": true, "chat_id": "x@g.us"}',
        },
        {"role": "assistant", "content": "earlier reply"},
        # --- THIS turn: user msg + a real current-turn send ---
        {"role": "user", "content": "another question"},
        {
            "role": "tool",
            "tool_name": "send_message",
            "content": '{"success": true, "chat_id": "x@g.us"}',
        },
        {"role": "assistant", "content": "trailing narration to dedup"},
    ]
    assert _scan(transcript) is True


def test_many_stale_sends_then_clean_turn_is_not_deduped():
    """Direct model of the live incident: many historical successful sends
    followed by a clean current turn with no send must deliver."""
    history = []
    for i in range(32):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({
            "role": "tool",
            "tool_name": "send_message",
            "content": '{"success": true, "chat_id": "5215551000001@s.whatsapp.net"}',
        })
        history.append({"role": "assistant", "content": f"a{i}"})
    # current turn: user asks, model answers in plain text, no send
    history.append({"role": "user", "content": "Ya?"})
    history.append({"role": "assistant", "content": "aquí estoy"})
    assert _scan(history) is False


def test_no_user_message_falls_back_to_whole_scan():
    """Defensive: if the transcript somehow has no user-role message, the
    scan falls back to scanning the whole list (prior behavior), so a send
    is still detected rather than silently ignored."""
    msgs = [{"role": "tool", "tool_name": "send_message",
             "content": '{"success": true}'}]
    assert _scan(msgs) is True
