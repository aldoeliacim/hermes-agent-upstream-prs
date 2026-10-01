"""Regression: the queued-follow-up direct-send branch must respect the
reply gate's tool-send dedup.

CONFIRMED production leak (2026-08-20, WhatsApp group): a voice-note turn
delivered its real reply via send_message (tool result success in the
transcript), and the reply gate correctly suppressed the model's free-text
tail — internal narration ("Respondí con calidez a la historia de ...").
Seconds later another user's message arrived and was QUEUED. The queued
follow-up branch in gateway/run.py then sent `final_response` directly via
adapter.send(), bypassing the reply gate entirely — the suppressed internal
narration leaked verbatim into the group.

The branch previously filtered only intentional-silence markers. It must
apply the same authoritative dedup the tool-mode gate uses: a successful
current-chat send in the turn transcript means the free-text tail is
narration, never a deliverable.
"""

from gateway.run import GatewayRunner

_action = GatewayRunner._queued_followup_tail_action
_scan = GatewayRunner._transcript_has_current_chat_send


def _transcript_with_successful_send():
    return [
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "send_message",
                                      "arguments": '{"target": "current", "message": "respuesta real"}'}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "send_message",
         "content": '{"success": true, "platform": "whatsapp", "chat_id": "1@g.us", "message_id": "X", "mirrored": true}'},
        {"role": "assistant", "content": "Respondí con calidez a la historia. Conversación en modo casual."},
    ]


def test_incident_2026_08_20_narration_tail_is_deduped():
    """Tool send in transcript + queued follow-up => tail suppressed."""
    delivered = _scan(_transcript_with_successful_send())
    assert delivered  # transcript proves the send
    assert _action(
        intentional_silence=False,
        delivered_via_tool=delivered,
        first_response="Respondí con calidez a la historia. Conversación en modo casual.",
        already_streamed=False,
    ) == "tool_dedup"


def test_no_tool_send_still_delivers_real_answer():
    assert _action(
        intentional_silence=False,
        delivered_via_tool=False,
        first_response="respuesta genuina",
        already_streamed=False,
    ) == "send"


def test_intentional_silence_wins_over_everything():
    assert _action(
        intentional_silence=True,
        delivered_via_tool=True,
        first_response="[SILENT]",
        already_streamed=False,
    ) == "silence"


def test_already_streamed_or_empty_skips():
    assert _action(
        intentional_silence=False,
        delivered_via_tool=False,
        first_response="ya entregado por stream",
        already_streamed=True,
    ) == "skip"
    assert _action(
        intentional_silence=False,
        delivered_via_tool=False,
        first_response="",
        already_streamed=False,
    ) == "skip"


def test_already_streamed_but_has_media_still_sends():
    """Streamed text does NOT suppress an explicit attachment.

    The delivery block passes text_already_delivered=True so it skips the
    duplicate text and only uploads the media. Regression guard for
    test_run_agent_queued_message_delivers_streamed_first_response_media,
    which the cherry-picked helper (media-blind) broke.
    """
    assert _action(
        intentional_silence=False,
        delivered_via_tool=False,
        first_response="here you go MEDIA:/tmp/x.png",
        already_streamed=True,
        has_media=True,
    ) == "send"
    # ...but a tool-send still wins: narration tail with media stays deduped.
    assert _action(
        intentional_silence=False,
        delivered_via_tool=True,
        first_response="Respondí con calidez MEDIA:/tmp/x.png",
        already_streamed=True,
        has_media=True,
    ) == "tool_dedup"
