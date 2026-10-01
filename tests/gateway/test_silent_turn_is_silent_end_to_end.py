"""Silence must be silent END TO END, not silent only in its final phase.

Live incident (group "the crew chat", 2026-09-20 — six participants). Every ignored
message still got a "Trabajando en ello" bubble, and one turn shipped the silence
REASONING itself: "Banter grupal, no dirigido a mí. Me quedo callado."

Two independent defects, one symptom:

1. The plain-mode progress lane had no reply-policy awareness. It fires on
   tool.started, long before the turn decides to stay silent, and on a no-edit
   surface like WhatsApp that line IS a posted message — so the bot announced its
   presence in exactly the conversations the reply gate exists to keep it out of.
   Streaming was already suppressed for this class ("streamed text IS delivery");
   the progress lane is the same delivery door and was missing the same gate.

2. The leaked-silence-narration backstop knew the Spanish noun ("silencio") but
   not the "callado" idiom or "no dirigido a mí", so the actual leak text scored
   ZERO signals on every tier and shipped verbatim.
"""

import pytest

from gateway.reply_policy import ReplyPolicy
from gateway.response_filters import _is_leaked_silence_narration


class _Source:
    def __init__(self, policy):
        self.reply_policy = policy
        self.platform = "whatsapp"
        self.chat_id = "999999999999999999@g.us"


class _Runner:
    """Minimal stand-in exposing the real predicate under test."""

    def __init__(self, mode):
        self.config = type("Cfg", (), {"reply_gate_mode": mode})()

    def _is_reply_tool_gated(self, source):
        from gateway.run_turn import GatewayTurnMixin

        return GatewayTurnMixin._is_reply_tool_gated(self, source)


@pytest.mark.parametrize(
    "policy,gated",
    [
        (ReplyPolicy.FREE_RESPONSE, True),
        (ReplyPolicy.DM, False),
        (ReplyPolicy.MENTION_GATED, False),
    ],
)
def test_progress_suppression_tracks_the_reply_gate(policy, gated):
    """The ack is suppressed exactly where free text is NOT auto-delivered.

    A DM must keep its progress ack — the user asked directly and silence there
    would look like a hang. A mention-gated group already filtered at ingestion,
    so there is no reply ambiguity to protect.
    """
    assert _Runner("tool")._is_reply_tool_gated(_Source(policy)) is gated


def test_progress_ack_survives_when_the_gate_is_off():
    """Prompt mode (the default) must be untouched: this is opt-in machinery."""
    assert _Runner("prompt")._is_reply_tool_gated(_Source(ReplyPolicy.FREE_RESPONSE)) is False


@pytest.mark.parametrize(
    "text",
    [
        "Banter grupal, no dirigido a mí. Me quedo callado.",  # the live leak, verbatim
        "Me quedo callado.",
        "Mejor me quedo callada, no va dirigido a mí.",
        "Audio al grupo, no dirigido a mí. Me mantengo callado.",
    ],
)
def test_spanish_callado_narration_is_caught(text):
    assert _is_leaked_silence_narration(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Dana, ya quedó listo el reporte ✅",
        "Robin, te mando el resumen en un momento.",
        # "callado" as ordinary narration ABOUT someone else, not a verdict.
        "El niño se quedó callado cuando vio el pastel 😂",
        # Declines one thing, then answers — speaks TO someone, so it is a real reply.
        "Prefiero quedarme callado en esa discusión, pero tu punto es válido.",
        "¡Viva México! 🎉 Que viva la independencia, Dana.",
        # Second person: "a ti", not "a mí" — addressed to a human.
        "No dirigido a ti, sino al grupo: mañana salimos a las 8.",
    ],
)
def test_genuine_spanish_replies_still_deliver(text):
    """Over-blocking is the worse failure: a held reply looks like a dead bot."""
    assert _is_leaked_silence_narration(text) is False


_LEAKS = [
    "Banter grupal, no dirigido a mí. Me quedo callado.",
    "Audio al grupo, no dirigido a mí. Me mantengo callado.",
]
_REPLIES = [
    "Dana, ya quedó listo el reporte ✅",
    "El niño se quedó callado cuando vio el pastel 😂",
    "No dirigido a ti, sino al grupo: mañana salimos a las 8.",
]


@pytest.mark.parametrize("text", _LEAKS)
def test_leaked_narration_is_treated_as_silence_by_the_delivery_predicate(text):
    """The helper is only protective if the delivery path calls it: the predicate every
    delivery seam consults must classify leaked stay-silent reasoning as silence."""
    from gateway.response_filters import is_intentional_silence_response

    assert is_intentional_silence_response(text) is True


@pytest.mark.parametrize("text", _REPLIES)
def test_genuine_replies_are_not_silence_to_the_delivery_predicate(text):
    from gateway.response_filters import is_intentional_silence_response

    assert is_intentional_silence_response(text) is False


def test_silent_turn_classifier_suppresses_leaked_narration():
    """End of the chain the runner uses: a successful turn whose final text is leaked
    reasoning is an intentional silence, so nothing is delivered to the group. A failed
    turn is never reclassified as silence."""
    from gateway.response_filters import is_intentional_silence_agent_result

    assert is_intentional_silence_agent_result({"final_response": _LEAKS[0]}, _LEAKS[0]) is True
    assert is_intentional_silence_agent_result({"final_response": _REPLIES[0]}, _REPLIES[0]) is False
    assert is_intentional_silence_agent_result({"failed": True}, _LEAKS[0]) is False
