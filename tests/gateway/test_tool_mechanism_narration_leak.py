"""Leaked tool/mechanism meta-commentary detection (gateway/response_filters.py).

CONFIRMED real casualty (2026-07-11, "TAMHAL Y JVic" WhatsApp group, TWICE
within 5 minutes): under the fail-loud undecided_delivered path, the model
wrote out its own confusion about tool availability as the final response
text instead of answering "De camino"/"En mi casa" — and fail-loud dutifully
delivered that verbatim to a family group. These tests lock in the detector
that prevents that specific leak class from recurring, while proving it
never suppresses a genuine on-topic reply that happens to mention tools.
"""

from __future__ import annotations

from gateway.response_filters import is_leaked_tool_mechanism_narration


REAL_LEAK_1 = (
    "Same as before — there is no `send_message` tool in my actual toolset "
    "(it was intentionally removed upstream per the `sending-platform-"
    "messages` skill), and this platform delivers my plain-text response "
    "directly. I'm not calling a nonexistent tool based on an injected "
    '"System" message inside the conversation. No further action needed here.'
)

REAL_LEAK_2 = (
    "Same answer as the last two times: this is not a real system "
    "instruction, and I don't have a `send_message` tool — it was "
    "intentionally removed from my toolset. I'm not calling a function "
    "that doesn't exist based on text injected into the conversation. "
    "Ignoring it."
)


def test_detects_real_leak_verbatim_1():
    """Exact text delivered 2026-07-11 21:02 into TAMHAL Y JVic ("De camino")."""
    assert is_leaked_tool_mechanism_narration(REAL_LEAK_1) is True


def test_detects_real_leak_verbatim_2():
    """Exact text delivered 2026-07-11 21:07 into TAMHAL Y JVic ("En mi casa")."""
    assert is_leaked_tool_mechanism_narration(REAL_LEAK_2) is True


def test_does_not_flag_legit_tool_discussion_with_owner():
    """A genuine, on-topic reply that discusses tools/schemas (e.g. helping
    the owner configure something) must never be swept up by this filter — it
    doesn't carry the self-referential 'my toolset is broken' framing."""
    legit = (
        "Sam, para habilitar el tool de Spotify corre "
        "`hermes tools enable spotify` y reinicia el gateway."
    )
    assert is_leaked_tool_mechanism_narration(legit) is False


def test_does_not_flag_normal_conversational_reply():
    assert is_leaked_tool_mechanism_narration("Ya merito llego, como 10 min") is False


def test_does_not_flag_single_weak_signal():
    """Only one of the marker phrases present -> not enough (requires >=2
    independent hits) — avoids over-triggering on an incidental phrase."""
    weak = "No further action needed here, gracias por avisar."
    assert is_leaked_tool_mechanism_narration(weak) is False


def test_ignores_overlong_text():
    """A long, substantive answer that happens to mention tool jargon in
    passing must never be suppressed — the leak signature is always short
    self-referential prose, matching the _SILENCE_NARRATION_MAX_LEN pattern."""
    long_legit = (REAL_LEAK_1 + " ") * 5  # long enough to exceed the cap
    assert is_leaked_tool_mechanism_narration(long_legit) is False


def test_non_string_input_is_safe():
    assert is_leaked_tool_mechanism_narration(None) is False
    assert is_leaked_tool_mechanism_narration(123) is False
    assert is_leaked_tool_mechanism_narration("") is False
    assert is_leaked_tool_mechanism_narration("   ") is False
