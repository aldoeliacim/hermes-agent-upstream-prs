from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.response_filters import (
    SILENT_REPLY_TOKEN,
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def normalize_live_gateway_response(response, *, failed=False):
    """Test shim preserving the pre-refactor delivery-boundary contract on top
    of the current API. Upstream split ``normalize_live_gateway_response`` into
    ``is_intentional_silence_response`` (exact markers + leaked-narration
    backstop) gated by ``is_intentional_silence_agent_result`` (success-only).
    Exercising it here drives the REAL suppression path end to end.
    """
    if is_intentional_silence_agent_result({"failed": failed}, response):
        return ""
    return response if response is not None else ""


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        # Exact canonical markers (case/whitespace-insensitive) are suppressed.
        ("[SILENT]", ""),
        # Upstream narrowed silence to EXACT markers only: punctuation-wrapped or
        # prose placeholders are now DELIVERED, not special-cased away.
        ("(No message)", "(No message)"),
        ("`(No reply)`", "`(No reply)`"),
        ("**(No response generated)**", "**(No response generated)**"),
        ("(empty)", "(empty)"),
        ("[SILENT] means stay quiet", "[SILENT] means stay quiet"),
        ("No message received from Discord", "No message received from Discord"),
    ],
)
def test_normalize_live_gateway_response(raw_text, expected):
    assert normalize_live_gateway_response(raw_text) == expected


def test_normalize_live_gateway_response_preserves_failed_output():
    assert normalize_live_gateway_response("[SILENT]", failed=True) == "[SILENT]"


@pytest.mark.parametrize(
    "raw_text",
    [
        SILENT_REPLY_TOKEN,
        f"  {SILENT_REPLY_TOKEN}\n",
        SILENT_REPLY_TOKEN.lower(),
    ],
)
def test_canonical_silent_token_is_suppressed(raw_text):
    """The canonical NO_REPLY token (bare, case- and whitespace-insensitive)
    must always be suppressed.

    Locks the control-token contract so a future tweak to the canonicalizer
    can't silently start leaking the bare token into live chats.
    """
    assert normalize_live_gateway_response(raw_text) == ""


@pytest.mark.parametrize(
    "raw_text",
    [
        f"**{SILENT_REPLY_TOKEN}**",
        f"({SILENT_REPLY_TOKEN})",
    ],
)
def test_punctuation_wrapped_token_is_suppressed(raw_text):
    """Punctuation-wrapped canonical tokens ARE suppressed.

    Upstream's punctuation-tolerance fix (``_strip_edge_silence_punctuation``)
    strips stray edge PUNCTUATION (Unicode category ``P*`` — asterisks,
    parens, brackets) before canonicalizing, so a model emitting
    ``**NO_REPLY**`` instead of the exact marker is still recognized as
    intentional silence. This documents the current (post-tolerance-fix)
    contract; distinguish this from genuine prose placeholders like
    "(No message)" above, which are a different phrase entirely and remain
    deliverable regardless of edge-punctuation handling.
    """
    assert normalize_live_gateway_response(raw_text) == ""


def test_backtick_wrapped_token_is_not_suppressed():
    """Backtick-wrapped tokens are NOT stripped by the tolerance fix.

    The backtick (`) is Unicode category ``Sk`` (Symbol, modifier), not
    ``P*`` (Punctuation) — ``_strip_edge_silence_punctuation`` only strips
    ``P*`` characters (by design, to avoid over-eager stripping of
    non-punctuation symbols), so a backtick-wrapped token is delivered as
    prose rather than suppressed. This is a real edge case in the upstream
    tolerance fix's scope, not a bug in the merge of the two patches.
    """
    raw_text = f"`{SILENT_REPLY_TOKEN}`"
    assert normalize_live_gateway_response(raw_text) == raw_text


def test_canonical_silent_token_survives_when_failed():
    """A real generation failure must never be silently swallowed.

    failed=True means the empty output is a model/transport failure, not an
    intentional silent turn, so the gateway keeps the sentinel for the
    user-facing error path rather than suppressing it.
    """
    assert normalize_live_gateway_response(SILENT_REPLY_TOKEN, failed=True) == SILENT_REPLY_TOKEN


def test_prose_mentioning_canonical_token_is_delivered():
    """Ordinary prose that merely mentions NO_REPLY must still be delivered."""
    text = f"Return {SILENT_REPLY_TOKEN} when you have nothing to add."
    assert normalize_live_gateway_response(text) == text


# --- Leaked reply-gate reasoning backstop (regression for 2026-06-12) ---------
# The model decided to stay silent on ambient group banter but emitted its
# reply-gate REASONING as the turn's text instead of the canonical token, so a
# 185-char chain-of-thought shipped to a WhatsApp group. The exact-marker filter
# can't catch free-form narration; normalize_live_gateway_response() now has a
# precision backstop that suppresses leaked stay-silent decision narration while
# leaving genuine prose (including an owner-DM explanation of the mechanism)
# untouched.


def test_leaked_reply_gate_reasoning_is_suppressed():
    """The exact production leak must be suppressed, not delivered to the chat."""
    leak = (
        'This is human-to-human group banter ("compitas" catching up) — nobody '
        "addressed me, and another person is the obvious recipient. Per the reply "
        "gate, I stay silent here. No message sent."
    )
    assert normalize_live_gateway_response(leak) == ""


@pytest.mark.parametrize(
    "leak",
    [
        # jargon ("reply gate") + a first-person decision statement
        "Per the reply gate, no message sent.",
        "Reply gate: I stay silent on banter.",
        # no jargon, but two+ independent first-person decision statements
        "Nobody addressed me, I will stay quiet, and no message sent.",
        "I won't reply — nobody addressed me and no message is sent.",
    ],
)
def test_leaked_silence_narration_variants_are_suppressed(leak):
    assert normalize_live_gateway_response(leak) == ""


@pytest.mark.parametrize(
    "text",
    [
        # Owner-DM explanation of the mechanism — descriptive, NOT a decision log.
        "The reply gate has three checks: addressed, unique value, or "
        "human-to-human exchange.",
        # A single first-person reason, lightly stated, no jargon, under-corroborated.
        "I stayed out of it because nobody addressed me directly.",
        # A genuine Spanish group message (the kind that should ship).
        "De perlas, compitas. Aquí andamos con dos trabajos y el side project.",
        # Long genuine reply that happens to mention staying silent once.
        ("Great question about the architecture. " * 12)
        + "I will stay silent on pricing until we finalize.",
        # Mentioning the token in instructional prose.
        "Use the silence marker only when you truly have nothing to add.",
    ],
)
def test_genuine_prose_is_not_suppressed_by_silence_backstop(text):
    """The backstop must never eat a real message, especially an owner-facing
    explanation of why a turn stayed silent."""
    assert normalize_live_gateway_response(text) == text


def _make_runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = MagicMock()
    runner.session_store = MagicMock()
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner.adapters = {}
    runner._show_reasoning = False
    runner._session_db = None
    runner._set_session_env = MagicMock(return_value=[])
    runner._clear_session_env = MagicMock()
    runner._should_send_voice_reply = MagicMock(return_value=False)
    runner._deliver_media_from_response = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_handle_message_with_agent_suppresses_placeholder(monkeypatch):
    runner = _make_runner()

    session_entry = SimpleNamespace(
        session_id="sess-1",
        session_key="key-1",
        created_at=1,
        updated_at=2,
        was_auto_reset=False,
        last_prompt_tokens=0,
    )
    history = [{"role": "assistant", "content": "Earlier reply"}]

    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    runner._run_agent = AsyncMock(
        return_value={
            "final_response": SILENT_REPLY_TOKEN,
            "messages": history,
            "api_calls": 1,
            "last_prompt_tokens": 0,
        }
    )
    # _handle_message_with_agent now guards against stale runs via
    # _is_session_run_current(session_key, generation); treat this run as current.
    runner._is_session_run_current = MagicMock(return_value=True)

    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="chat-1",
        user_id="user-1",
        user_name="tester",
    )

    # build_session_context's return value feeds the byte-stable prompt-cache
    # key (upstream c0c76a471), which reads context.source plus the three
    # fields below. A bare {} stub raises AttributeError, so return a namespace
    # satisfying that contract, carrying the source this test actually drives.
    monkeypatch.setattr(
        "gateway.run.build_session_context",
        lambda *_a, **_kw: SimpleNamespace(
            source=source,
            connected_platforms=(),
            home_channels={},
            shared_multi_user_session=False,
        ),
    )
    monkeypatch.setattr("gateway.run.build_session_context_prompt", lambda *_a, **_kw: "")

    event = MessageEvent(text="test", message_type=MessageType.TEXT, source=source)

    result = await runner._handle_message_with_agent(event, source, "key-1", 1)

    # A bare silence marker on a human turn that expected a reply (this is a local DM, not a
    # reply-gated group) is replaced with a visible fallback (upstream 5ea8fb2b78): silence to
    # someone who addressed the agent reads as a dead bot. The placeholder "(No message)" user
    # row is still never persisted. Reply-gated group turns keep their silence; that side is
    # pinned in test_owner_register_gate_e2e.py::test_reply_gated_group_silence_stays_silent.
    from gateway.run_turn import _UNEXPECTED_SILENCE_REPLY

    assert result == _UNEXPECTED_SILENCE_REPLY
    appended = [call.args[1] for call in runner.session_store.append_to_transcript.call_args_list]
    assert any(entry["role"] == "user" for entry in appended)
    assert not any(entry.get("content") == "(No message)" for entry in appended)


@pytest.mark.asyncio
async def test_handle_message_with_agent_empty_failure_surfaces_notice(monkeypatch):
    """A genuine empty-generation FAILURE must still surface a user notice.

    Regression guard for #13248: intentional silence is suppressed, but a real
    failure (api_calls>0, no text, not a silence marker) must NOT be silently
    swallowed by the intentional-silence bypass — the user needs to know the
    turn produced nothing so they can retry.
    """
    runner = _make_runner()

    session_entry = SimpleNamespace(
        session_id="sess-2",
        session_key="key-2",
        created_at=1,
        updated_at=2,
        was_auto_reset=False,
        last_prompt_tokens=0,
    )
    history = [{"role": "assistant", "content": "Earlier reply"}]

    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    # Empty final_response with api_calls>0 and no silence marker == failure.
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "",
            "messages": history,
            "api_calls": 1,
            "last_prompt_tokens": 0,
        }
    )
    runner._is_session_run_current = MagicMock(return_value=True)

    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="chat-2",
        user_id="user-2",
        user_name="tester",
    )

    # build_session_context feeds the byte-stable prompt-cache key, which reads
    # context.source plus the three fields below. A bare {} stub raises
    # AttributeError at gateway/run.py:22446.
    monkeypatch.setattr(
        "gateway.run.build_session_context",
        lambda *_a, **_kw: SimpleNamespace(
            source=source,
            connected_platforms=(),
            home_channels={},
            shared_multi_user_session=False,
        ),
    )
    monkeypatch.setattr("gateway.run.build_session_context_prompt", lambda *_a, **_kw: "")

    event = MessageEvent(text="test", message_type=MessageType.TEXT, source=source)

    result = await runner._handle_message_with_agent(event, source, "key-2", 1)

    # An empty failure is NOT intentional silence — the notice must be returned.
    assert result
    assert "no response was generated" in result
