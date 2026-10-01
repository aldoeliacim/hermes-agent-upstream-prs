"""Tests for #46866: plain-text approval responses must resolve a blocking
dangerous-command approval instead of being steered/queued.

When the agent is blocked inside tools/approval.py waiting for a dangerous
command to be approved, a messaging user who replies "yes" / "approve" /
"deny" (without the leading slash) must have that response routed to the
approval handler.  Previously the bare-word reply fell through to the
steer/queue/interrupt logic in _handle_active_session_busy_message — the
approval never resolved, timed out, and auto-denied.

Slash forms (/approve, /deny) already bypass at the base-adapter guard;
this covers the bare-word forms Signal/SMS users naturally type.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id="m1",
    )


def _clear_approval_state():
    from tools import approval as mod
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


def _make_runner():
    """Minimal GatewayRunner that exercises the real busy-session handler."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    # Configure a home channel matching _make_source()'s chat_id so the owner
    # gate on the /approve resolve path recognizes this fake session as the
    # home/owner session, matching a real single-owner deployment. Without any
    # configured home channel, that gate conservatively (and correctly, by
    # design) treats EVERY source as a non-owner and fails closed — which would
    # deny every approval here, masking the plaintext-routing path under test.
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="test-token",
                home_channel=HomeChannel(
                    platform=Platform.TELEGRAM, chat_id="c1", name="test-home",
                ),
            )
        }
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._send_with_retry = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="reply1")
    )
    # _unwrap_ephemeral is a real base-adapter method; emulate its contract.
    adapter._unwrap_ephemeral = lambda r: (r, 0) if isinstance(r, str) else (None, 0)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.session_store = None
    runner._is_user_authorized = lambda _source: True
    # _handle_active_session_busy_message uses these only on the
    # non-approval fall-through path; harmless to stub.
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    return runner, adapter


def _register_blocking_approval(runner):
    """Register a real blocking approval entry for the runner's session."""
    from tools.approval import _gateway_queues
    from tools.approval_gateway_wait import _ApprovalEntry
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    entry = _ApprovalEntry({"command": "rm -rf /tmp/test"})
    _gateway_queues.setdefault(session_key, []).append(entry)
    return session_key, entry


@pytest.mark.parametrize("reply", ["yes", "approve", "ok", "y", "confirm"])
def test_plaintext_yes_resolves_approval(reply):
    _clear_approval_state()
    runner, adapter = _make_runner()
    session_key, entry = _register_blocking_approval(runner)

    handled = asyncio.run(
        runner._handle_active_session_busy_message(_make_event(reply), session_key)
    )

    assert handled is True
    assert entry.event.is_set()
    assert entry.result == "once"
    # The user gets a confirmation reply, not silence.
    adapter._send_with_retry.assert_awaited()
    _clear_approval_state()


def test_no_pending_approval_does_not_consume_conversational_yes():
    """A bare 'yes' with NO blocking approval must NOT be treated as an
    approval — it falls through to normal busy handling (design intent:
    'yes' in conversation must not execute a dangerous command)."""
    _clear_approval_state()
    runner, adapter = _make_runner()
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    # No approval registered.

    handled = asyncio.run(
        runner._handle_active_session_busy_message(_make_event("yes"), session_key)
    )

    # No approval existed, so nothing was resolved — the "yes" is treated
    # as ordinary text, not as a dangerous-command approval (design intent).
    # (It still flows through normal busy handling, which may send a busy
    # ack; the contract here is only that no approval was consumed.)
    from tools.approval import _gateway_queues
    assert session_key not in _gateway_queues
    _clear_approval_state()


# ── B2: non-owner /approve is fail-closed (denied, never executed) ───────────

def _make_non_owner_source() -> SessionSource:
    """A routed contact whose chat_id does NOT match the configured home."""
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="contact-99",
        chat_id="contact-99",
        user_name="contact",
        chat_type="dm",
    )


def _make_non_owner_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text, message_type=MessageType.TEXT,
        source=_make_non_owner_source(), message_id="mX",
    )


def test_non_owner_approve_is_denied_and_auto_denies_pending():
    """A /approve from a non-owner contact must NOT authorize the command.
    It fails closed: the pending approval is auto-DENIED and the caller is told
    there is nothing to approve — the dangerous command never runs."""
    _clear_approval_state()
    runner, _ = _make_runner()  # home_channel chat_id = "c1"

    # A blocking approval sits on the NON-OWNER's session key.
    from tools.approval import _gateway_queues, has_blocking_approval
    from tools.approval_gateway_wait import _ApprovalEntry

    contact_source = _make_non_owner_source()
    contact_key = runner._session_key_for_source(contact_source)
    entry = _ApprovalEntry({"command": "rm -rf /tmp/test"})
    _gateway_queues.setdefault(contact_key, []).append(entry)
    assert has_blocking_approval(contact_key)

    reply = asyncio.run(runner._handle_approve_command(_make_non_owner_event("approve")))

    # Fail closed: reported as "nothing to approve" and the pending entry denied.
    from agent.i18n import t
    assert reply == t("gateway.approve.no_pending")
    assert entry.event.is_set()
    assert entry.result == "deny"
    _clear_approval_state()


def test_owner_approve_still_resolves():
    """The owner (source matching the configured home) can still approve."""
    _clear_approval_state()
    runner, _ = _make_runner()  # home_channel chat_id = "c1" == _make_source() chat_id
    session_key, entry = _register_blocking_approval(runner)

    reply = asyncio.run(runner._handle_approve_command(_make_event("approve")))

    assert entry.event.is_set()
    assert entry.result == "once"
    assert reply is not None
    _clear_approval_state()
