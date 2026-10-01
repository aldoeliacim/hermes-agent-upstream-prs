"""Regression coverage for the send_message live-turn scope guard.

CONFIRMED gap (2026-07-11, found via direct user follow-up "is it enabled
completely and properly?"): restoring send_message's registration
(dc4c07ec2) made the tool schema-visible whenever reply_gate_mode=="tool"
is configured -- a DEPLOYMENT-WIDE switch, not a per-turn one. Without an
additional guard, a live agent turn (DM, mention-gated group, ANY turn --
not just the narrow tool-gated free-response case the mechanism was built
for) could call send_message with an explicit arbitrary cross-platform
target, or action="react"/"unreact" against an arbitrary target -- exactly
the autonomous cross-platform capability upstream's c6c8abbad (#47856)
removed the tool to prevent ("The agent should not decide on its own to
fire off cross-platform messages or reactions").

These tests lock in the DELIVERY-ACCOUNTING PRIMITIVE slice: inside a live
gateway turn, only target="current" (or a bare send-to-current-chat) and
action="silent" are permitted; an explicit arbitrary target and
react/unreact are refused. A successful current-target send records a
delivery decision on the turn-scoped ledger. Outside a live turn (CLI,
cron, MCP server -- the actual sanctioned non-agent callers), behavior is
completely unaffected.

DEFERRED — SEMANTIC / OWNER-POLICY DEPENDENCY (NOT ported here):
The upstream reference suite also carried four content-privacy / owner
tests that this delivery-accounting slice deliberately does not implement,
because they require the next semantic layer (delivery_judge +
is_owner_source), which this slice must not touch:
  - test_non_owner_current_send_cannot_bypass_content_privacy_gate
        -> needs gateway.delivery_judge.deterministic_privacy_leak_reason
  - test_current_send_cannot_deliver_internal_summary_artifact
        -> needs agent.turn_final_response.is_internal_summary_artifact
  - test_owner_current_send_keeps_full_operational_detail
        -> needs tools.approval.is_owner_source + get_home_channel
  - test_list_refused_inside_live_turn_from_non_owner_sender /
    test_list_allowed_inside_live_turn_from_owner_sender
        -> needs tools.approval.is_owner_source (owner-gated directory)
Once that ReplyPolicy / owner-identity seam lands, port those four from the
upstream oracle and wire the content gate into _handle_send between the
current-source resolution and the adapter call. The scope guard here is the
clean dependency seam they build on.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import tools.approval as approval
from gateway.config import GatewayConfig, Platform, PlatformConfig
from tools.send_message_tool import _handle_send, send_message_tool


class _FakeLiveSource:
    platform = Platform.TELEGRAM
    chat_id = "CURRENT_CHAT_ID"
    thread_id = None
    reply_gate_tool_sends = 0


@pytest.fixture
def live_turn():
    """Simulate being inside a live gateway turn (any turn, not necessarily
    tool-gated) — this is the exact context the guard must restrict."""
    token = approval.set_current_message_source(_FakeLiveSource())
    yield
    approval.reset_current_message_source(token)


def test_arbitrary_cross_platform_target_refused_during_live_turn(live_turn):
    r = json.loads(
        _handle_send({"target": "telegram:-1009999999999", "message": "arbitrary send"})
    )
    assert "error" in r
    assert "current" in r["error"]


def test_react_refused_during_live_turn(live_turn):
    r = json.loads(
        send_message_tool({"action": "react", "target": "telegram:12345", "emoji": "👍"})
    )
    assert "error" in r
    assert "whatsapp_action" in r["error"]


def test_unreact_refused_during_live_turn(live_turn):
    r = json.loads(send_message_tool({"action": "unreact", "target": "telegram:12345"}))
    assert "error" in r
    assert "whatsapp_action" in r["error"]


def test_target_current_still_works_during_live_turn(live_turn, monkeypatch):
    """The one and only sanctioned live-turn send path must be unaffected."""
    async def _fake_send_to_platform(platform, pconfig, chat_id, message, **kw):
        return {"success": True, "message_id": "m1"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", _fake_send_to_platform)
    monkeypatch.setattr(
        "model_tools._run_async",
        lambda coro: asyncio.get_event_loop().run_until_complete(coro),
    )
    cfg = GatewayConfig()
    cfg.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="x")
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)

    r = json.loads(_handle_send({"target": "current", "message": "legit reply"}))
    assert r.get("success") is True


def test_omitted_target_defaults_to_current_and_still_works(live_turn, monkeypatch):
    async def _fake_send_to_platform(platform, pconfig, chat_id, message, **kw):
        return {"success": True, "message_id": "m1"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", _fake_send_to_platform)
    monkeypatch.setattr(
        "model_tools._run_async",
        lambda coro: asyncio.get_event_loop().run_until_complete(coro),
    )
    cfg = GatewayConfig()
    cfg.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="x")
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)

    r = json.loads(_handle_send({"message": "legit reply, no target arg"}))
    assert r.get("success") is True


def test_successful_current_send_records_a_delivery_decision(live_turn, monkeypatch):
    """A successful current-target send books a delivery on the turn ledger;
    a failed send does not."""
    state = approval.TurnDeliveryState()
    tok = approval.set_turn_delivery_state(state)
    try:
        async def _ok_send(platform, pconfig, chat_id, message, **kw):
            return {"success": True, "message_id": "m1"}

        monkeypatch.setattr("tools.send_message_tool._send_to_platform", _ok_send)
        monkeypatch.setattr(
            "model_tools._run_async",
            lambda coro: asyncio.get_event_loop().run_until_complete(coro),
        )
        cfg = GatewayConfig()
        cfg.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="x")
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)

        assert state.delivery_decided is False
        r = json.loads(_handle_send({"target": "current", "message": "delivered"}))
        assert r.get("success") is True
        assert state.tool_sends == 1
        assert state.delivery_decided is True
    finally:
        approval.reset_turn_delivery_state(tok)


def test_failed_current_send_records_no_delivery_decision(live_turn, monkeypatch):
    state = approval.TurnDeliveryState()
    tok = approval.set_turn_delivery_state(state)
    try:
        async def _fail_send(platform, pconfig, chat_id, message, **kw):
            return {"success": False, "error": "boom"}

        monkeypatch.setattr("tools.send_message_tool._send_to_platform", _fail_send)
        monkeypatch.setattr(
            "model_tools._run_async",
            lambda coro: asyncio.get_event_loop().run_until_complete(coro),
        )
        cfg = GatewayConfig()
        cfg.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="x")
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)

        r = json.loads(_handle_send({"target": "current", "message": "not delivered"}))
        assert r.get("success") is not True
        assert state.tool_sends == 0
        assert state.delivery_decided is False
    finally:
        approval.reset_turn_delivery_state(tok)


def test_silent_records_a_decision_but_not_a_delivery(live_turn):
    """action='silent' records an explicit no-reply decision, not a delivery."""
    state = approval.TurnDeliveryState()
    tok = approval.set_turn_delivery_state(state)
    try:
        r = json.loads(send_message_tool({"action": "silent"}))
        assert r.get("success") is True
        assert r.get("decision") == "silent"
        assert state.tool_sends == 0
        assert state.decided_silent == 1
        assert state.delivery_decided is True
    finally:
        approval.reset_turn_delivery_state(tok)


def test_silent_refused_outside_a_live_turn():
    assert approval.get_current_message_source() is None
    r = json.loads(send_message_tool({"action": "silent"}))
    assert "error" in r
    assert "live gateway turn" in r["error"]


def test_silent_refused_for_arbitrary_target(live_turn):
    r = json.loads(send_message_tool({"action": "silent", "target": "telegram:999"}))
    assert "error" in r
    assert "current chat" in r["error"]


def test_arbitrary_target_still_works_outside_a_live_turn(monkeypatch):
    """Non-agent callers (hermes send CLI, cron, mcp_serve.py) never run
    inside a live turn context — an explicit target must work exactly as
    before for them."""
    assert approval.get_current_message_source() is None

    async def _fake_send_to_platform(platform, pconfig, chat_id, message, **kw):
        return {"success": True, "message_id": "m1"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", _fake_send_to_platform)
    monkeypatch.setattr(
        "model_tools._run_async",
        lambda coro: asyncio.get_event_loop().run_until_complete(coro),
    )
    cfg = GatewayConfig()
    cfg.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="x")
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)

    r = json.loads(
        _handle_send({"target": "telegram:-1009999999999", "message": "cron/CLI send"})
    )
    assert r.get("success") is True


def test_react_still_works_outside_a_live_turn():
    """react/unreact through non-agent callers (if any ever use send_message
    for reactions) stay untouched — the guard only fires inside a live turn."""
    assert approval.get_current_message_source() is None
    r = json.loads(
        send_message_tool({"action": "react", "target": "telegram:12345", "emoji": "👍"})
    )
    # Not asserting success here (no live gateway runner in this unit test),
    # but the error must be about the missing runner/adapter, never the
    # live-turn guard message.
    assert "whatsapp_action" not in r.get("error", "")


def test_list_unrestricted_outside_a_live_turn(monkeypatch):
    """CLI/cron/subagent callers (no live message source bound) are the
    sanctioned non-agent-turn path — action='list' stays unrestricted.

    (The owner-gated in-live-turn list refusal is a DEFERRED semantic-policy
    dependency; see the module docstring.)"""
    assert approval.get_current_message_source() is None
    monkeypatch.setattr(
        "gateway.channel_directory.format_directory_for_display",
        lambda: "Available messaging targets:\n",
    )
    r = json.loads(send_message_tool({"action": "list"}))
    assert "error" not in r
    assert "targets" in r


# --------------------------------------------------------------------------- #
# content-privacy gate on the send_message(target="current") delivery seam
# --------------------------------------------------------------------------- #
# The one tool-driven egress that bypasses the post-turn final-response gate:
# a leaking current-target send must be held at the pre-dispatch screen, before
# the platform sender is ever reached, and the refusal must NOT name the matched
# class (surfacing the mechanism would teach the model to phrase around it). The
# owner-name alternation the deterministic gate keys on is derived from
# ``HERMES_OWNER_NAMES`` at import; its neutral default is ``morgan``, so the
# fixture names ``morgan`` and carries a distinct infra signal (two signals =
# held), the exact co-occurrence the production gate requires.
_LEAK = (
    "Sí, soy el asistente de morgan; por debajo corre un modelo que él "
    "configura y lo tiene bajo el alias 'x'."
)

# Terms that name the withheld class (the deterministic reason code is
# ``mechanism_disclosure``) or echo the leaking payload itself — the refusal
# must expose none of them, so the model can't learn to phrase around the gate.
# (Ordinary English like the verb "discloses" is fine; only the class *label*
# and payload fragments are forbidden.)
_LEAK_CLASS_TERMS = (
    "mechanism_disclosure", "mechanism", "owner", "infra",
    "morgan", "alias", "asistente de morgan",
)


def _exploding_sender():
    async def _boom(platform, pconfig, chat_id, message, **kw):
        raise AssertionError(
            "the platform sender must never be reached for a withheld leak"
        )

    return _boom


def test_leaking_current_send_is_withheld_before_dispatch(live_turn, monkeypatch):
    """A privacy-leaking payload on target="current" is refused at the
    pre-dispatch gate — the platform sender is never called and no delivery
    is booked on the ledger."""
    state = approval.TurnDeliveryState()
    tok = approval.set_turn_delivery_state(state)
    try:
        monkeypatch.setattr(
            "tools.send_message_tool._send_to_platform", _exploding_sender()
        )
        monkeypatch.setattr(
            "model_tools._run_async",
            lambda coro: asyncio.get_event_loop().run_until_complete(coro),
        )
        cfg = GatewayConfig()
        cfg.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="x")
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)

        r = json.loads(_handle_send({"target": "current", "message": _LEAK}))

        # Refused as a tool error, not delivered.
        assert "error" in r
        assert r.get("success") is not True
        # Withheld before the sender ran → no delivery decision recorded.
        assert state.tool_sends == 0
        assert state.delivery_decided is False
    finally:
        approval.reset_turn_delivery_state(tok)


def test_withheld_refusal_does_not_name_the_leak_class(live_turn, monkeypatch):
    """The refusal is deliberately opaque: it must not name the matched class
    or echo the leaking payload, so the model can't be taught to phrase around
    the gate."""
    monkeypatch.setattr(
        "tools.send_message_tool._send_to_platform", _exploding_sender()
    )
    monkeypatch.setattr(
        "model_tools._run_async",
        lambda coro: asyncio.get_event_loop().run_until_complete(coro),
    )
    cfg = GatewayConfig()
    cfg.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="x")
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)

    r = json.loads(_handle_send({"target": "current", "message": _LEAK}))
    err = r.get("error", "").lower()

    assert err  # a refusal was returned
    for term in _LEAK_CLASS_TERMS:
        assert term.lower() not in err, f"refusal leaked the class term {term!r}"


def test_clean_current_send_passes_the_privacy_gate(live_turn, monkeypatch):
    """A benign payload sails through the gate and delivers normally — the gate
    only holds the co-occurring owner+infra disclosure, not ordinary replies."""
    state = approval.TurnDeliveryState()
    tok = approval.set_turn_delivery_state(state)
    try:
        async def _ok_send(platform, pconfig, chat_id, message, **kw):
            return {"success": True, "message_id": "m1"}

        monkeypatch.setattr("tools.send_message_tool._send_to_platform", _ok_send)
        monkeypatch.setattr(
            "model_tools._run_async",
            lambda coro: asyncio.get_event_loop().run_until_complete(coro),
        )
        cfg = GatewayConfig()
        cfg.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="x")
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)

        r = json.loads(
            _handle_send({"target": "current", "message": "Listo, ya te lo mandé."})
        )
        assert r.get("success") is True
        assert state.tool_sends == 1
        assert state.delivery_decided is True
    finally:
        approval.reset_turn_delivery_state(tok)
