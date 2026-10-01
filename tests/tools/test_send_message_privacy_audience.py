"""send_message(target="current") privacy screen is audience-scoped.

The deterministic privacy classes (internal endpoints, backend versions, mechanism
narration, cross-user request history) protect an UNENTITLED audience. The owner's own
private DM is the one audience entitled to detail about the owner's own infrastructure,
so a live-turn send there must not be refused; a group or a non-owner DM stays screened.
Mirrors the final-response gate in gateway/run_turn.py (see
test_owner_register_gate_e2e.py::test_owner_own_dm_is_never_content_filtered).
"""
from __future__ import annotations

import asyncio
import json

import pytest

import tools.approval as approval
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from tools.send_message_tool import _handle_send

OWNER_CHAT = "900000001"
CONTACT_CHAT = "900000002"
GROUP_CHAT = "-1000000000001"

BACKEND_DETAIL = (
    "El servidor está sano. El problema: tu app está probando `localhost:8096` y "
    "`:8920`, que son puertos internos. Jellyfin 10.11.11 responde bien."
)


class _Source:
    platform = Platform.TELEGRAM
    thread_id = None
    reply_gate_tool_sends = 0
    chat_id_alt = None
    user_id_alt = None

    def __init__(self, chat_id, chat_type):
        self.chat_id = chat_id
        self.user_id = chat_id
        self.chat_type = chat_type


def _send_current(monkeypatch, source, message):
    sent = []

    async def _fake_send_to_platform(platform, pconfig, chat_id, message, **kw):
        sent.append((chat_id, message))
        return {"success": True, "message_id": "m1"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", _fake_send_to_platform)
    monkeypatch.setattr(
        "model_tools._run_async",
        lambda coro: asyncio.get_event_loop().run_until_complete(coro),
    )
    cfg = GatewayConfig()
    cfg.platforms[Platform.TELEGRAM] = PlatformConfig(
        enabled=True, token="x",
        home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id=OWNER_CHAT, name="Home"),
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)
    token = approval.set_current_message_source(source)
    try:
        return json.loads(_handle_send({"target": "current", "message": message})), sent
    finally:
        approval.reset_current_message_source(token)


def test_owner_private_dm_receives_backend_detail(monkeypatch):
    r, sent = _send_current(monkeypatch, _Source(OWNER_CHAT, "dm"), BACKEND_DETAIL)
    assert "error" not in r, r
    assert sent and sent[0][1] == BACKEND_DETAIL


@pytest.mark.parametrize(
    "chat_id,chat_type",
    [(CONTACT_CHAT, "dm"), (GROUP_CHAT, "group"), (OWNER_CHAT, "group")],
    ids=["contact-dm", "group", "owner-id-but-shared-chat"],
)
def test_unentitled_audiences_are_still_screened(monkeypatch, chat_id, chat_type):
    r, sent = _send_current(monkeypatch, _Source(chat_id, chat_type), BACKEND_DETAIL)
    assert "error" in r and "withheld" in r["error"]
    assert sent == [], "a withheld message must never reach the adapter"
    assert "backend" not in r["error"].lower(), "the refusal must not name the matched class"


def test_owner_check_failure_screens(monkeypatch):
    """Fail-closed: if the owner check raises, the owner's DM is screened, not waved through."""
    import gateway.owner_authority as oa

    def _boom(*_a, **_k):
        raise RuntimeError("owner lookup unavailable")

    monkeypatch.setattr(oa, "source_is_owner", _boom)
    r, sent = _send_current(monkeypatch, _Source(OWNER_CHAT, "dm"), BACKEND_DETAIL)
    assert "error" in r and sent == []
