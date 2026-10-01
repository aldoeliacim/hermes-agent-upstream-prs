"""Tests for tools/whatsapp_action_tool.py.

Split (2026-08-11) mirrors the SAFE/ADMIN action split in the module under
test: whatsapp_action only accepts WHATSAPP_SAFE_ACTIONS, whatsapp_action_admin
only accepts WHATSAPP_ADMIN_ACTIONS and additionally enforces a call-time
owner-identity check when invoked from inside a live gateway turn.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.approval as approval
from gateway.config import Platform
from tools.whatsapp_action_tool import (
    _is_owner_whatsapp_source,
    _post_bridge,
    whatsapp_action_admin_tool,
    whatsapp_action_tool,
)


class _AsyncResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status = status
        self._payload = payload if payload is not None else {"success": True, "messageId": "msg1"}
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


def _run_async_immediately(coro):
    import asyncio
    return asyncio.run(coro)


def _config(port=18792):
    whatsapp_cfg = SimpleNamespace(enabled=True, extra={"bridge_port": port})
    return SimpleNamespace(platforms={Platform.WHATSAPP: whatsapp_cfg})


def _config_with_home(port=18792, home_chat_id="5215550000001@s.whatsapp.net"):
    """Config with a resolvable get_home_channel() for the owner-identity guard."""
    whatsapp_cfg = SimpleNamespace(enabled=True, extra={"bridge_port": port})
    home = SimpleNamespace(chat_id=home_chat_id) if home_chat_id else None

    def _get_home_channel(platform):
        return home if platform == Platform.WHATSAPP else None

    return SimpleNamespace(
        platforms={Platform.WHATSAPP: whatsapp_cfg},
        get_home_channel=_get_home_channel,
    )


class _WhatsAppSource:
    def __init__(self, chat_id, user_id=None):
        self.platform = Platform.WHATSAPP
        self.chat_id = chat_id
        self.user_id = user_id
        self.chat_id_alt = None
        self.user_id_alt = None


class _TelegramSource:
    platform = Platform.TELEGRAM
    chat_id = "12345"
    user_id = "12345"
    chat_id_alt = None
    user_id_alt = None


@pytest.mark.asyncio
async def test_post_bridge_uses_configured_whatsapp_bridge_port():
    response = _AsyncResponse(payload={"success": True, "messageId": "react1"})
    session = MagicMock()
    session.post = MagicMock(return_value=response)

    class SessionFactory:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *exc):
            return False

    with patch("aiohttp.ClientSession", return_value=SessionFactory()):
        result = await _post_bridge({"bridge_port": 18792}, "react", {"chatId": "chat", "messageId": "m1", "emoji": "💚"})

    assert result["success"] is True
    assert result["message_id"] == "react1"
    session.post.assert_called_once()
    assert session.post.call_args.args[0] == "http://127.0.0.1:18792/react"


# ---------------------------------------------------------------------------
# whatsapp_action (SAFE actions)
# ---------------------------------------------------------------------------

def test_status_reply_posts_to_status_reply_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True, "message_id": "msg1"})) as post_mock:
        result = json.loads(whatsapp_action_tool({
            "action": "status_reply",
            "status_message_id": "status1",
            "message": "Qué bonito ❤️",
            "status_author_jid": "5215551000005@s.whatsapp.net",
        }))

    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792},
        "status-reply",
        {
            "statusMessageId": "status1",
            "message": "Qué bonito ❤️",
            "statusAuthorJid": "5215551000005@s.whatsapp.net",
        },
        "POST",
    )


def test_dry_run_does_not_call_bridge():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock()) as post_mock:
        result = json.loads(whatsapp_action_tool({
            "action": "status_react",
            "status_message_id": "status1",
            "emoji": "💚",
            "dry_run": True,
        }))

    assert result["success"] is True
    assert result["dry_run"] is True
    assert result["endpoint"] == "status-react"
    post_mock.assert_not_called()


def test_star_message_filters_malformed_entries():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_tool({
            "action": "star_message",
            "chat_id": "c@s.whatsapp.net",
            "star_messages": [],
            "star": True,
        }))
    assert "star_messages" in result["error"]


def test_mute_chat_accepts_null_to_unmute():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({
            "action": "mute_chat",
            "chat_id": "c@s.whatsapp.net",
            "mute_ms": None,
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792}, "mute", {"chatId": "c@s.whatsapp.net", "muteMs": None}, "POST",
    )


def test_check_numbers_requires_non_empty_list():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_tool({"action": "check_numbers", "phone_numbers": []}))
    assert "phone_numbers" in result["error"]


def test_get_privacy_settings_uses_get_with_optional_force_query():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({"action": "get_privacy_settings", "privacy_force_refresh": True}))
    assert result["success"] is True
    post_mock.assert_awaited_once_with({"bridge_port": 18792}, "privacy?force=true", {}, "GET")


def test_send_presence_requires_valid_type():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_tool({"action": "send_presence", "presence_type": "typing"}))
    assert "presence_type" in result["error"]


def test_get_profile_picture_url_builds_endpoint_with_full_res_query():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({
            "action": "get_profile_picture_url", "lookup_jid": "c@s.whatsapp.net", "picture_full_res": True,
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792}, "profile-picture-url/c@s.whatsapp.net?type=image", {}, "GET",
    )


def test_get_business_profile_builds_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({"action": "get_business_profile", "lookup_jid": "c@s.whatsapp.net"}))
    assert result["success"] is True
    post_mock.assert_awaited_once_with({"bridge_port": 18792}, "business-profile/c@s.whatsapp.net", {}, "GET")


def test_group_invite_preview_uses_get():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({"action": "group_invite_preview", "invite_code": "AbCdEf123"}))
    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792}, "groups/invite-info/AbCdEf123", {}, "GET",
    )


def test_get_status_text_requires_non_empty_jids():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_tool({"action": "get_status_text", "lookup_jids": []}))
    assert "lookup_jids" in result["error"]


def test_get_status_text_builds_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({
            "action": "get_status_text", "lookup_jids": ["c@s.whatsapp.net"],
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792}, "status-lookup", {"jids": ["c@s.whatsapp.net"]}, "POST",
    )


def test_get_default_ephemeral_builds_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({
            "action": "get_default_ephemeral", "lookup_jids": ["c@s.whatsapp.net"],
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792}, "ephemeral-lookup", {"jids": ["c@s.whatsapp.net"]}, "POST",
    )


def test_presence_subscribe_requires_jid():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_tool({"action": "presence_subscribe"}))
    assert "presence_subscribe_jid" in result["error"]


def test_presence_subscribe_builds_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({
            "action": "presence_subscribe", "presence_subscribe_jid": "c@s.whatsapp.net",
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792}, "presence-subscribe", {"jid": "c@s.whatsapp.net"}, "POST",
    )


def test_group_join_requests_list_uses_get():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({"action": "group_join_requests_list", "chat_id": "g@g.us"}))
    assert result["success"] is True
    post_mock.assert_awaited_once_with({"bridge_port": 18792}, "groups/g@g.us/join-requests", {}, "GET")


def test_get_blocklist_uses_get():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({"action": "get_blocklist"}))
    assert result["success"] is True
    post_mock.assert_awaited_once_with({"bridge_port": 18792}, "blocklist", {}, "GET")


def test_get_own_profile_uses_get_with_no_params():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={
             "success": True, "raw_response": {"success": True, "id": "5215550000001@s.whatsapp.net", "lid": "90000000000001@lid", "name": "Hal"},
         })) as post_mock:
        result = json.loads(whatsapp_action_tool({"action": "get_own_profile"}))
    assert result["success"] is True
    assert result["raw_response"]["id"] == "5215550000001@s.whatsapp.net"
    post_mock.assert_awaited_once_with({"bridge_port": 18792}, "me", {}, "GET")


def test_get_own_profile_dry_run_reports_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_tool({"action": "get_own_profile", "dry_run": True}))
    assert result["success"] is True
    assert result["dry_run"] is True
    assert result["endpoint"] == "me"
    assert result["method"] == "GET"
    assert result["payload"] == {}


def test_list_groups_uses_get_with_no_params():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={
             "success": True, "raw_response": {"count": 1, "groups": [{"id": "120363012345@g.us", "subject": "Entrenamiento dev", "participants": 3}]},
         })) as post_mock:
        result = json.loads(whatsapp_action_tool({"action": "list_groups"}))
    assert result["success"] is True
    assert result["raw_response"]["groups"][0]["subject"] == "Entrenamiento dev"
    post_mock.assert_awaited_once_with({"bridge_port": 18792}, "groups", {}, "GET")


def test_list_groups_dry_run_reports_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_tool({"action": "list_groups", "dry_run": True}))
    assert result["success"] is True
    assert result["dry_run"] is True
    assert result["endpoint"] == "groups"
    assert result["method"] == "GET"
    assert result["payload"] == {}


def test_resolve_chat_requires_chat_id():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_tool({"action": "resolve_chat"}))
    assert "chat_id" in result["error"]


def test_resolve_chat_builds_endpoint_for_group():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={
             "success": True, "raw_response": {"name": "Entrenamiento dev", "isGroup": True, "participants": ["a@lid", "b@lid"]},
         })) as post_mock:
        result = json.loads(whatsapp_action_tool({
            "action": "resolve_chat", "chat_id": "120363012345@g.us",
        }))
    assert result["success"] is True
    assert result["raw_response"]["isGroup"] is True
    post_mock.assert_awaited_once_with({"bridge_port": 18792}, "chat/120363012345@g.us", {}, "GET")


def test_resolve_chat_builds_endpoint_for_dm():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({
            "action": "resolve_chat", "chat_id": "5215551000001@s.whatsapp.net",
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once_with({"bridge_port": 18792}, "chat/5215551000001@s.whatsapp.net", {}, "GET")


def test_resolve_chat_dry_run_reports_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_tool({
            "action": "resolve_chat", "chat_id": "120363012345@g.us", "dry_run": True,
        }))
    assert result["success"] is True
    assert result["dry_run"] is True
    assert result["endpoint"] == "chat/120363012345@g.us"
    assert result["method"] == "GET"


# --- whatsapp_action must refuse admin actions outright (belt-and-suspenders) ---

@pytest.mark.parametrize("admin_action", [
    "delete_message", "create_group", "group_leave",
    "block_contact", "update_contact", "group_participants_update",
])
def test_whatsapp_action_refuses_admin_actions(admin_action):
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_tool({"action": admin_action}))
    assert "error" in result
    assert "whatsapp_action_admin" in result["error"]


# ---------------------------------------------------------------------------
# whatsapp_action (SELF actions) — Hal's own profile/status/privacy, no
# owner gate even inside a live turn from a non-owner sender.
# ---------------------------------------------------------------------------

def test_post_text_status_requires_explicit_recipient_list():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_tool({
            "action": "post_text_status",
            "message": "hello status",
        }))
    assert "explicit non-empty status_jid_list" in result["error"]


def test_update_profile_name_builds_profile_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({"action": "update_profile_name", "name": "Hal"}))
    assert result["success"] is True
    post_mock.assert_awaited_once_with({"bridge_port": 18792}, "profile/name", {"name": "Hal"}, "POST")


def test_update_profile_status_builds_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({"action": "update_profile_status", "status_text": "Building things"}))
    assert result["success"] is True
    post_mock.assert_awaited_once_with({"bridge_port": 18792}, "profile/status", {"status": "Building things"}, "POST")


def test_update_privacy_requires_category_and_value():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_tool({"action": "update_privacy", "privacy_category": "last"}))
    assert "privacy_value" in result["error"]


def test_update_privacy_builds_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({
            "action": "update_privacy", "privacy_category": "last", "privacy_value": "contacts",
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792}, "privacy", {"category": "last", "value": "contacts"}, "POST",
    )


def test_update_link_previews_privacy_requires_boolean():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_tool({"action": "update_link_previews_privacy"}))
    assert "link_previews_disabled" in result["error"]


def test_update_link_previews_privacy_builds_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({
            "action": "update_link_previews_privacy", "link_previews_disabled": True,
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792}, "link-previews-privacy", {"disabled": True}, "POST",
    )


def test_update_default_ephemeral_builds_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_tool({"action": "update_default_ephemeral", "ephemeral_seconds": 86400}))
    assert result["success"] is True
    post_mock.assert_awaited_once_with({"bridge_port": 18792}, "default-ephemeral", {"seconds": 86400}, "POST")


def test_self_action_unrestricted_inside_live_turn_from_non_owner_sender():
    """The whole point of the SELF tier: Hal's own identity/account
    settings must NOT be gated by sender identity — that would defeat
    'Hal needs independence for what's his'. A non-owner sender's live
    turn still lets whatsapp_action update Hal's own profile, because
    whatsapp_action carries no owner check at all (unlike the admin tool)."""
    token = approval.set_current_message_source(
        _WhatsAppSource(chat_id="120363012345@g.us", user_id="5215500000000@lid")
    )
    try:
        config = _config_with_home(home_chat_id="5215550000001@s.whatsapp.net")
        with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=config), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
            result = json.loads(whatsapp_action_tool({"action": "update_profile_status", "status_text": "hi"}))
        assert result["success"] is True
        post_mock.assert_awaited_once()
    finally:
        approval.reset_current_message_source(token)


# --- whatsapp_action_admin must refuse SELF actions too (they moved tools) ---

@pytest.mark.parametrize("self_action", [
    "update_profile_name", "update_profile_status", "update_profile_picture",
    "remove_profile_picture", "post_text_status", "update_privacy",
    "update_default_ephemeral", "update_link_previews_privacy",
])
def test_whatsapp_action_admin_refuses_self_actions(self_action):
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_admin_tool({"action": self_action}))
    assert "error" in result
    assert "not an admin action" in result["error"]


# ---------------------------------------------------------------------------
# whatsapp_action_admin (ADMIN actions) — outside a live turn, unrestricted
# ---------------------------------------------------------------------------

def test_delete_message_requires_explicit_boolean_from_me():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_admin_tool({
            "action": "delete_message",
            "chat_id": "5215551000001@s.whatsapp.net",
            "message_id": "m1",
        }))
    assert "boolean from_me" in result["error"]


def test_delete_message_builds_delete_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_admin_tool({
            "action": "delete_message",
            "chat_id": "5215551000001@s.whatsapp.net",
            "message_id": "m1",
            "from_me": True,
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792},
        "delete",
        {"chatId": "5215551000001@s.whatsapp.net", "messageId": "m1", "fromMe": True},
        "POST",
    )


def test_group_invite_code_uses_get_method_with_reset_query():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_admin_tool({
            "action": "group_invite_code",
            "chat_id": "120363012345@g.us",
            "reset_invite": True,
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792}, "groups/120363012345@g.us/invite-code?reset=true", {}, "GET",
    )


def test_group_participants_update_requires_valid_action():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_admin_tool({
            "action": "group_participants_update",
            "chat_id": "120363012345@g.us",
            "group_participants": ["5215551000001@s.whatsapp.net"],
            "group_action": "nuke",
        }))
    assert "group_action" in result["error"]


def test_create_group_requires_subject_and_participants():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_admin_tool({"action": "create_group", "group_subject": "Test"}))
    assert "group_participants" in result["error"]


def test_group_ephemeral_update_rejects_invalid_duration():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_admin_tool({
            "action": "group_ephemeral_update",
            "chat_id": "g@g.us",
            "ephemeral_seconds": 3600,
        }))
    assert "ephemeral_seconds" in result["error"]


def test_group_ephemeral_update_builds_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_admin_tool({
            "action": "group_ephemeral_update", "chat_id": "g@g.us", "ephemeral_seconds": 86400,
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792}, "groups/g@g.us/ephemeral", {"seconds": 86400}, "POST",
    )


def test_block_contact_requires_valid_action():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_admin_tool({
            "action": "block_contact", "block_jid": "c@s.whatsapp.net", "block_action": "ban",
        }))
    assert "block_action" in result["error"]


def test_update_contact_requires_name_or_remove():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_admin_tool({"action": "update_contact", "contact_jid": "c@s.whatsapp.net"}))
    assert "contact_name" in result["error"] or "contact_remove" in result["error"]


def test_update_contact_remove_builds_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_admin_tool({
            "action": "update_contact", "contact_jid": "c@s.whatsapp.net", "contact_remove": True,
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792}, "contact", {"jid": "c@s.whatsapp.net", "remove": True}, "POST",
    )


def test_group_join_requests_update_requires_valid_action():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_admin_tool({
            "action": "group_join_requests_update",
            "chat_id": "g@g.us",
            "join_request_participants": ["p@s.whatsapp.net"],
            "join_request_action": "ignore",
        }))
    assert "join_request_action" in result["error"]


def test_group_join_by_invite_builds_endpoint():
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_admin_tool({"action": "group_join_by_invite", "invite_code": "AbCdEf123"}))
    assert result["success"] is True
    post_mock.assert_awaited_once_with(
        {"bridge_port": 18792}, "groups/accept-invite", {"code": "AbCdEf123"}, "POST",
    )


def test_whatsapp_action_admin_refuses_safe_actions():
    """The admin tool must not silently accept a safe action either — a
    caller confused about which tool to use gets a clear redirect, not a
    successful call through the wrong door."""
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()):
        result = json.loads(whatsapp_action_admin_tool({"action": "react_message"}))
    assert "error" in result
    assert "not an admin action" in result["error"]


# ---------------------------------------------------------------------------
# _is_owner_whatsapp_source — the identity resolver the live-turn guard uses
# ---------------------------------------------------------------------------

def test_is_owner_whatsapp_source_true_for_exact_home_channel_match():
    config = _config_with_home(home_chat_id="5215550000001@s.whatsapp.net")
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=config), \
         patch("gateway.whatsapp_identity.expand_whatsapp_aliases", side_effect=lambda v: {v}):
        source = _WhatsAppSource(chat_id="5215550000001@s.whatsapp.net")
        assert _is_owner_whatsapp_source(source) is True


def test_is_owner_whatsapp_source_false_for_different_whatsapp_sender():
    config = _config_with_home(home_chat_id="5215550000001@s.whatsapp.net")
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=config), \
         patch("gateway.whatsapp_identity.expand_whatsapp_aliases", side_effect=lambda v: {v}):
        source = _WhatsAppSource(chat_id="120363012345@g.us", user_id="5215500000000@lid")
        assert _is_owner_whatsapp_source(source) is False


def test_is_owner_whatsapp_source_true_via_lid_alias_expansion():
    """The owner speaking from inside a group is identified by their LID, not
    their phone JID — expand_whatsapp_aliases is what bridges the two, exactly
    like the existing USER.md-leak guard in gateway/run.py depends on."""
    config = _config_with_home(home_chat_id="5215550000001@s.whatsapp.net")

    def _fake_expand(value):
        # Simulate a bridge lid-mapping: the owner's phone JID and LID
        # resolve to the same alias set.
        aliases = {"5215550000001@s.whatsapp.net", "90000000000001@lid"}
        return aliases if value in aliases else {value}

    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=config), \
         patch("gateway.whatsapp_identity.expand_whatsapp_aliases", side_effect=_fake_expand):
        source = _WhatsAppSource(chat_id="120363012345@g.us", user_id="90000000000001@lid")
        assert _is_owner_whatsapp_source(source) is True


def test_is_owner_whatsapp_source_false_for_non_whatsapp_platform():
    config = _config_with_home(home_chat_id="5215550000001@s.whatsapp.net")
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=config):
        assert _is_owner_whatsapp_source(_TelegramSource()) is False


def test_is_owner_whatsapp_source_false_when_no_home_channel_configured():
    config = _config_with_home(home_chat_id=None)
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=config):
        source = _WhatsAppSource(chat_id="5215550000001@s.whatsapp.net")
        assert _is_owner_whatsapp_source(source) is False


def test_is_owner_whatsapp_source_false_for_none_source():
    assert _is_owner_whatsapp_source(None) is False


# ---------------------------------------------------------------------------
# whatsapp_action_admin_tool — the live-turn owner-only guard itself
# ---------------------------------------------------------------------------

@pytest.fixture
def live_turn_as_owner():
    token = approval.set_current_message_source(
        _WhatsAppSource(chat_id="5215550000001@s.whatsapp.net")
    )
    yield
    approval.reset_current_message_source(token)


@pytest.fixture
def live_turn_as_group_member():
    token = approval.set_current_message_source(
        _WhatsAppSource(chat_id="120363012345@g.us", user_id="5215500000000@lid")
    )
    yield
    approval.reset_current_message_source(token)


def test_admin_action_refused_inside_live_turn_from_non_owner_sender(live_turn_as_group_member):
    config = _config_with_home(home_chat_id="5215550000001@s.whatsapp.net")
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=config), \
         patch("gateway.whatsapp_identity.expand_whatsapp_aliases", side_effect=lambda v: {v}):
        result = json.loads(whatsapp_action_admin_tool({
            "action": "group_leave", "chat_id": "120363012345@g.us",
        }))
    assert "error" in result
    assert "refused" in result["error"]
    assert "owner" in result["error"]


def test_admin_action_allowed_inside_live_turn_from_owner_sender(live_turn_as_owner):
    config = _config_with_home(home_chat_id="5215550000001@s.whatsapp.net")
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=config), \
         patch("gateway.whatsapp_identity.expand_whatsapp_aliases", side_effect=lambda v: {v}), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_admin_tool({
            "action": "group_leave", "chat_id": "120363012345@g.us",
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once()


def test_admin_action_unrestricted_outside_a_live_turn():
    """CLI/cron/subagent callers (no live message source bound) are the
    sanctioned non-agent-turn path — mirrors send_message's precedent."""
    assert approval.get_current_message_source() is None
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
        result = json.loads(whatsapp_action_admin_tool({
            "action": "group_leave", "chat_id": "120363012345@g.us",
        }))
    assert result["success"] is True
    post_mock.assert_awaited_once()


def test_admin_action_refused_for_non_whatsapp_live_turn():
    """A live turn on ANOTHER platform (Telegram) must never be treated as
    'outside a live turn' just because the source isn't WhatsApp — that
    would let a Telegram turn silently drive WhatsApp admin actions."""
    token = approval.set_current_message_source(_TelegramSource())
    try:
        config = _config_with_home(home_chat_id="5215550000001@s.whatsapp.net")
        with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=config):
            result = json.loads(whatsapp_action_admin_tool({
                "action": "group_leave", "chat_id": "120363012345@g.us",
            }))
        assert "error" in result
        assert "refused" in result["error"]
    finally:
        approval.reset_current_message_source(token)


# ---------------------------------------------------------------------------
# WHATSAPP_DISCLOSURE_ACTIONS (list_groups, resolve_chat) — the privacy gate
# that stops Hal disclosing who else he talks to.
# ---------------------------------------------------------------------------

def test_list_groups_refused_inside_live_turn_from_non_owner_sender(live_turn_as_group_member):
    config = _config_with_home(home_chat_id="5215550000001@s.whatsapp.net")
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=config), \
         patch("gateway.whatsapp_identity.expand_whatsapp_aliases", side_effect=lambda v: {v}):
        result = json.loads(whatsapp_action_tool({"action": "list_groups"}))
    assert "error" in result
    assert "refused" in result["error"]
    assert "disclose" in result["error"]


def test_resolve_chat_refused_inside_live_turn_from_non_owner_sender(live_turn_as_group_member):
    config = _config_with_home(home_chat_id="5215550000001@s.whatsapp.net")
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=config), \
         patch("gateway.whatsapp_identity.expand_whatsapp_aliases", side_effect=lambda v: {v}):
        result = json.loads(whatsapp_action_tool({
            "action": "resolve_chat", "chat_id": "120363099999@g.us",
        }))
    assert "error" in result
    assert "refused" in result["error"]


def test_list_groups_allowed_inside_live_turn_from_owner_sender(live_turn_as_owner):
    config = _config_with_home(home_chat_id="5215550000001@s.whatsapp.net")
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=config), \
         patch("gateway.whatsapp_identity.expand_whatsapp_aliases", side_effect=lambda v: {v}), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"groups": []})) as post_mock:
        result = json.loads(whatsapp_action_tool({"action": "list_groups"}))
    assert "error" not in result
    post_mock.assert_awaited_once()


def test_list_groups_unrestricted_outside_a_live_turn():
    """CLI/cron/subagent callers (no live message source bound) are the
    sanctioned non-agent-turn path — mirrors the admin-tool precedent."""
    assert approval.get_current_message_source() is None
    with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=_config()), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"groups": []})) as post_mock:
        result = json.loads(whatsapp_action_tool({"action": "list_groups"}))
    assert "error" not in result
    post_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# Chat-targetable SAFE actions are scoped to the CURRENT chat on non-owner
# turns (2026-09-11). "Safe" = reversible and chat-scoped, never "any chat":
# a stranger's turn must not be able to archive/mute/react inside the owner's
# other conversations. Owner turns and out-of-turn callers are unrestricted.
# ---------------------------------------------------------------------------

def _scoped_call(action_args, *, source, home="5215550000001@s.whatsapp.net"):
    token = approval.set_current_message_source(source)
    try:
        config = _config_with_home(home_chat_id=home)
        with patch("tools.whatsapp_action_tool.load_gateway_config", return_value=config), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.whatsapp_action_tool._post_bridge", new=AsyncMock(return_value={"success": True})) as post_mock:
            result = json.loads(whatsapp_action_tool(action_args))
        return result, post_mock
    finally:
        approval.reset_current_message_source(token)


_STRANGER_GROUP = "120363012345@g.us"
_OTHER_CHAT = "120363099999@g.us"


@pytest.mark.parametrize("action,extra", [
    ("react_message", {"message_id": "m1", "emoji": "👍"}),
    ("star_message", {"star_messages": [{"id": "m1", "from_me": False}], "star": True}),
    ("archive_chat", {"archive": True}),
    ("mute_chat", {"mute_ms": 3600000}),
    ("pin_chat", {"pin": True}),
    ("mark_chat_read", {"read": True}),
])
def test_non_owner_turn_cannot_target_another_chat(action, extra):
    """NEGATIVE: from a stranger's group, a different chat_id is refused
    BEFORE any bridge I/O."""
    src = _WhatsAppSource(chat_id=_STRANGER_GROUP, user_id="5215500000000@lid")
    result, post_mock = _scoped_call({"action": action, "chat_id": _OTHER_CHAT, **extra}, source=src)
    assert "error" in result and "current chat" in result["error"]
    post_mock.assert_not_awaited()


@pytest.mark.parametrize("action,extra", [
    ("react_message", {"message_id": "m1", "emoji": "👍"}),
    ("archive_chat", {"archive": True}),
])
def test_non_owner_turn_may_target_its_own_chat(action, extra):
    """POSITIVE: the same actions on the chat the turn is in still work."""
    src = _WhatsAppSource(chat_id=_STRANGER_GROUP, user_id="5215500000000@lid")
    result, post_mock = _scoped_call({"action": action, "chat_id": _STRANGER_GROUP, **extra}, source=src)
    assert result.get("success") is True
    post_mock.assert_awaited_once()


def test_owner_turn_may_target_any_chat():
    """The owner's own turn is unrestricted — this gate is about strangers."""
    owner = "5215550000001@s.whatsapp.net"
    src = _WhatsAppSource(chat_id=owner, user_id=owner)
    result, post_mock = _scoped_call(
        {"action": "archive_chat", "chat_id": _OTHER_CHAT, "archive": True}, source=src, home=owner
    )
    assert result.get("success") is True
    post_mock.assert_awaited_once()


def test_send_presence_scope_uses_presence_chat_id():
    src = _WhatsAppSource(chat_id=_STRANGER_GROUP, user_id="5215500000000@lid")
    result, post_mock = _scoped_call(
        {"action": "send_presence", "presence_type": "composing", "presence_chat_id": _OTHER_CHAT}, source=src
    )
    assert "error" in result and "current chat" in result["error"]
    post_mock.assert_not_awaited()


def test_same_chat_helper_treats_dm_phone_and_lid_as_one_chat():
    from tools.whatsapp_action_tool import _same_whatsapp_chat
    with patch("gateway.whatsapp_identity.expand_whatsapp_aliases",
               side_effect=lambda x: {"5215550000001@s.whatsapp.net", "140000000000009@lid"}):
        assert _same_whatsapp_chat("5215550000001@s.whatsapp.net", "140000000000009@lid")
    assert _same_whatsapp_chat("120363012345@g.us", "120363012345@g.us")
    assert not _same_whatsapp_chat("120363012345@g.us", "120363099999@g.us")
    assert not _same_whatsapp_chat("", "120363012345@g.us")
