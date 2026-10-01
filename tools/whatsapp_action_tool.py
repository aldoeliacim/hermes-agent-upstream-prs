"""WhatsApp-specific messaging actions.

This tool intentionally keeps WhatsApp-only semantics out of the generic
``send_message`` tool. It exposes bridge primitives that require WhatsApp
message/status identifiers, own-account profile control, chat state, or
group administration.

THREE-TIER split by blast radius (2026-08-11, closing a gap found during
the capability audit that first shipped all 42 actions under one unguarded
tool, then adding a dedicated self-account tier the same day):

``whatsapp_action`` exposes the SAFE + SELF subsets — always available, no
restriction:

- SAFE: reversible, chat-scoped, or read-only primitives (react, reply to
  a status, pin/mute/archive/mark-read a chat, star a message, look up
  privacy/blocklist/profile-picture/business-profile/status-text, list
  pending group-join requests, preview an invite without joining,
  broadcast presence).
- SELF: Hal's own identity and own-account settings — nothing here touches
  another person's data, another chat's shared state, or the owner's account
  (update own profile name/status-text/picture, post own text status to
  an explicit recipient list, change own privacy settings, own default
  disappearing-messages duration, own link-preview generation toggle).
  These are Hal expressing/configuring HIMSELF, not acting on the owner's
  behalf or on anyone else's data — no owner gate applies. Explicit
  decision (2026-08-11): "Hal needs independence for what's his, like
  profile picture, status, etc" — profile identity + status + own-account
  settings, not just appearance.

``whatsapp_action_admin`` exposes the ADMIN subset — irreversible,
account-wide-but-SHARED, or other-people-affecting primitives (delete a
message, create/leave/administer a group you're in, block/unblock a
contact, mutate the saved-contact book). These affect state OTHER people
see or depend on (a group's membership/settings, a contact's own block
status, a message another person can still see), so they are gated by a
RUNTIME identity check, not a schema description: ``_is_owner_whatsapp_source``
resolves the current live turn's inbound message source through the exact
same WhatsApp phone/LID alias expansion
(``gateway.whatsapp_identity.expand_whatsapp_aliases``) the codebase
already trusts to decide whether a WhatsApp sender IS the owner (see
``gateway/run.py::_should_skip_user_profile_for_source``, the guard that
keeps USER.md out of non-owner chats). Outside a live gateway turn (CLI,
cron, subagent — contexts the owner already controls directly) the admin
tool is unrestricted, mirroring ``send_message``'s
``_guarded_live_turn_call`` precedent. Inside a live turn, admin actions
execute ONLY when the turn's own sender resolves to the configured
WhatsApp home-channel identity; every other sender (group members,
non-owner DMs, anyone else) gets a clear refusal. This is deliberately a
call-time check inside the handler, not a ``check_fn`` gate on tool
registration — ``check_fn`` results are cached up to 30s and are NOT
turn-scoped (see ``tools/registry.py::_check_fn_cached``), so gating
identity there would let one turn's verdict leak into a different sender's
turn within the same cache window. A call-time check reads
``tools.approval.get_current_message_source()`` fresh on every invocation
and can never leak across turns.

The SELF/ADMIN line is drawn on "does this action change something another
party (a contact, a group's other members, the owner's account) can observe or
depend on" — not on "is it reversible" (a group rename IS reversible but
still admin, because other group members see it happen) and not on "is it
account-wide" (own-account privacy settings ARE account-wide but still
self, because only Hal's own experience/visibility changes).

``dry_run=true`` on any of the three surfaces lets the caller inspect the
exact bridge endpoint/payload before performing any external side effect.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from agent.redact import redact_sensitive_text
from gateway.config import Platform, load_gateway_config
from tools.registry import registry, tool_error


# Read-only, but disclose WHO Hal talks to (every group he's in, or a named
# chat's display name / a group's member roster) rather than mutating
# anything. That makes them a privacy leak, not a blast-radius risk: a
# non-owner group member or non-owner DM contact could otherwise ask Hal
# "what groups are you in" or "who's in <other chat>" and get back the owner's
# full contact/group graph across every session. Gated by
# _is_owner_whatsapp_source() at call time inside whatsapp_action_tool —
# same call-time-not-check_fn reasoning as WHATSAPP_ADMIN_ACTIONS (see the
# module docstring), because this is a per-turn-sender identity fact, not a
# static registration-time property.
WHATSAPP_DISCLOSURE_ACTIONS = frozenset({
    "list_groups",
    "resolve_chat",
})

# Actions with no meaningful blast radius: reversible, scoped to a single
# chat/message, or read-only. Always available via ``whatsapp_action``.
WHATSAPP_SAFE_ACTIONS = frozenset({
    "react_message",
    "status_reply",
    "status_react",
    "star_message",
    "archive_chat",
    "mute_chat",
    "pin_chat",
    "mark_chat_read",
    "check_numbers",
    "group_invite_preview",
    "group_join_requests_list",
    "get_privacy_settings",
    "get_blocklist",
    "get_profile_picture_url",
    "get_business_profile",
    "get_status_text",
    "get_default_ephemeral",
    "get_own_profile",
    "send_presence",
    "presence_subscribe",
    "list_groups",
    "resolve_chat",
})

# Hal's own identity and own-account settings. Nothing here touches another
# person's data, a shared group's state, or the owner's account — only Hal's own
# self-presentation and how HIS account behaves. No owner gate: this is Hal
# being himself, not Hal acting on the owner's behalf. Always available via
# ``whatsapp_action`` alongside WHATSAPP_SAFE_ACTIONS.
WHATSAPP_SELF_ACTIONS = frozenset({
    "update_profile_name",
    "update_profile_status",
    "update_profile_picture",
    "remove_profile_picture",
    "post_text_status",
    "update_privacy",
    "update_default_ephemeral",
    "update_link_previews_privacy",
})

# Irreversible, or affecting state OTHER people see or depend on (a shared
# group's membership/settings, a contact's block status, a message another
# party can still see). Gated by _is_owner_whatsapp_source() at call time
# inside whatsapp_action_admin_tool.
WHATSAPP_ADMIN_ACTIONS = frozenset({
    "delete_message",
    "create_group",
    "group_participants_update",
    "group_update_subject",
    "group_update_description",
    "group_leave",
    "group_invite_code",
    "group_setting_update",
    "group_ephemeral_update",
    "group_member_add_mode_update",
    "group_join_approval_mode_update",
    "group_join_requests_update",
    "group_join_by_invite",
    "block_contact",
    "update_contact",
})


# Safe actions that take a chat target. Subject to the current-chat scope gate
# in whatsapp_action_tool() for non-owner live turns. (delete_message and the
# group_* actions are ADMIN and already owner-gated wholesale.)
WHATSAPP_CHAT_TARGETABLE_SAFE_ACTIONS = frozenset({
    "react_message",
    "star_message",
    "archive_chat",
    "mute_chat",
    "pin_chat",
    "mark_chat_read",
    "send_presence",
})


def _same_whatsapp_chat(a: str, b: str) -> bool:
    """True when two chat identifiers denote the same WhatsApp chat.

    Groups compare by exact JID. DMs may surface as a phone JID
    (``5215...@s.whatsapp.net``) in one place and a LID (``1423...@lid``) in
    another for the same person, so DMs compare through the shared alias
    expansion the owner-identity guard uses.
    """
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return False
    if a == b:
        return True
    if a.endswith("@g.us") or b.endswith("@g.us"):
        return False
    try:
        from gateway.whatsapp_identity import expand_whatsapp_aliases
        return bool((expand_whatsapp_aliases(a) | {a}) & (expand_whatsapp_aliases(b) | {b}))
    except Exception:
        return False


def _is_owner_whatsapp_source(source: Any) -> bool:
    """True when ``source`` (a live turn's inbound message source) resolves,
    via the same WhatsApp phone/LID alias expansion the owner-profile-leak
    guard uses, to this deployment's configured WhatsApp home channel — i.e.
    the message came from the owner's own WhatsApp identity. False for any
    other WhatsApp sender (group members, non-owner DMs) and for any
    non-WhatsApp or unresolvable source.

    Deliberately mirrors ``gateway/run.py::_should_skip_user_profile_for_source``'s
    WhatsApp branch rather than importing it directly: that method lives on
    ``GatewayRunner`` and needs a live runner instance plus a resolved
    ``user_config``; this helper only needs the module-level home-channel
    config and the shared alias-expansion primitive, so it can run from a
    plain tool handler with no runner reference.
    """
    try:
        from gateway.whatsapp_identity import expand_whatsapp_aliases
    except Exception:
        return False
    if source is None:
        return False
    if getattr(source, "platform", None) != Platform.WHATSAPP:
        return False

    try:
        config = load_gateway_config()
        home = config.get_home_channel(Platform.WHATSAPP) if config else None
    except Exception:
        home = None
    if not home or not home.chat_id:
        return False

    home_ids = {str(home.chat_id)}
    for value in list(home_ids):
        home_ids |= expand_whatsapp_aliases(value)

    source_values = {
        str(getattr(source, "chat_id", "") or ""),
        str(getattr(source, "user_id", "") or ""),
        str(getattr(source, "chat_id_alt", "") or ""),
        str(getattr(source, "user_id_alt", "") or ""),
    }
    source_values.discard("")
    for value in list(source_values):
        source_values |= expand_whatsapp_aliases(value)

    return bool(home_ids & source_values)



WHATSAPP_ACTION_SCHEMA = {
    "name": "whatsapp_action",
    "description": (
        "Perform WhatsApp actions through the local WhatsApp bridge that are "
        "either SAFE (reversible/chat-scoped/read-only) or SELF (Hal's own "
        "identity/account settings, touching nothing another person can see or "
        "depend on). SAFE: react to a cached message, privately reply to a "
        "cached status, react to a cached status, star a message, archive/mute/"
        "pin/mark-read a chat, check whether phone numbers are on WhatsApp, "
        "preview a group invite without joining, list pending group join "
        "requests, inspect privacy settings and the blocklist, look up a "
        "profile picture URL or business profile, look up your OWN account "
        "identity (own JID/LID/display name — the response's lookupId/lookupLid "
        "fields are pre-normalized for chaining: pass either straight into "
        "lookup_jid/lookup_jids on get_status_text/get_profile_picture_url/"
        "get_default_ephemeral to introspect your own status text/picture/"
        "ephemeral setting; the raw id/lid fields carry a device suffix that "
        "breaks those lookups), look up a contact's public "
        "status text or default disappearing-messages duration, broadcast "
        "presence (available/typing/recording), and subscribe to a contact's "
        "presence updates. SELF: "
        "update your own profile (name/about text/picture), post your own text "
        "status to an explicit recipient list, change your own privacy settings "
        "(last-seen/online/profile-picture/status/read-receipts/group-add/call/"
        "message visibility), set your own default disappearing-messages "
        "duration, and toggle your own link-preview generation. Also SAFE: "
        "list_groups (every group this account is a member of, with id/subject/"
        "participant count — the authoritative way to resolve a group JID, never "
        "guess it from message-log volume) and resolve_chat (look up a single "
        "known chat_id — group or DM — and get back its display name/subject; "
        "for a group also its member JIDs). Both are always "
        "available with no additional restriction — this is Hal acting on his "
        "own account/self-presentation, not on the owner's behalf. For irreversible "
        "or other-people-affecting actions (delete a message, create/leave/"
        "administer a group, block a contact, edit the saved-contact book), use "
        "whatsapp_action_admin instead. Set dry_run=true to inspect the exact "
        "bridge endpoint/payload without sending."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(WHATSAPP_SAFE_ACTIONS | WHATSAPP_SELF_ACTIONS),
                "description": "WhatsApp action to perform.",
            },
            "chat_id": {
                "type": "string",
                "description": "WhatsApp chat JID, e.g. 5215551000001@s.whatsapp.net or 120...@g.us. Used by react_message, delete_message, star_message, archive_chat, mute_chat, pin_chat, mark_chat_read, resolve_chat.",
            },
            "message_id": {
                "type": "string",
                "description": "Cached WhatsApp message id for react_message or delete_message.",
            },
            "from_me": {
                "type": "boolean",
                "description": "Required for delete_message: whether the message being deleted was sent by this account.",
            },
            "participant": {
                "type": "string",
                "description": "Optional group participant JID for delete_message (admin-deleting someone else's message in a group).",
            },
            "star_messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "from_me": {"type": "boolean"},
                    },
                },
                "description": "For star_message: list of {id, from_me} identifying which cached messages to star/unstar.",
            },
            "star": {
                "type": "boolean",
                "description": "For star_message: true to star, false to unstar.",
            },
            "archive": {
                "type": "boolean",
                "description": "For archive_chat: true to archive, false to unarchive.",
            },
            "mute_ms": {
                "type": ["integer", "null"],
                "description": "For mute_chat: mute duration in milliseconds (e.g. 28800000 for 8h, 604800000 for 7d), or null to unmute.",
            },
            "pin": {
                "type": "boolean",
                "description": "For pin_chat: true to pin, false to unpin.",
            },
            "read": {
                "type": "boolean",
                "description": "For mark_chat_read: true to mark read, false to mark unread.",
            },
            "phone_numbers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "For check_numbers: E.164 phone numbers (or bare digits) to check for a WhatsApp account. LIDs are not supported.",
            },
            "status_message_id": {
                "type": "string",
                "description": "Cached WhatsApp status/story message id for status_reply or status_react.",
            },
            "status_author_jid": {
                "type": "string",
                "description": "Optional author JID for a status/story when the cached message lacks participant metadata.",
            },
            "message": {
                "type": "string",
                "description": "Text for status_reply or post_text_status.",
            },
            "emoji": {
                "type": "string",
                "description": "Emoji reaction for react_message or status_react. Empty string removes a reaction.",
            },
            "status_jid_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit WhatsApp recipient JIDs for post_text_status. Required and non-empty; there is no all-contacts default.",
            },
            "background_color": {
                "type": "string",
                "description": "Optional WhatsApp text status background color, e.g. #315575.",
            },
            "font": {
                "type": "integer",
                "description": "Optional WhatsApp text status font id.",
            },
            "name": {
                "type": "string",
                "description": "For update_profile_name: the new display name.",
            },
            "status_text": {
                "type": "string",
                "description": "For update_profile_status: the new about/status text (NOT a story).",
            },
            "file_path": {
                "type": "string",
                "description": "For update_profile_picture: local path to the image file to upload.",
            },
            "profile_jid": {
                "type": "string",
                "description": "For update_profile_picture/remove_profile_picture: target a group's picture instead of the own account (omit for own account).",
            },
            "group_subject": {
                "type": "string",
                "description": "For create_group (initial name) or group_update_subject (rename).",
            },
            "group_participants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "For create_group (initial members) or group_participants_update (JIDs to add/remove/promote/demote).",
            },
            "group_action": {
                "type": "string",
                "enum": ["add", "remove", "promote", "demote"],
                "description": "For group_participants_update: which membership action to apply.",
            },
            "group_description": {
                "type": "string",
                "description": "For group_update_description: new description text. Omit/empty to clear it.",
            },
            "group_setting": {
                "type": "string",
                "enum": ["announcement", "not_announcement", "locked", "unlocked"],
                "description": "For group_setting_update: announcement/not_announcement controls who can send messages; locked/unlocked controls who can edit group info.",
            },
            "reset_invite": {
                "type": "boolean",
                "description": "For group_invite_code: true to revoke and rotate the invite link instead of just fetching it.",
            },
            "ephemeral_seconds": {
                "type": "integer",
                "description": "For group_ephemeral_update/update_default_ephemeral: 0 (off), 86400 (24h), 604800 (7d), or 7776000 (90d).",
            },
            "member_add_mode": {
                "type": "string",
                "enum": ["all_member_add", "admin_add"],
                "description": "For group_member_add_mode_update: who can add new members without an admin.",
            },
            "join_approval_mode": {
                "type": "string",
                "enum": ["on", "off"],
                "description": "For group_join_approval_mode_update: whether new joins require admin approval.",
            },
            "join_request_participants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "For group_join_requests_update: JIDs of pending join requests to approve/reject.",
            },
            "join_request_action": {
                "type": "string",
                "enum": ["approve", "reject"],
                "description": "For group_join_requests_update: whether to approve or reject the listed requests.",
            },
            "privacy_category": {
                "type": "string",
                "enum": ["last", "online", "profile", "status", "readreceipts", "groupadd", "calladd", "messages"],
                "description": "For update_privacy: which privacy setting to change (last-seen, online, profile picture, status, read receipts, who-can-add-to-groups, who-can-call, who-can-message).",
            },
            "privacy_value": {
                "type": "string",
                "description": "For update_privacy: the new value. Allowed values depend on privacy_category — most accept all/contacts/contact_blacklist/none; online accepts all/match_last_seen; groupadd accepts all/contacts/contact_blacklist; calladd accepts all/known; messages accepts all/contacts.",
            },
            "privacy_force_refresh": {
                "type": "boolean",
                "description": "For get_privacy_settings: bypass the bridge's cached settings and refetch from WhatsApp.",
            },
            "block_jid": {
                "type": "string",
                "description": "For block_contact: the JID to block or unblock.",
            },
            "block_action": {
                "type": "string",
                "enum": ["block", "unblock"],
                "description": "For block_contact: block or unblock the given jid.",
            },
            "contact_jid": {
                "type": "string",
                "description": "For update_contact: the JID of the contact to add/edit/remove.",
            },
            "contact_name": {
                "type": "string",
                "description": "For update_contact: the name to save (ignored if contact_remove is true).",
            },
            "contact_remove": {
                "type": "boolean",
                "description": "For update_contact: true to remove the saved contact instead of adding/editing it.",
            },
            "presence_type": {
                "type": "string",
                "enum": ["available", "unavailable", "composing", "recording", "paused"],
                "description": "For send_presence: the presence state to broadcast.",
            },
            "presence_chat_id": {
                "type": "string",
                "description": "For send_presence: optional chat to target (required for composing/recording/paused to mean anything; omit for account-wide available/unavailable).",
            },
            "lookup_jid": {
                "type": "string",
                "description": "For get_profile_picture_url/get_business_profile: the JID to look up (own account, a contact, or a group).",
            },
            "picture_full_res": {
                "type": "boolean",
                "description": "For get_profile_picture_url: true for the full-resolution image instead of the preview thumbnail.",
            },
            "invite_code": {
                "type": "string",
                "description": "For group_invite_preview/group_join_by_invite: the invite code (the part of a chat.whatsapp.com/XXXX link after the last slash).",
            },
            "lookup_jids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "For get_status_text/get_default_ephemeral: JIDs to look up (one call can batch several).",
            },
            "link_previews_disabled": {
                "type": "boolean",
                "description": "For update_link_previews_privacy: true to stop generating link previews on your own outgoing messages (account-wide, not per-chat).",
            },
            "presence_subscribe_jid": {
                "type": "string",
                "description": "For presence_subscribe: the JID whose presence updates (online/typing/last-seen) you want to start receiving. No unsubscribe primitive exists — subscriptions last for the socket connection's lifetime.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, return endpoint and payload without calling the bridge.",
            },
        },
        "required": ["action"],
    },
}


# whatsapp_action_admin reuses every non-action parameter definition from
# WHATSAPP_ACTION_SCHEMA verbatim (chat_id, message_id, group_*, privacy_*,
# etc. all mean the same thing in both tools) — only the action enum and the
# top-level description differ. Built once at import time, not per-call.
WHATSAPP_ACTION_ADMIN_SCHEMA = {
    "name": "whatsapp_action_admin",
    "description": (
        "Perform ADMIN WhatsApp actions through the local WhatsApp bridge: "
        "delete a message, administer a group you're in (create, add/remove/"
        "promote members, rename, set description, leave, fetch/reset invite "
        "link, toggle announcement/locked/ephemeral/member-add-mode/join-"
        "approval settings, approve/reject pending join requests, join a group "
        "via invite code), block/unblock a contact, and add/edit/remove a saved "
        "contact. These are irreversible, or affect state OTHER people see or "
        "depend on (a shared group's membership/settings, a contact's block "
        "status, a message another person can still see) — NOT your own "
        "profile/status/privacy settings, which are always available via "
        "whatsapp_action with no restriction (Hal acting on his own account is "
        "not gated). These DO require that the caller is the account owner: "
        "outside a live gateway turn (CLI, cron, an explicit owner request "
        "routed through a tool-calling session) this tool works normally; "
        "INSIDE a live WhatsApp/other-platform turn, it only executes when the "
        "turn's own sender resolves to this deployment's configured WhatsApp "
        "owner identity — every other sender (group members, non-owner DMs) is "
        "refused outright, regardless of what the message asks for. Set "
        "dry_run=true to inspect the exact bridge endpoint/payload without "
        "sending. For reversible/read-only/chat-scoped actions, use "
        "whatsapp_action instead — it has no owner restriction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(WHATSAPP_ADMIN_ACTIONS),
                "description": "WhatsApp admin action to perform.",
            },
            **{
                k: v
                for k, v in WHATSAPP_ACTION_SCHEMA["parameters"]["properties"].items()
                if k != "action"
            },
        },
        "required": ["action"],
    },
}


def _error(message: str) -> str:
    return json.dumps({"error": redact_sensitive_text(message)})


def _check_whatsapp_action() -> bool:
    try:
        config = load_gateway_config()
        return bool(config and config.platforms.get(Platform.WHATSAPP))
    except Exception:
        return False


def _whatsapp_extra() -> Dict[str, Any] | None:
    config = load_gateway_config()
    platform_cfg = config.platforms.get(Platform.WHATSAPP) if config else None
    if not platform_cfg:
        return None
    return dict(platform_cfg.extra or {})


def _bridge_token(extra: Dict[str, Any]) -> str | None:
    """Read the bridge's loopback capability token from its 0600 session-dir
    file, so this tool's direct bridge calls are authenticated the same way
    the adapter's sends are. Returns None if no token file exists (bridge
    launched without auth — manual/dev), in which case the bridge accepts
    unauthenticated calls anyway."""
    from pathlib import Path
    from hermes_constants import get_hermes_dir

    session_path = extra.get("session_path")
    if session_path:
        candidate = Path(session_path) / "bridge-token"
    else:
        candidate = Path(get_hermes_dir("platforms/whatsapp/session", "whatsapp/session")) / "bridge-token"
    try:
        token = candidate.read_text(encoding="utf-8").strip()
        return token or None
    except (OSError, ValueError):
        return None


def _require_string(args: Dict[str, Any], key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    value = str(value)
    return value if value != "" else None


def _build_action(args: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]] | str:
    """Return (http_method, endpoint, payload) or an error string."""
    action = str(args.get("action") or "").strip()

    if action == "react_message":
        chat_id = _require_string(args, "chat_id")
        message_id = _require_string(args, "message_id")
        if not chat_id or not message_id or "emoji" not in args:
            return "react_message requires chat_id, message_id, and emoji"
        return "POST", "react", {"chatId": chat_id, "messageId": message_id, "emoji": str(args.get("emoji") or "")}

    if action == "status_reply":
        status_message_id = _require_string(args, "status_message_id")
        message = _require_string(args, "message")
        if not status_message_id or not message:
            return "status_reply requires status_message_id and message"
        payload: Dict[str, Any] = {"statusMessageId": status_message_id, "message": message}
        if author := _require_string(args, "status_author_jid"):
            payload["statusAuthorJid"] = author
        return "POST", "status-reply", payload

    if action == "status_react":
        status_message_id = _require_string(args, "status_message_id")
        if not status_message_id or "emoji" not in args:
            return "status_react requires status_message_id and emoji"
        payload = {"statusMessageId": status_message_id, "emoji": str(args.get("emoji") or "")}
        if author := _require_string(args, "status_author_jid"):
            payload["statusAuthorJid"] = author
        return "POST", "status-react", payload

    if action == "post_text_status":
        message = _require_string(args, "message")
        status_jid_list = args.get("status_jid_list")
        if not message:
            return "post_text_status requires message"
        if not isinstance(status_jid_list, list) or not [x for x in status_jid_list if str(x).strip()]:
            return "post_text_status requires an explicit non-empty status_jid_list"
        payload = {
            "text": message,
            "statusJidList": [str(x).strip() for x in status_jid_list if str(x).strip()],
        }
        if background_color := _require_string(args, "background_color"):
            payload["backgroundColor"] = background_color
        if args.get("font") is not None:
            payload["font"] = int(args["font"])
        return "POST", "post-status", payload

    if action == "delete_message":
        chat_id = _require_string(args, "chat_id")
        message_id = _require_string(args, "message_id")
        from_me = args.get("from_me")
        if not chat_id or not message_id or not isinstance(from_me, bool):
            return "delete_message requires chat_id, message_id, and a boolean from_me"
        payload = {"chatId": chat_id, "messageId": message_id, "fromMe": from_me}
        if participant := _require_string(args, "participant"):
            payload["participant"] = participant
        return "POST", "delete", payload

    if action == "star_message":
        chat_id = _require_string(args, "chat_id")
        messages = args.get("star_messages")
        star = args.get("star")
        if not chat_id or not isinstance(messages, list) or not messages or not isinstance(star, bool):
            return "star_message requires chat_id, a non-empty star_messages list, and a boolean star"
        clean = [
            {"id": str(m.get("id")), "fromMe": bool(m.get("from_me"))}
            for m in messages
            if isinstance(m, dict) and m.get("id") is not None and isinstance(m.get("from_me"), bool)
        ]
        if not clean:
            return "star_message requires each star_messages entry to have id and from_me"
        return "POST", "star", {"chatId": chat_id, "messages": clean, "star": star}

    if action == "archive_chat":
        chat_id = _require_string(args, "chat_id")
        archive = args.get("archive")
        if not chat_id or not isinstance(archive, bool):
            return "archive_chat requires chat_id and a boolean archive"
        return "POST", "archive", {"chatId": chat_id, "archive": archive}

    if action == "mute_chat":
        chat_id = _require_string(args, "chat_id")
        if not chat_id or "mute_ms" not in args:
            return "mute_chat requires chat_id and mute_ms (integer ms, or null to unmute)"
        mute_ms = args.get("mute_ms")
        if mute_ms is not None and not isinstance(mute_ms, (int, float)):
            return "mute_chat requires mute_ms to be an integer or null"
        return "POST", "mute", {"chatId": chat_id, "muteMs": mute_ms}

    if action == "pin_chat":
        chat_id = _require_string(args, "chat_id")
        pin = args.get("pin")
        if not chat_id or not isinstance(pin, bool):
            return "pin_chat requires chat_id and a boolean pin"
        return "POST", "pin", {"chatId": chat_id, "pin": pin}

    if action == "mark_chat_read":
        chat_id = _require_string(args, "chat_id")
        read = args.get("read")
        if not chat_id or not isinstance(read, bool):
            return "mark_chat_read requires chat_id and a boolean read"
        return "POST", "mark-read", {"chatId": chat_id, "read": read}

    if action == "check_numbers":
        phone_numbers = args.get("phone_numbers")
        if not isinstance(phone_numbers, list) or not [p for p in phone_numbers if str(p).strip()]:
            return "check_numbers requires a non-empty phone_numbers list"
        return "POST", "check-numbers", {"phoneNumbers": [str(p).strip() for p in phone_numbers if str(p).strip()]}

    if action == "update_profile_name":
        name = _require_string(args, "name")
        if not name:
            return "update_profile_name requires name"
        return "POST", "profile/name", {"name": name}

    if action == "update_profile_status":
        status_text = args.get("status_text")
        if status_text is None:
            return "update_profile_status requires status_text"
        return "POST", "profile/status", {"status": str(status_text)}

    if action == "update_profile_picture":
        file_path = _require_string(args, "file_path")
        if not file_path:
            return "update_profile_picture requires file_path"
        payload = {"filePath": file_path}
        if profile_jid := _require_string(args, "profile_jid"):
            payload["jid"] = profile_jid
        return "POST", "profile/picture", payload

    if action == "remove_profile_picture":
        payload = {}
        if profile_jid := _require_string(args, "profile_jid"):
            payload["jid"] = profile_jid
        return "POST", "profile/picture/remove", payload

    if action == "create_group":
        subject = _require_string(args, "group_subject")
        participants = args.get("group_participants")
        if not subject or not isinstance(participants, list) or not participants:
            return "create_group requires group_subject and a non-empty group_participants list"
        return "POST", "groups/create", {
            "subject": subject,
            "participants": [str(p) for p in participants if str(p).strip()],
        }

    if action == "group_participants_update":
        chat_id = _require_string(args, "chat_id")
        participants = args.get("group_participants")
        group_action = _require_string(args, "group_action")
        if not chat_id or not isinstance(participants, list) or not participants or group_action not in (
            "add", "remove", "promote", "demote",
        ):
            return "group_participants_update requires chat_id, a non-empty group_participants list, and group_action (add/remove/promote/demote)"
        return "POST", f"groups/{chat_id}/participants", {
            "participants": [str(p) for p in participants if str(p).strip()],
            "action": group_action,
        }

    if action == "group_update_subject":
        chat_id = _require_string(args, "chat_id")
        subject = _require_string(args, "group_subject")
        if not chat_id or not subject:
            return "group_update_subject requires chat_id and group_subject"
        return "POST", f"groups/{chat_id}/subject", {"subject": subject}

    if action == "group_update_description":
        chat_id = _require_string(args, "chat_id")
        if not chat_id:
            return "group_update_description requires chat_id"
        return "POST", f"groups/{chat_id}/description", {"description": args.get("group_description") or ""}

    if action == "group_leave":
        chat_id = _require_string(args, "chat_id")
        if not chat_id:
            return "group_leave requires chat_id"
        return "POST", f"groups/{chat_id}/leave", {}

    if action == "group_invite_code":
        chat_id = _require_string(args, "chat_id")
        if not chat_id:
            return "group_invite_code requires chat_id"
        query = "?reset=true" if args.get("reset_invite") else ""
        return "GET", f"groups/{chat_id}/invite-code{query}", {}

    if action == "group_setting_update":
        chat_id = _require_string(args, "chat_id")
        setting = _require_string(args, "group_setting")
        if not chat_id or setting not in ("announcement", "not_announcement", "locked", "unlocked"):
            return "group_setting_update requires chat_id and group_setting (announcement/not_announcement/locked/unlocked)"
        return "POST", f"groups/{chat_id}/setting", {"setting": setting}

    if action == "group_ephemeral_update":
        chat_id = _require_string(args, "chat_id")
        seconds = args.get("ephemeral_seconds")
        if not chat_id or seconds not in (0, 86400, 604800, 7776000):
            return "group_ephemeral_update requires chat_id and ephemeral_seconds (0/86400/604800/7776000)"
        return "POST", f"groups/{chat_id}/ephemeral", {"seconds": seconds}

    if action == "group_member_add_mode_update":
        chat_id = _require_string(args, "chat_id")
        mode = _require_string(args, "member_add_mode")
        if not chat_id or mode not in ("all_member_add", "admin_add"):
            return "group_member_add_mode_update requires chat_id and member_add_mode (all_member_add/admin_add)"
        return "POST", f"groups/{chat_id}/member-add-mode", {"mode": mode}

    if action == "group_join_approval_mode_update":
        chat_id = _require_string(args, "chat_id")
        mode = _require_string(args, "join_approval_mode")
        if not chat_id or mode not in ("on", "off"):
            return "group_join_approval_mode_update requires chat_id and join_approval_mode (on/off)"
        return "POST", f"groups/{chat_id}/join-approval-mode", {"mode": mode}

    if action == "group_join_requests_list":
        chat_id = _require_string(args, "chat_id")
        if not chat_id:
            return "group_join_requests_list requires chat_id"
        return "GET", f"groups/{chat_id}/join-requests", {}

    if action == "group_join_requests_update":
        chat_id = _require_string(args, "chat_id")
        participants = args.get("join_request_participants")
        join_action = _require_string(args, "join_request_action")
        if not chat_id or not isinstance(participants, list) or not participants or join_action not in (
            "approve", "reject",
        ):
            return "group_join_requests_update requires chat_id, a non-empty join_request_participants list, and join_request_action (approve/reject)"
        return "POST", f"groups/{chat_id}/join-requests", {
            "participants": [str(p) for p in participants if str(p).strip()],
            "action": join_action,
        }

    if action == "update_privacy":
        category = _require_string(args, "privacy_category")
        value = _require_string(args, "privacy_value")
        if not category or not value:
            return "update_privacy requires privacy_category and privacy_value"
        return "POST", "privacy", {"category": category, "value": value}

    if action == "get_privacy_settings":
        query = "?force=true" if args.get("privacy_force_refresh") else ""
        return "GET", f"privacy{query}", {}

    if action == "block_contact":
        jid = _require_string(args, "block_jid")
        block_action = _require_string(args, "block_action")
        if not jid or block_action not in ("block", "unblock"):
            return "block_contact requires block_jid and block_action (block/unblock)"
        return "POST", "block", {"jid": jid, "action": block_action}

    if action == "get_blocklist":
        return "GET", "blocklist", {}

    if action == "update_contact":
        jid = _require_string(args, "contact_jid")
        remove = bool(args.get("contact_remove"))
        name = _require_string(args, "contact_name")
        if not jid or (not remove and not name):
            return "update_contact requires contact_jid, plus either contact_name or contact_remove:true"
        payload = {"jid": jid, "remove": remove}
        if name:
            payload["name"] = name
        return "POST", "contact", payload

    if action == "update_default_ephemeral":
        seconds = args.get("ephemeral_seconds")
        if seconds not in (0, 86400, 604800, 7776000):
            return "update_default_ephemeral requires ephemeral_seconds (0/86400/604800/7776000)"
        return "POST", "default-ephemeral", {"seconds": seconds}

    if action == "send_presence":
        presence_type = _require_string(args, "presence_type")
        if presence_type not in ("available", "unavailable", "composing", "recording", "paused"):
            return "send_presence requires presence_type (available/unavailable/composing/recording/paused)"
        payload = {"type": presence_type}
        if chat_id := _require_string(args, "presence_chat_id"):
            payload["chatId"] = chat_id
        return "POST", "presence", payload

    if action == "get_profile_picture_url":
        jid = _require_string(args, "lookup_jid")
        if not jid:
            return "get_profile_picture_url requires lookup_jid"
        query = "?type=image" if args.get("picture_full_res") else ""
        return "GET", f"profile-picture-url/{jid}{query}", {}

    if action == "get_business_profile":
        jid = _require_string(args, "lookup_jid")
        if not jid:
            return "get_business_profile requires lookup_jid"
        return "GET", f"business-profile/{jid}", {}

    if action == "group_invite_preview":
        code = _require_string(args, "invite_code")
        if not code:
            return "group_invite_preview requires invite_code"
        return "GET", f"groups/invite-info/{code}", {}

    if action == "group_join_by_invite":
        code = _require_string(args, "invite_code")
        if not code:
            return "group_join_by_invite requires invite_code"
        return "POST", "groups/accept-invite", {"code": code}

    if action == "get_status_text":
        jids = args.get("lookup_jids")
        if not isinstance(jids, list) or not [j for j in jids if str(j).strip()]:
            return "get_status_text requires a non-empty lookup_jids list"
        return "POST", "status-lookup", {"jids": [str(j).strip() for j in jids if str(j).strip()]}

    if action == "get_own_profile":
        return "GET", "me", {}

    if action == "list_groups":
        return "GET", "groups", {}

    if action == "resolve_chat":
        chat_id = str(args.get("chat_id") or "").strip()
        if not chat_id:
            return "resolve_chat requires chat_id (a group @g.us JID or a phone/@s.whatsapp.net JID)"
        return "GET", f"chat/{chat_id}", {}

    if action == "get_default_ephemeral":
        jids = args.get("lookup_jids")
        if not isinstance(jids, list) or not [j for j in jids if str(j).strip()]:
            return "get_default_ephemeral requires a non-empty lookup_jids list"
        return "POST", "ephemeral-lookup", {"jids": [str(j).strip() for j in jids if str(j).strip()]}

    if action == "update_link_previews_privacy":
        disabled = args.get("link_previews_disabled")
        if not isinstance(disabled, bool):
            return "update_link_previews_privacy requires a boolean link_previews_disabled"
        return "POST", "link-previews-privacy", {"disabled": disabled}

    if action == "presence_subscribe":
        jid = _require_string(args, "presence_subscribe_jid")
        if not jid:
            return "presence_subscribe requires presence_subscribe_jid"
        return "POST", "presence-subscribe", {"jid": jid}

    return f"Unknown WhatsApp action: {action}"


async def _post_bridge(extra: Dict[str, Any], endpoint: str, payload: Dict[str, Any], method: str = "POST") -> Dict[str, Any]:
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}

    bridge_port = extra.get("bridge_port", 3000)
    url = f"http://127.0.0.1:{bridge_port}/{endpoint.lstrip('/')}"
    token = _bridge_token(extra)
    headers = {"X-Hermes-Bridge-Token": token} if token else None
    try:
        async with aiohttp.ClientSession() as session:
            request = session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) if method == "GET" \
                else session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30))
            async with request as resp:
                try:
                    data = await resp.json()
                except Exception:
                    data = {"error": await resp.text()}
                if resp.status == 200:
                    return {
                        "success": True,
                        "platform": "whatsapp",
                        "endpoint": endpoint,
                        "message_id": data.get("messageId"),
                        "raw_response": data,
                    }
                return {"error": f"WhatsApp bridge error ({resp.status}): {redact_sensitive_text(str(data))}"}
    except Exception as exc:
        return {"error": f"WhatsApp action failed: {redact_sensitive_text(str(exc))}"}


def whatsapp_action_tool(args: Dict[str, Any], **_kw) -> str:
    extra = _whatsapp_extra()
    if extra is None:
        return tool_error("WhatsApp is not configured in the gateway")

    action = str(args.get("action") or "").strip()
    # Belt-and-suspenders: the schema enum already restricts what a
    # spec-compliant caller can send, but don't trust the enum alone —
    # some providers/harnesses don't hard-enforce JSON Schema enums. An
    # admin action reaching this handler must still be refused here.
    if action in WHATSAPP_ADMIN_ACTIONS:
        return _error(
            f"'{action}' is an admin action and is not available through "
            "whatsapp_action. Use whatsapp_action_admin instead."
        )

    # Privacy gate for the group/chat directory. Reads the live turn's
    # inbound message source fresh on every call (never cached) — same
    # call-time-not-check_fn reasoning as the admin gate below: this is a
    # per-turn-sender identity fact, so caching it in check_fn would let one
    # sender's verdict leak into a different sender's turn within the 30s
    # check_fn cache window (tools/registry.py::_check_fn_cached).
    if action in WHATSAPP_DISCLOSURE_ACTIONS:
        from tools.approval import get_current_message_source
        source = get_current_message_source()
        if source is not None and not _is_owner_whatsapp_source(source):
            return _error(
                f"whatsapp_action(action='{action}') refused: this live "
                "turn's sender does not resolve to the configured WhatsApp "
                "owner identity. Hal must never disclose which other groups "
                "or contacts he talks to -- that information is only "
                "available to the owner's own WhatsApp identity, or from "
                "outside a live gateway turn (CLI, cron)."
            )

    # Scope gate for chat-TARGETABLE safe actions (react/star/archive/mute/
    # pin/mark-read/presence). "Safe" means reversible and chat-scoped -- it
    # was never meant to mean "any chat". From a NON-owner live turn (a group
    # member, a stranger's DM) the only chat these may touch is the one the
    # turn is in; otherwise a stranger could steer the model into archiving,
    # muting or reacting inside the owner's OTHER conversations. Same class as
    # #47856 (a chosen cross-chat target from an untrusted turn) and the same
    # construction as the `react` tool's fix: the target is the current chat
    # or nothing. Owner turns and out-of-turn callers (CLI, cron) are
    # unrestricted. Call-time, uncached, for the reason given above.
    if action in WHATSAPP_CHAT_TARGETABLE_SAFE_ACTIONS:
        from tools.approval import get_current_message_source
        source = get_current_message_source()
        if source is not None and not _is_owner_whatsapp_source(source):
            requested = str(args.get("chat_id") or args.get("presence_chat_id") or "").strip()
            current = str(getattr(source, "chat_id", "") or "").strip()
            if requested and current and not _same_whatsapp_chat(requested, current):
                return _error(
                    f"whatsapp_action(action='{action}') refused: from a "
                    "non-owner conversation this action may only target the "
                    "current chat."
                )

    built = _build_action(args)
    if isinstance(built, str):
        return _error(built)
    method, endpoint, payload = built

    if args.get("dry_run") is True:
        return json.dumps({
            "success": True,
            "dry_run": True,
            "platform": "whatsapp",
            "method": method,
            "endpoint": endpoint,
            "payload": payload,
        })

    from model_tools import _run_async

    return json.dumps(_run_async(_post_bridge(extra, endpoint, payload, method)))


def whatsapp_action_admin_tool(args: Dict[str, Any], **_kw) -> str:
    extra = _whatsapp_extra()
    if extra is None:
        return tool_error("WhatsApp is not configured in the gateway")

    action = str(args.get("action") or "").strip()
    if action not in WHATSAPP_ADMIN_ACTIONS:
        return _error(
            f"'{action}' is not an admin action. Use whatsapp_action for "
            "safe/self actions (reversible, read-only, chat-scoped, or your "
            "own profile/status/privacy settings)."
        )

    # The owner-only gate. Reads the live turn's inbound message source
    # fresh on every call (never cached) — see the module docstring for why
    # this must be a call-time check, not a check_fn registration gate.
    from tools.approval import get_current_message_source
    source = get_current_message_source()
    if source is not None and not _is_owner_whatsapp_source(source):
        return _error(
            f"whatsapp_action_admin(action='{action}') refused: this live "
            "turn's sender does not resolve to the configured WhatsApp "
            "owner identity. Admin actions (irreversible, account-wide, or "
            "affecting other people) can only be triggered by the owner's "
            "own WhatsApp identity, or from outside a live gateway turn "
            "(CLI, cron)."
        )

    built = _build_action(args)
    if isinstance(built, str):
        return _error(built)
    method, endpoint, payload = built

    if args.get("dry_run") is True:
        return json.dumps({
            "success": True,
            "dry_run": True,
            "platform": "whatsapp",
            "method": method,
            "endpoint": endpoint,
            "payload": payload,
        })

    from model_tools import _run_async

    return json.dumps(_run_async(_post_bridge(extra, endpoint, payload, method)))


registry.register(
    name="whatsapp_action",
    toolset="messaging",
    schema=WHATSAPP_ACTION_SCHEMA,
    handler=whatsapp_action_tool,
    check_fn=_check_whatsapp_action,
    emoji="💬",
)

registry.register(
    name="whatsapp_action_admin",
    toolset="messaging",
    schema=WHATSAPP_ACTION_ADMIN_SCHEMA,
    handler=whatsapp_action_admin_tool,
    check_fn=_check_whatsapp_action,
    emoji="🛡️",
)
