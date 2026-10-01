"""Gateway-layer tests for WHATSAPP_GROUP_ALLOWED_USERS (chat-scoped allowlist).

Background (issue #7992, PR #8278): WhatsApp had no gateway-level equivalent of
TELEGRAM_GROUP_ALLOWED_CHATS / SIGNAL_GROUP_ALLOWED_USERS. With a populated
``WHATSAPP_ALLOWED_USERS`` (per-user DM allowlist), every group sender NOT in
that list was rejected by ``_is_user_authorized`` *after* the adapter had
already admitted the group at intake — so only the owner could invoke Hermes in
an allowlisted group, even when another member @-mentioned the bot directly.

The fix mirrors the existing chat-scoped path: ``WHATSAPP_GROUP_ALLOWED_USERS``
holds group JIDs; ANY member of a listed group is authorized, with WhatsApp JID
normalization so the group can be listed with or without the ``@g.us`` suffix.

The security-critical invariant: group authorization MUST NOT grant DM access.
A user only reachable via an allowed group must still be rejected in a DM.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


_GROUP_JID = "5215550000001-1503324008@g.us"
_GROUP_JID_BARE = "5215550000001-1503324008"
_MEMBER_LID = "236635652407450@lid"  # a non-DM-allowlisted group member


def _clear_env(monkeypatch) -> None:
    for key in (
        "WHATSAPP_ALLOWED_USERS",
        "WHATSAPP_GROUP_ALLOWED_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_runner(*, enforces: bool = True, adapter_groups=None):
    """Bare GatewayRunner with a WhatsApp adapter that owns its intake policy.

    ``adapter_groups`` simulates the adapter's resolved ``_group_allow_from``
    (the ``platforms.whatsapp.extra.group_allow_from`` config path).
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(platforms={})
    adapter = SimpleNamespace(send=AsyncMock(), enforces_own_access_policy=enforces)
    if adapter_groups is not None:
        adapter._group_allow_from = set(adapter_groups)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    return runner


def _group_source(chat_id: str = _GROUP_JID, user_id: str = _MEMBER_LID) -> SessionSource:
    return SessionSource(
        platform=Platform.WHATSAPP,
        user_id=user_id,
        chat_id=chat_id,
        user_name="member",
        chat_type="group",
    )


def _dm_source(user_id: str = _MEMBER_LID) -> SessionSource:
    return SessionSource(
        platform=Platform.WHATSAPP,
        user_id=user_id,
        chat_id=user_id,
        user_name="member",
        chat_type="dm",
    )


# ---------------------------------------------------------------------------
# Core behavior: a member of an allowed group is authorized in that group.
# ---------------------------------------------------------------------------


def test_group_member_authorized_when_group_listed(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", _GROUP_JID)
    runner = _make_runner()
    assert runner._is_user_authorized(_group_source()) is True


def test_group_listed_without_gus_suffix_still_matches(monkeypatch):
    # Operator lists the bare group id; normalization must still match the
    # @g.us-suffixed chat id the adapter reports.
    _clear_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", _GROUP_JID_BARE)
    runner = _make_runner()
    assert runner._is_user_authorized(_group_source(chat_id=_GROUP_JID)) is True


def test_wildcard_authorizes_any_group(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "*")
    runner = _make_runner()
    assert runner._is_user_authorized(_group_source(chat_id="999@g.us")) is True


def test_member_in_unlisted_group_is_rejected(monkeypatch):
    # With a populated DM allowlist (the real deployment shape), a member of a
    # group that is NOT on the group allowlist falls through to the per-user
    # check and is rejected.
    _clear_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "5215550000001")
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", _GROUP_JID)
    runner = _make_runner()
    other = _group_source(chat_id="120000000000000000@g.us")
    assert runner._is_user_authorized(other) is False


# ---------------------------------------------------------------------------
# Security invariant: group authorization MUST NOT grant DM access.
# ---------------------------------------------------------------------------


def test_group_allowlist_does_not_grant_dm(monkeypatch):
    # The member is reachable in the allowed GROUP, but the SAME user opening a
    # DM must still be rejected. A populated DM allowlist (NOT containing the
    # member) reproduces the real deployment: it disables the adapter-trust
    # fall-through so the per-user DM gate actually runs.
    _clear_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "5215550000001")
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", _GROUP_JID)
    runner = _make_runner()
    assert runner._is_user_authorized(_group_source()) is True
    assert runner._is_user_authorized(_dm_source()) is False


def test_dm_allowlist_unaffected_by_group_allowlist(monkeypatch):
    # A user who IS on the DM allowlist stays authorized in DM, independent of
    # the group allowlist.
    _clear_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "5215550000001")
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", _GROUP_JID)
    runner = _make_runner()
    owner_dm = _dm_source(user_id="5215550000001")
    assert runner._is_user_authorized(owner_dm) is True


def test_no_group_allowlist_falls_through_to_user_allowlist(monkeypatch):
    # Without WHATSAPP_GROUP_ALLOWED_USERS, a non-DM-listed group member is
    # rejected (the pre-fix behavior is preserved when the feature is unused).
    _clear_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "5215550000001")
    runner = _make_runner()
    assert runner._is_user_authorized(_group_source()) is False


# ---------------------------------------------------------------------------
# Adapter-config path: group_allow_from under platforms.whatsapp.extra, which
# the env bridge does NOT populate (it only mirrors a top-level `whatsapp:`
# key). The live adapter resolves it into `_group_allow_from`; authz must read
# that, not rely on the env var. This is the real-deployment bug (2026-06-11).
# ---------------------------------------------------------------------------


def test_group_authz_reads_adapter_config_when_env_empty(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "5215550000001")
    # No WHATSAPP_GROUP_ALLOWED_USERS env — only the adapter's resolved config.
    runner = _make_runner(adapter_groups={_GROUP_JID})
    assert runner._is_user_authorized(_group_source()) is True
    # Still no DM leak.
    assert runner._is_user_authorized(_dm_source()) is False


def test_adapter_config_group_does_not_grant_other_groups(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "5215550000001")
    runner = _make_runner(adapter_groups={_GROUP_JID})
    other = _group_source(chat_id="120000000000000000@g.us")
    assert runner._is_user_authorized(other) is False