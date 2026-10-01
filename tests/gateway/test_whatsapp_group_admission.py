"""Gateway-authorization tests for WhatsApp group admission (chat-scoped allowlist).

Background: WhatsApp had no gateway-level equivalent of TELEGRAM_GROUP_ALLOWED_CHATS.
With a populated ``WHATSAPP_ALLOWED_USERS`` (per-user DM allowlist), every group
sender NOT in that list was rejected by ``_is_user_authorized`` *after* the adapter
had already admitted the group at intake — so only the DM-listed owner could invoke
Hermes in an allowlisted group, even when another member @-mentioned the bot directly.

The fix (``_whatsapp_group_chat_authorized`` in ``gateway/authz_mixin.py``) mirrors the
existing chat-scoped path: ``WHATSAPP_GROUP_ALLOWED_USERS`` / the adapter's resolved
``group_allow_from`` hold group JIDs; ANY member of a listed group is authorized, with
WhatsApp JID normalization so the group can be listed with or without the ``@g.us``
suffix, and phone/LID aliases of the same principal are recognized.

Security invariants (all fail-closed):
  * Group authorization MUST NOT grant DM access (a member only reachable via an
    allowed group is still rejected in a DM).
  * A member of an UNLISTED group falls through to the per-user check and is rejected.
  * No group allowlist configured => pre-fix behavior preserved (per-user gate only).

Identifiers here are synthetic fixtures — no deployment phone numbers or LIDs are
hardcoded, so the test stays valid regardless of which account is deployed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


# --- synthetic fixtures (NOT real deployment identities) -------------------
# A group JID: <phone>-<epoch>@g.us is the legacy group form; the bare form
# drops the @g.us suffix. Normalization must treat them as the same group.
_GROUP_JID = "10000000001-1700000000@g.us"
_GROUP_JID_BARE = "10000000001-1700000000"
_OTHER_GROUP_JID = "10000000002-1700000000@g.us"

# A group member who is NOT on the DM allowlist, expressed as a LID.
_MEMBER_LID = "29999999999999@lid"

# The DM-allowlisted principal (e.g. the operator/owner), expressed as a phone.
_DM_ALLOWED_USER = "10000000009"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in (
        "WHATSAPP_ALLOWED_USERS",
        "WHATSAPP_GROUP_ALLOWED_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


def _make_runner(*, enforces: bool = True, adapter_groups=None):
    """Bare GatewayRunner with a WhatsApp adapter that owns its intake policy.

    ``adapter_groups`` simulates the adapter's resolved ``_group_allow_from``
    (the ``platforms.whatsapp.extra.group_allow_from`` config path), which the
    env bridge does NOT populate.
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
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", _GROUP_JID)
    runner = _make_runner()
    assert runner._is_user_authorized(_group_source()) is True


def test_group_listed_without_gus_suffix_still_matches(monkeypatch):
    # Operator lists the bare group id; normalization must still match the
    # @g.us-suffixed chat id the adapter reports.
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", _GROUP_JID_BARE)
    runner = _make_runner()
    assert runner._is_user_authorized(_group_source(chat_id=_GROUP_JID)) is True


def test_wildcard_authorizes_any_group(monkeypatch):
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "*")
    runner = _make_runner()
    assert runner._is_user_authorized(_group_source(chat_id="99999@g.us")) is True


def test_member_in_unlisted_group_is_rejected(monkeypatch):
    # With a populated DM allowlist (the real deployment shape), a member of a
    # group that is NOT on the group allowlist falls through to the per-user
    # check and is rejected (fail closed).
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", _DM_ALLOWED_USER)
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", _GROUP_JID)
    runner = _make_runner()
    assert runner._is_user_authorized(_group_source(chat_id=_OTHER_GROUP_JID)) is False


# ---------------------------------------------------------------------------
# Security invariant: group authorization MUST NOT grant DM access.
# ---------------------------------------------------------------------------


def test_group_allowlist_does_not_grant_dm(monkeypatch):
    # The member is reachable in the allowed GROUP, but the SAME user opening a
    # DM must still be rejected. A populated DM allowlist (NOT containing the
    # member) reproduces the real deployment: it disables the adapter-trust
    # fall-through so the per-user DM gate actually runs.
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", _DM_ALLOWED_USER)
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", _GROUP_JID)
    runner = _make_runner()
    assert runner._is_user_authorized(_group_source()) is True
    assert runner._is_user_authorized(_dm_source()) is False


def test_dm_allowlist_unaffected_by_group_allowlist(monkeypatch):
    # A user who IS on the DM allowlist stays authorized in DM, independent of
    # the group allowlist.
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", _DM_ALLOWED_USER)
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", _GROUP_JID)
    runner = _make_runner()
    assert runner._is_user_authorized(_dm_source(user_id=_DM_ALLOWED_USER)) is True


def test_no_group_allowlist_falls_through_to_user_allowlist(monkeypatch):
    # Without WHATSAPP_GROUP_ALLOWED_USERS, a non-DM-listed group member is
    # rejected (pre-fix behavior preserved when the feature is unused).
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", _DM_ALLOWED_USER)
    runner = _make_runner()
    assert runner._is_user_authorized(_group_source()) is False


# ---------------------------------------------------------------------------
# Adapter-config path: group_allow_from under platforms.whatsapp.extra, which
# the env bridge does NOT populate. The live adapter resolves it into
# _group_allow_from; authz must read that, not rely on the env var.
# ---------------------------------------------------------------------------


def test_group_authz_reads_adapter_config_when_env_empty(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", _DM_ALLOWED_USER)
    # No WHATSAPP_GROUP_ALLOWED_USERS env — only the adapter's resolved config.
    runner = _make_runner(adapter_groups={_GROUP_JID})
    assert runner._is_user_authorized(_group_source()) is True
    # Still no DM leak.
    assert runner._is_user_authorized(_dm_source()) is False


def test_adapter_config_group_does_not_grant_other_groups(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", _DM_ALLOWED_USER)
    runner = _make_runner(adapter_groups={_GROUP_JID})
    assert runner._is_user_authorized(_group_source(chat_id=_OTHER_GROUP_JID)) is False


def test_adapter_config_accepts_frozenset_group_allow_from(monkeypatch):
    # _coerce_allow_set must accept tuple/set/frozenset (not only list), because
    # the adapter resolves _group_allow_from into a set. A frozenset must not be
    # stringified whole and fail to match.
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", _DM_ALLOWED_USER)
    runner = _make_runner(adapter_groups=frozenset({_GROUP_JID}))
    assert runner._is_user_authorized(_group_source()) is True


# ---------------------------------------------------------------------------
# Identity-alias recognition: the DM-allowlisted owner is recognized in DM
# under any of their phone/LID aliases (own-account identity aliases), driven
# by a fixture map — NO deployment identity is hardcoded into the resolver.
# ---------------------------------------------------------------------------


def test_owner_recognized_across_phone_and_lid_aliases(monkeypatch):
    # The operator listed their PHONE in WHATSAPP_ALLOWED_USERS, but the inbound
    # DM arrives under their LID (Baileys surfaces either). The alias expander
    # must recognize both as the same principal so the owner is authorized.
    owner_phone = "10000000009"
    owner_lid = "31111111111111@lid"

    def _fake_expand(identifier: str):
        norm = identifier.split("@")[0]
        # Map either alias of THIS owner to the shared principal set.
        if norm in {owner_phone, owner_lid.split("@")[0]}:
            return {owner_phone, owner_lid, owner_lid.split("@")[0]}
        return {norm} if norm else set()

    # Patch only the alias expander used by the gateway principal matcher; the
    # owner identity comes from the fixture, not from any deployment file.
    monkeypatch.setattr(
        "gateway.authz_mixin._expand_whatsapp_auth_aliases", _fake_expand
    )
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", owner_phone)
    runner = _make_runner()
    # Owner DM arriving under the LID alias is authorized.
    assert runner._is_user_authorized(_dm_source(user_id=owner_lid)) is True
    # A stranger under a different LID is still rejected (fail closed).
    assert runner._is_user_authorized(_dm_source(user_id="40000000000000@lid")) is False
