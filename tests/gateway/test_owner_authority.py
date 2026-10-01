"""Tests for the generic owner-authority predicate (gateway/owner_authority.py).

The predicate answers ONE question for security gates: "is this message source
the gateway's proven primary owner?" It is deliberately narrow and FAILS CLOSED
(treats an unproven source as a non-owner). It is NOT the semantic USER.md
profile-suppression helper — it owns no identity semantics beyond a home-channel
match, and hardcodes no IDs.
"""
from types import SimpleNamespace

import pytest

from gateway.config import HomeChannel, Platform
from gateway.owner_authority import (
    home_is_configured_for,
    source_is_non_owner,
    source_is_owner,
)


def _source(platform, chat_id, user_id=None, **kw):
    return SimpleNamespace(
        platform=platform, chat_id=chat_id, user_id=user_id,
        chat_id_alt=kw.get("chat_id_alt"), user_id_alt=kw.get("user_id_alt"),
    )


class _Config:
    """Minimal stand-in exposing get_home_channel like GatewayConfig."""

    def __init__(self, homes):
        self._homes = homes  # {Platform: HomeChannel}

    def get_home_channel(self, platform):
        return self._homes.get(platform)


def _home(platform, chat_id, user_id=None):
    return HomeChannel(platform=platform, chat_id=chat_id, name="Home", user_id=user_id)


# ── LOCAL is always the owner (the operator's own terminal) ──────────────────

def test_local_source_is_owner():
    cfg = _Config({})
    src = _source(Platform.LOCAL, "cli")
    assert source_is_owner(cfg, src) is True
    assert source_is_non_owner(cfg, src) is False


# ── Exact home-channel chat_id match => owner ────────────────────────────────

def test_matching_home_chat_id_is_owner():
    cfg = _Config({Platform.TELEGRAM: _home(Platform.TELEGRAM, "555111")})
    src = _source(Platform.TELEGRAM, "555111", user_id="555111")
    assert source_is_owner(cfg, src) is True
    assert source_is_non_owner(cfg, src) is False


def test_shared_home_chat_does_not_make_members_owners():
    cfg = _Config({Platform.WHATSAPP: _home(Platform.WHATSAPP, "12345@g.us")})
    member = _source(Platform.WHATSAPP, "12345@g.us", user_id="unrelated@lid")
    assert source_is_owner(cfg, member, session_key="whatsapp:group:12345@g.us") is False
    assert source_is_non_owner(cfg, member, session_key="whatsapp:group:12345@g.us") is True


def test_shared_home_with_explicit_owner_id_requires_matching_actor():
    cfg = _Config({Platform.WHATSAPP: _home(Platform.WHATSAPP, "12345@g.us", "owner@lid")})
    member = _source(Platform.WHATSAPP, "12345@g.us", user_id="unrelated@lid")
    owner = _source(Platform.WHATSAPP, "12345@g.us", user_id="owner@lid")
    assert source_is_owner(cfg, member) is False
    assert source_is_owner(cfg, owner) is True


def test_shared_home_alt_room_alias_cannot_prove_owner():
    cfg = _Config({Platform.WHATSAPP: _home(Platform.WHATSAPP, "12345@g.us")})
    member = _source(Platform.WHATSAPP, "opaque-room", user_id="unrelated@lid", chat_id_alt="12345@g.us")
    member.chat_type = "group"
    assert source_is_owner(cfg, member, session_key="whatsapp:group:12345@g.us") is False


def test_shared_home_channel_requires_actor_not_channel_id():
    cfg = _Config({Platform.TELEGRAM: _home(Platform.TELEGRAM, "-10012345", "owner-id")})
    member = _source(Platform.TELEGRAM, "-10012345", user_id="not-owner")
    member.chat_type = "channel"
    owner = _source(Platform.TELEGRAM, "-10012345", user_id="owner-id")
    owner.chat_type = "channel"
    assert source_is_owner(cfg, member) is False
    assert source_is_owner(cfg, owner) is True


# ── A different chat on the same platform => NON-owner (fail closed) ──────────

def test_non_home_chat_is_non_owner():
    cfg = _Config({Platform.TELEGRAM: _home(Platform.TELEGRAM, "555111")})
    src = _source(Platform.TELEGRAM, "999222", user_id="999222")
    assert source_is_owner(cfg, src) is False
    assert source_is_non_owner(cfg, src) is True


# ── No home configured for the platform => NON-owner (fail closed) ───────────

def test_unconfigured_platform_is_non_owner():
    cfg = _Config({})
    src = _source(Platform.TELEGRAM, "555111", user_id="555111")
    assert source_is_owner(cfg, src) is False
    assert source_is_non_owner(cfg, src) is True


# ── user_config home_channel is honored (secondary profile owner) ────────────

def test_user_config_home_channel_matches():
    cfg = _Config({})
    user_config = {"platforms": {"telegram": {"home_channel": {"chat_id": "777333"}}}}
    src = _source(Platform.TELEGRAM, "777333", user_id="777333")
    assert source_is_owner(cfg, src, user_config=user_config) is True


# ── Phone-style digit normalization: '+' prefix vs bare digits still matches ─

def test_phone_digit_normalization_matches():
    cfg = _Config({Platform.SMS: _home(Platform.SMS, "+52155510000")})
    src = _source(Platform.SMS, "52155510000", user_id="52155510000")
    assert source_is_owner(cfg, src) is True


# ── session_key tail digits can prove the owner when source ids are opaque ───

def test_session_key_tail_digits_match_home():
    cfg = _Config({Platform.WHATSAPP: _home(Platform.WHATSAPP, "5215551234567")})
    # Legacy private-session key can carry a resolved phone tail.
    src = _source(Platform.WHATSAPP, "1111@lid", user_id="1111@lid")
    key = "whatsapp:dm:5215551234567"
    assert source_is_owner(cfg, src, session_key=key) is True


def test_group_session_key_tail_cannot_prove_actor():
    cfg = _Config({Platform.WHATSAPP: _home(Platform.WHATSAPP, "5215551234567")})
    src = _source(Platform.WHATSAPP, "12345@g.us", user_id="unrelated@lid")
    src.chat_type = "group"
    assert source_is_owner(cfg, src, session_key="whatsapp:group:5215551234567") is False


# ── Fail closed on WhatsApp opaque id with NO proving signal => non-owner ────

def test_whatsapp_opaque_without_proof_is_non_owner():
    cfg = _Config({Platform.WHATSAPP: _home(Platform.WHATSAPP, "5215551234567")})
    src = _source(Platform.WHATSAPP, "9999@lid", user_id="9999@lid")
    assert source_is_owner(cfg, src, session_key="whatsapp:group:abcd@lid") is False
    assert source_is_non_owner(cfg, src, session_key="whatsapp:group:abcd@lid") is True


# ── A None/absent source fails closed (non-owner) ───────────────────────────

def test_none_source_is_non_owner():
    cfg = _Config({Platform.TELEGRAM: _home(Platform.TELEGRAM, "555111")})
    assert source_is_owner(cfg, None) is False
    assert source_is_non_owner(cfg, None) is True


# ── An exploding config never crashes the gate; it fails CLOSED ─────────────

def test_exploding_config_fails_closed():
    class _Boom:
        def get_home_channel(self, platform):
            raise RuntimeError("boom")

    src = _source(Platform.TELEGRAM, "555111", user_id="555111")
    # No proof obtainable => non-owner.
    assert source_is_owner(_Boom(), src) is False
    assert source_is_non_owner(_Boom(), src) is True


# ── chat_id_alt / user_id_alt participate in the match ──────────────────────

def test_alt_identifier_matches_home():
    cfg = _Config({Platform.SIGNAL: _home(Platform.SIGNAL, "uuid-abc")})
    src = _source(Platform.SIGNAL, "group-xyz", user_id="phone", user_id_alt="uuid-abc")
    assert source_is_owner(cfg, src) is True


# ── home_is_configured_for: the suppression-gate guard ──────────────────────

def test_home_configured_true_when_platform_home_present():
    cfg = _Config({Platform.TELEGRAM: _home(Platform.TELEGRAM, "555111")})
    src = _source(Platform.TELEGRAM, "999222")
    assert home_is_configured_for(cfg, src) is True


def test_home_configured_false_when_no_home():
    cfg = _Config({})
    src = _source(Platform.TELEGRAM, "999222")
    # No home to protect => suppression gate stays inert (single-user deploy).
    assert home_is_configured_for(cfg, src) is False


def test_home_configured_false_for_local_and_none():
    cfg = _Config({Platform.TELEGRAM: _home(Platform.TELEGRAM, "555111")})
    assert home_is_configured_for(cfg, _source(Platform.LOCAL, "cli")) is False
    assert home_is_configured_for(cfg, None) is False


def test_home_configured_false_when_config_explodes():
    class _Boom:
        def get_home_channel(self, platform):
            raise RuntimeError("boom")

    assert home_is_configured_for(_Boom(), _source(Platform.TELEGRAM, "x")) is False


def test_suppression_gate_composition_matches_intent():
    """The notify-suppression gate = home_is_configured_for AND source_is_non_owner."""
    cfg = _Config({Platform.TELEGRAM: _home(Platform.TELEGRAM, "555111")})
    owner = _source(Platform.TELEGRAM, "555111", user_id="555111")
    stranger = _source(Platform.TELEGRAM, "999222", user_id="999222")
    no_home_cfg = _Config({})

    # Suppress the stranger (home exists, not owner).
    assert home_is_configured_for(cfg, stranger) and source_is_non_owner(cfg, stranger)
    # Never suppress the owner.
    assert not (home_is_configured_for(cfg, owner) and source_is_non_owner(cfg, owner))
    # Never suppress in a no-home single-user deploy.
    assert not (
        home_is_configured_for(no_home_cfg, stranger)
        and source_is_non_owner(no_home_cfg, stranger)
    )
