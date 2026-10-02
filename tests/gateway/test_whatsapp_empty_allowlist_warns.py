"""An ``allowlist`` policy with an empty list must be loud, not silently total.

``WHATSAPP_GROUP_POLICY=allowlist`` reads like a securely-restricted configuration. Paired with
an empty ``WHATSAPP_GROUP_ALLOWED_USERS`` it instead rejects **100%** of group traffic — and
because the bridge destructively dequeues before the gate runs, each rejection leaves no trace:
no error, no log line, no retry. A deployment ran that way for weeks before anyone noticed.

The DM half has the same shape, so it is asserted here too, along with the cases that must stay
quiet — a warning that fires on correct configuration is noise that trains people to ignore it.
"""

from __future__ import annotations

import logging

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

GROUP_WARNING = "group allowlist is EMPTY"
DM_WARNING = "WHATSAPP_ALLOWED_USERS is empty"


def _adapter(*, group_policy="allowlist", group_allow_from=None,
             dm_policy="allowlist", allow_from=None) -> WhatsAppAdapter:
    """Build an adapter carrying only the four policy fields under test."""
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._group_policy = group_policy
    adapter._group_allow_from = WhatsAppAdapter._coerce_allow_list(group_allow_from)
    adapter._dm_policy = dm_policy
    adapter._allow_from = WhatsAppAdapter._coerce_allow_list(allow_from)
    return adapter


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_empty_group_allowlist_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _adapter(group_policy="allowlist", group_allow_from=None)._warn_on_empty_allowlist()
    assert any(GROUP_WARNING in m for m in _messages(caplog)), _messages(caplog)


def test_empty_dm_allowlist_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _adapter(dm_policy="allowlist", allow_from=None)._warn_on_empty_allowlist()
    assert any(DM_WARNING in m for m in _messages(caplog)), _messages(caplog)


@pytest.mark.parametrize(
    "group_allow_from",
    [
        pytest.param(["120363400236163214@g.us"], id="group-jid"),
        pytest.param(["5215514706713-1463600871@g.us"], id="family-ap"),
        pytest.param(["  120363400236163214@g.us  "], id="padded"),
    ],
)
def test_populated_group_allowlist_is_quiet(caplog, group_allow_from):
    """A correctly configured allowlist must NOT warn — no crying wolf."""
    with caplog.at_level(logging.WARNING):
        _adapter(group_policy="allowlist", group_allow_from=group_allow_from)._warn_on_empty_allowlist()
    assert not any(GROUP_WARNING in m for m in _messages(caplog)), _messages(caplog)


def test_populated_dm_allowlist_is_quiet(caplog):
    with caplog.at_level(logging.WARNING):
        _adapter(dm_policy="allowlist", allow_from=["5215527115258"])._warn_on_empty_allowlist()
    assert not any(DM_WARNING in m for m in _messages(caplog)), _messages(caplog)


@pytest.mark.parametrize("policy", ["open", "pairing", "disabled"])
def test_non_allowlist_policy_does_not_warn(caplog, policy):
    """Only ``allowlist`` has the empty-list footgun; the rest are self-explanatory."""
    with caplog.at_level(logging.WARNING):
        _adapter(group_policy=policy, group_allow_from=None)._warn_on_empty_allowlist()
    assert not any(GROUP_WARNING in m for m in _messages(caplog)), _messages(caplog)