"""The judges are told the name the recipient is addressed by, from config.

A recipient's stored name can differ from the name they are addressed by (stored
"Morgan", addressed "Robin"). A register judge that only hears "Morgan" reads
"¡Perfecto, Robin!" as a message ABOUT a third person and votes LEAK, stably. The
mapping is operator knowledge, so it lives in config (privacy.recipient_aliases),
never in source.
"""
from __future__ import annotations

import asyncio

import pytest

import gateway.response_filters as rf

CHAT = "80000000000002@lid"


def _config(monkeypatch, table):
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run, "_load_gateway_config", lambda: {"privacy": {"recipient_aliases": table}}
    )


def test_label_adds_the_configured_address_name(monkeypatch):
    _config(monkeypatch, {CHAT: ["Robin"]})
    assert rf.recipient_label_for_judge(CHAT, "Morgan") == "Morgan (also addressed as: Robin)"


def test_bare_string_value_is_one_name(monkeypatch):
    _config(monkeypatch, {CHAT: "Robin"})
    assert rf.recipient_aliases_for(CHAT, "Morgan") == ("Robin",)


def test_no_entry_leaves_the_label_as_the_stored_name(monkeypatch):
    _config(monkeypatch, {"80000000000009@lid": ["Alex"]})
    assert rf.recipient_label_for_judge(CHAT, "Morgan") == "Morgan"


def test_alias_equal_to_the_stored_name_is_not_repeated(monkeypatch):
    _config(monkeypatch, {CHAT: ["morgan", "Robin"]})
    assert rf.recipient_label_for_judge(CHAT, "Morgan") == "Morgan (also addressed as: Robin)"


def test_configured_alias_also_arms_the_deterministic_name_guard(monkeypatch):
    _config(monkeypatch, {CHAT: ["Robin"]})
    assert "Robin" in rf.recipient_aliases_for(CHAT, "Morgan")


def test_empty_stored_name_still_gets_the_configured_names(monkeypatch):
    _config(monkeypatch, {CHAT: ["Robin"]})
    assert rf.recipient_label_for_judge(CHAT, "") == "Robin"


@pytest.mark.parametrize("bad", [None, [], "not-a-dict", {CHAT: 7}, {CHAT: None}])
def test_malformed_config_degrades_to_derivation_alone(monkeypatch, bad):
    _config(monkeypatch, bad)
    assert rf.recipient_label_for_judge(CHAT, "Morgan") == "Morgan"


def test_config_error_degrades_to_derivation_alone(monkeypatch):
    import gateway.run as gateway_run

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(gateway_run, "_load_gateway_config", _boom)
    assert rf.recipient_label_for_judge(CHAT, "Morgan") == "Morgan"


def test_the_live_turn_tells_the_judges_the_address_name(monkeypatch, tmp_path):
    """Wiring through the real inbound path: the recipient label the judge lanes see
    carries the configured address name."""
    from tests.gateway import test_owner_register_gate_e2e as e2e

    _config(monkeypatch, {e2e.CONTACT: ["Robin"]})
    seen = []
    import gateway.delivery_judge as dj

    real = dj.judge_delivery

    def _spy(text, recipient, **kw):
        seen.append(recipient)
        return real(text, recipient, **kw)

    monkeypatch.setattr(dj, "judge_delivery", _spy)
    asyncio.run(e2e._turn(monkeypatch, tmp_path, text=e2e.REPLY, verdict="DELIVER", name="Morgan"))
    assert seen and seen[0] == "Morgan (also addressed as: Robin)"
