"""Sender model defaults: persistence, identity isolation, precedence and local management."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.sender_model_preference import get_sender_model_preference, sender_keys
from gateway.session import SessionSource
from hermes_state import SessionDB


def test_persistence_platform_isolation_and_alias_conflict(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.set_sender_model_preference("whatsapp", "999", "openrouter", "vendor/model")
    assert db.get_sender_model_preference("telegram", ["999"]) is None
    db.close()
    db = SessionDB(path)
    monkeypatch.setattr("gateway.sender_model_preference.expand_whatsapp_aliases",
                        lambda value: {"555", "999"})
    source = SessionSource(platform=Platform.WHATSAPP, chat_id="room@g.us", user_id="555@s.whatsapp.net")
    assert sender_keys(source) == ("555", "999")
    assert get_sender_model_preference(db, source) == ("openrouter", "vendor/model")
    db.set_sender_model_preference("whatsapp", "555", "other", "different")
    with pytest.raises(ValueError, match="Conflicting"):
        get_sender_model_preference(db, source)
    db.close()


def test_sender_is_not_chat_and_unknown_author_has_no_preference():
    assert sender_keys(SessionSource(platform=Platform.TELEGRAM, chat_id="room", user_id="123")) == ("123",)
    assert sender_keys(SessionSource(platform=Platform.TELEGRAM, chat_id="room")) == ()


def test_gateway_precedence_and_provider_bundle(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.set_sender_model_preference("discord", "person", "openrouter", "vendor/sender")
    runner = object.__new__(GatewayRunner)
    runner.__dict__["session_store"] = SimpleNamespace(_db_for_key=lambda key: db)
    runner._session_model_overrides = {}
    runner.config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(
        enabled=True, channel_overrides={"room": ChannelOverride(model="vendor/channel", provider="anthropic")})})
    source = SessionSource(platform=Platform.DISCORD, chat_id="dm", user_id="person")
    calls = []

    def route(provider, target_model=None):
        calls.append((provider, target_model))
        return {"provider": provider, "api_key": "test-only", "base_url": "https://example.invalid/v1",
                "api_mode": "chat_completions"}

    with (patch("gateway.run._resolve_gateway_model", return_value="vendor/default"),
          patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"provider": "default", "api_key": "test-only"}),
          patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", side_effect=route)):
        model, runtime = runner._resolve_session_agent_runtime(source=source)
        assert (model, runtime["provider"]) == ("vendor/sender", "openrouter")
        assert calls[-1] == ("openrouter", "vendor/sender")
        source.chat_id = "room"
        model, runtime = runner._resolve_session_agent_runtime(source=source)
        assert (model, runtime["provider"]) == ("vendor/channel", "anthropic")
        assert calls[-1] == ("anthropic", "vendor/channel")
    db.close()


def test_multiplex_store_scope_and_no_db_legacy(tmp_path):
    first = SessionDB(tmp_path / "first.db")
    second = SessionDB(tmp_path / "second.db")
    first.set_sender_model_preference("discord", "person", "first", "first/model")
    second.set_sender_model_preference("discord", "person", "second", "second/model")
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.__dict__["config"] = None
    runner.__dict__["session_store"] = None
    source = SessionSource(platform=Platform.DISCORD, chat_id="room", user_id="person")
    with (patch("gateway.run._resolve_gateway_model", return_value="default/model"),
          patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"provider": "default", "api_key": "test-only"}),
          patch("gateway.run._resolve_runtime_agent_kwargs_for_provider",
                side_effect=lambda provider, target_model=None: {"provider": provider, "api_key": "test-only"})):
        assert runner._resolve_session_agent_runtime(source=source)[0] == "default/model"
        runner.__dict__["session_store"] = SimpleNamespace(_db_for_key=lambda key: second if key.startswith("agent:other:") else first)
        assert runner._resolve_session_agent_runtime(source=source, session_key="agent:other:discord:room")[0] == "second/model"
        assert runner._resolve_session_agent_runtime(source=source, session_key="agent:main:discord:room")[0] == "first/model"
    first.close()
    second.close()


def test_local_cli_set_list_clear(tmp_path, monkeypatch, capsys):
    import argparse
    from hermes_cli.subcommands.sender_model import cmd_sender_model
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={"provider": "openrouter"}):
        cmd_sender_model(argparse.Namespace(sender_model_action="set", platform="telegram",
                                             sender_id="123", provider="openrouter", model="vendor/model"))
    capsys.readouterr()
    cmd_sender_model(argparse.Namespace(sender_model_action="list"))
    assert "telegram\t123\topenrouter\tvendor/model" in capsys.readouterr().out
    cmd_sender_model(argparse.Namespace(sender_model_action="clear", platform="telegram", sender_id="123"))
    capsys.readouterr()
    cmd_sender_model(argparse.Namespace(sender_model_action="list"))
    assert capsys.readouterr().out == ""
