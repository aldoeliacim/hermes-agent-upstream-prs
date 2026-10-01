"""A one-call delegation route must reach child credential resolution, not global config."""

from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent
from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task


def _resolve_without_spawning(config, *, provider=None, model=None):
    """Use the real delegation credential resolver, stopping before child creation."""
    from tools.delegate_tool_config import _resolve_delegation_credentials
    resolved = []

    def capture(route, parent):
        value = _resolve_delegation_credentials(route, parent)
        resolved.append(value)
        return value

    with (patch("tools.delegate_tool._load_config", return_value=config),
          patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=capture),
          patch("tools.delegate_tool._normalize_task_list", return_value=([], "nothing to spawn"))):
        delegate_task(tasks=[{"goal": "inspect"}], parent_agent=SimpleNamespace(_delegate_depth=0),
                      provider=provider, model=model, background=False)
    return resolved[0]


def test_model_and_provider_are_visible_and_forwarded():
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert {"provider", "model"} <= props.keys()
    agent = object.__new__(AIAgent)
    with patch("tools.delegate_tool.delegate_task", return_value="started") as spawn:
        result = agent._dispatch_delegate_task({
            "tasks": [{"goal": "inspect"}], "provider": "openrouter", "model": "vendor/model",
        })
    assert result == "started"
    assert spawn.call_args.kwargs["provider"] == "openrouter"
    assert spawn.call_args.kwargs["model"] == "vendor/model"


def test_override_does_not_mutate_global_or_borrow_old_transport():
    global_config = {
        "provider": "old", "model": "old-model", "base_url": "https://old.invalid/v1",
        "api_key": "old-placeholder", "api_mode": "codex_responses", "command": "old-command",
        "args": ["old-arg"], "request_overrides": {"old": True},
        "fallback_providers": [{"provider": "old"}],
    }
    seen = []

    def resolve(config, _parent):
        seen.append(dict(config))
        return {"provider": config.get("provider"), "model": config.get("model")}

    with (patch("tools.delegate_tool._load_config", return_value=global_config),
          patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=resolve),
          patch("tools.delegate_tool._normalize_task_list", return_value=([], "nothing to spawn"))):
        parent = SimpleNamespace(_delegate_depth=0)
        delegate_task(tasks=[{"goal": "inspect"}], parent_agent=parent,
                      provider="new", model="new-model", background=False)
        delegate_task(tasks=[{"goal": "inspect"}], parent_agent=parent, background=False)
    assert seen[0] == {"provider": "new", "model": "new-model"}
    assert seen[1] == global_config
    assert global_config["api_key"] == "old-placeholder"


def test_provider_override_without_model_is_rejected():
    with patch("tools.delegate_tool._load_config", return_value={"provider": "old", "model": "old"}):
        result = delegate_task(tasks=[{"goal": "inspect"}], provider="new",
                               parent_agent=SimpleNamespace(_delegate_depth=0))
    assert "requires model" in result


def test_real_resolver_model_only_keeps_direct_endpoint_and_auth():
    config = {"provider": "openrouter", "model": "old-model", "base_url": "https://pinned.invalid/v1",
              "api_key": "pinned-placeholder", "api_mode": "chat_completions"}
    creds = _resolve_without_spawning(config, model="other-model")
    assert (creds["model"], creds["base_url"], creds["api_key"]) == (
        "other-model", "https://pinned.invalid/v1", "pinned-placeholder")
    assert config["model"] == "old-model"


def test_real_resolver_explicit_provider_owns_endpoint_even_when_slug_matches():
    config = {"provider": "openrouter", "model": "old-model", "base_url": "https://pinned.invalid/v1",
              "api_key": "pinned-placeholder", "api_mode": "codex_responses",
              "fallback_providers": [{"provider": "old-provider"}]}
    runtime = {"provider": "openrouter", "model": "new-model", "api_key": "new-placeholder",
               "base_url": "https://new.invalid/v1", "api_mode": "chat_completions",
               "request_overrides": {"new": True}}
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime) as resolve:
        creds = _resolve_without_spawning(config, provider="openrouter", model="new-model")
    resolve.assert_called_once_with(requested="openrouter", target_model="new-model")
    assert (creds["model"], creds["base_url"], creds["api_key"], creds["api_mode"]) == (
        "new-model", "https://new.invalid/v1", "new-placeholder", "chat_completions")
    assert creds["request_overrides"] == {"new": True}
    assert config["base_url"] == "https://pinned.invalid/v1"
