"""Agent dispatch must not turn an explicit deny-all into legacy defaults."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import invoke_tool
from agent.tool_executor import execute_tool_calls_sequential
from run_agent import AIAgent


@pytest.fixture
def agent(tmp_path):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", tmp_path),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    agent._subdirectory_hints.check_tool_call = MagicMock(return_value="")
    return agent


@pytest.mark.parametrize("caller", ["sequential-quiet", "sequential-loud", "helper"])
@pytest.mark.parametrize(
    "authority",
    [pytest.param(set(), id="deny-all"),
     pytest.param({"execute_code", "read_file"}, id="bounded"),
     pytest.param(None, id="legacy-default")],
)
def test_dispatch_preserves_explicit_tool_authority(agent, caller, authority):
    # Exercise real caller and middleware code; intercept only the downstream
    # dispatcher so no Python execution or external tool side effects occur.
    agent.valid_tool_names = authority
    agent.quiet_mode = caller != "sequential-loud"
    args = {"code": "print('authority probe')"}
    with patch("model_tools.handle_function_call", return_value='{"ok": true}') as dispatch:
        if caller == "helper":
            result = invoke_tool(agent, "execute_code", args, "task", tool_call_id="call")
            assert result == '{"ok": true}'
        else:
            import json

            assistant = SimpleNamespace(tool_calls=[SimpleNamespace(
                id="call", type="function",
                function=SimpleNamespace(name="execute_code", arguments=json.dumps(args)),
            )])
            messages = []
            execute_tool_calls_sequential(agent, assistant, messages, "task")
            assert [m["tool_call_id"] for m in messages] == ["call"]

    dispatch.assert_called_once()
    assert dispatch.call_args.args == ("execute_code", args, "task")
    forwarded = dispatch.call_args.kwargs["enabled_tools"]
    if authority is None:
        assert forwarded is None
    else:
        assert isinstance(forwarded, list)
        assert set(forwarded) == authority
        if not authority:
            assert forwarded == []
