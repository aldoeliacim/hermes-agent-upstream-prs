"""Regression coverage for send_message's agent-tool registration.

CONFIRMED real casualty (2026-07-11, group "TAMHAL Y JVic"): the entire
tool-gated reply-gate mechanism (gateway/run.py, gateway/reply_policy.py,
agent/reply_decision_stop.py — built and shipped across three commits,
2026-07-06 through 2026-07-11) depended on the model being able to call
send_message(target="current", ...). It never actually could: upstream
c6c8abbad (2026-06-17, #47856) removed both the registry.register() call in
tools/send_message_tool.py AND the "send_message" entry from
toolsets.py's _HERMES_CORE_TOOLS, and neither was ever restored. The model,
correctly finding no such tool in its schema, fell back to a `hermes send`
CLI workaround (via terminal) that bypassed reply-gate bookkeeping entirely,
producing a narrated-status double-send into a real family WhatsApp group.

These tests assert the tool is genuinely reachable end-to-end (registered +
toolset-listed + check_fn-gated correctly) so this specific class of "the
mechanism looks complete in gateway/run.py but the model never had the tool"
bug can never silently recur.
"""

from __future__ import annotations

import pytest

import tools.registry as registry
import model_tools  # noqa: F401  # imports every tool module, populating the registry
from toolsets import resolve_toolset


@pytest.fixture(autouse=True)
def _clean_check_fn_cache():
    """check_fn results are TTL-cached process-wide (tools/registry.py).
    Tests in this file monkeypatch gateway.config.load_gateway_config to
    flip reply_gate_mode mid-run — without invalidating the cache before
    AND after, a cached True/False from one test leaks into the next test
    (in this file or any other file sharing the process) and produces a
    flaky, order-dependent failure instead of a deterministic one."""
    registry.invalidate_check_fn_cache()
    yield
    registry.invalidate_check_fn_cache()


def test_send_message_is_registered_in_tool_registry():
    """The upstream de-registration (c6c8abbad) must be reversed: send_message
    needs a real ToolEntry, not just a schema constant sitting unused."""
    entry = registry.registry._tools.get("send_message")
    assert entry is not None, (
        "send_message has no registry.register() call — the entire "
        "tool-gated reply-gate mechanism has no tool to call"
    )
    assert entry.toolset == "messaging"
    assert entry.check_fn is not None, (
        "send_message must be check_fn-gated, never unconditionally on "
        "(that would violate upstream's #47856 rationale for prompt mode)"
    )


def test_send_message_in_whatsapp_core_toolset():
    """toolsets.py must actually route send_message to messaging platforms —
    registering the tool is necessary but not sufficient; it also has to be
    a member of the toolset every platform's composite resolves to."""
    resolved = set(resolve_toolset("hermes-whatsapp"))
    assert "send_message" in resolved
    # Sanity: the other messaging platforms that share _HERMES_CORE_TOOLS
    # must carry it too, not just WhatsApp.
    assert "send_message" in set(resolve_toolset("hermes-telegram"))
    assert "send_message" in set(resolve_toolset("hermes-cli"))


def test_check_fn_gates_on_reply_gate_mode_tool(monkeypatch):
    """The tool must be schema-invisible under prompt mode (the default —
    upstream's original rationale: 'the agent should not decide on its own
    to fire off cross-platform messages') and schema-visible only when
    reply_gate_mode="tool" is actually configured."""
    from tools.send_message_tool import _check_send_message_tool_gated

    class _FakeCfg:
        def __init__(self, mode):
            self.reply_gate_mode = mode

    import gateway.config as gwcfg

    monkeypatch.setattr(gwcfg, "load_gateway_config", lambda: _FakeCfg("prompt"))
    assert _check_send_message_tool_gated() is False

    monkeypatch.setattr(gwcfg, "load_gateway_config", lambda: _FakeCfg("tool"))
    assert _check_send_message_tool_gated() is True

    monkeypatch.setattr(gwcfg, "load_gateway_config", lambda: _FakeCfg("TOOL"))
    assert _check_send_message_tool_gated() is True, "must be case-insensitive"


def test_check_fn_fails_closed_on_error(monkeypatch):
    """A broken config load must never accidentally EXPOSE the tool — fail
    closed (hidden), matching every other check_fn in this codebase."""
    from tools.send_message_tool import _check_send_message_tool_gated
    import gateway.config as gwcfg

    def _raise():
        raise RuntimeError("boom")

    monkeypatch.setattr(gwcfg, "load_gateway_config", _raise)
    assert _check_send_message_tool_gated() is False


def test_end_to_end_schema_visible_when_tool_gate_configured(monkeypatch):
    """Full path: check_fn True -> get_definitions() actually returns the
    send_message schema, not just registry membership. This is the exact
    gap that let the bug ship silently — the tool LOOKED wired throughout
    gateway/run.py's reply-gate code while the schema itself was empty."""
    from tools.send_message_tool import _check_send_message_tool_gated
    import gateway.config as gwcfg

    class _FakeCfg:
        reply_gate_mode = "tool"

    monkeypatch.setattr(gwcfg, "load_gateway_config", lambda: _FakeCfg())
    assert _check_send_message_tool_gated() is True

    resolved = set(resolve_toolset("hermes-whatsapp"))
    defs = registry.registry.get_definitions(tool_names=resolved)
    names = {d["function"]["name"] for d in defs}
    assert "send_message" in names


def test_end_to_end_schema_hidden_when_tool_gate_off(monkeypatch):
    """Companion: under prompt mode, the schema must NOT include
    send_message — confirms the gate actually suppresses exposure, not just
    that the flag flips."""
    import gateway.config as gwcfg

    class _FakeCfg:
        reply_gate_mode = "prompt"

    monkeypatch.setattr(gwcfg, "load_gateway_config", lambda: _FakeCfg())

    resolved = set(resolve_toolset("hermes-whatsapp"))
    defs = registry.registry.get_definitions(tool_names=resolved)
    names = {d["function"]["name"] for d in defs}
    assert "send_message" not in names
