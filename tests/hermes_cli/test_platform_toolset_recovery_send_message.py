"""Regression coverage for send_message/whatsapp_action visibility under an
EXPLICIT platform_toolsets config -- the real production code path that the
2026-07-11 fix (dc4c07ec2) never actually exercised.

CONFIRMED gap (2026-07-12, "Quetzalogic - Hal" WhatsApp group, second
incident within 24h): dc4c07ec2 restored send_message's
_HERMES_CORE_TOOLS membership and verified it by resolving the bare
"hermes-whatsapp" composite name directly (resolve_toolset("hermes-whatsapp")).
That is NOT the code path a real deployment with a saved
platform_toolsets.whatsapp list uses -- gateway/run.py calls
hermes_cli.tools_config._get_platform_tools(config, "whatsapp"), which for
an EXPLICIT toolset list only recovers a non-configurable _HERMES_CORE_TOOLS
member if it ALSO has its own standalone (non-"hermes-*", no "includes")
TOOLSETS entry. send_message and whatsapp_action had NEITHER -- upstream's
c6c8abbad (#47856) deleted the old standalone "messaging" entry along with
the tool's registration, and it was never restored. So send_message
silently never reached the schema even after dc4c07ec2 "fixed" it: the
model, mid-turn, again correctly reported "Tool 'send_message' does not
exist" -- a second incident, a week after the first, on a config path the
prior fix's own verification never touched. whatsapp_action had the exact
same latent gap, independently confirmed missing from the live schema the
same day, despite a skill claiming it worked.

Fixed by adding two standalone TOOLSETS entries ("messaging",
"whatsapp_action_recovery") so hermes_cli.tools_config's non-configurable-
toolset recovery loop can restore both tools onto a platform with an
explicit saved toolset list. These tests exercise the REAL
_get_platform_tools -> get_tool_definitions path against a config shaped
exactly like this deployment's (an explicit platform_toolsets.whatsapp
list saved months ago, well before either tool was restored) -- not a
shortcut resolve_toolset() call on the composite name.

check_fn gating: tests/conftest.py isolates HERMES_HOME to a per-test
tempdir, so send_message's check_fn (reads reply_gate_mode from real
config.yaml) and whatsapp_action's check_fn (reads platforms.whatsapp from
real config.yaml) both correctly evaluate False against that empty sandbox.
Monkeypatch gateway.config.load_gateway_config the same way
test_reply_gate_tool_mode.py / test_send_message_tool_registration.py
already do, rather than relying on the real ~/.hermes/config.yaml.
"""

from __future__ import annotations

import pytest

import tools.registry as registry
from hermes_cli.tools_config import _get_platform_tools
from model_tools import get_tool_definitions


@pytest.fixture(autouse=True)
def _clean_check_fn_cache():
    """Same discipline as test_send_message_tool_registration.py: check_fn
    results are TTL-cached process-wide, so monkeypatching config mid-run
    needs an invalidate before AND after to avoid order-dependent leakage."""
    registry.invalidate_check_fn_cache()
    yield
    registry.invalidate_check_fn_cache()


@pytest.fixture
def _tool_gated_and_whatsapp_configured(monkeypatch):
    """Configure BOTH gates true: reply_gate_mode='tool' (send_message's
    check_fn) and a configured WhatsApp platform (whatsapp_action's
    check_fn) -- mirrors this deployment's real live config.

    send_message_tool.py resolves gateway.config.load_gateway_config lazily
    (imported inside the function body) so patching the module attribute is
    sufficient. whatsapp_action_tool.py binds load_gateway_config via a
    top-level `from gateway.config import ... load_gateway_config` -- a
    name bound at import time -- so it must be patched on
    tools.whatsapp_action_tool directly or the module-level patch above is
    silently a no-op for that check_fn."""
    import gateway.config as gwcfg
    import tools.whatsapp_action_tool as wat
    from gateway.config import GatewayConfig, Platform, PlatformConfig

    cfg = GatewayConfig()
    cfg.reply_gate_mode = "tool"
    cfg.platforms[Platform.WHATSAPP] = PlatformConfig(enabled=True, token="x")
    monkeypatch.setattr(gwcfg, "load_gateway_config", lambda: cfg)
    monkeypatch.setattr(wat, "load_gateway_config", lambda: cfg)
    return cfg


# Mirrors this deployment's real, long-saved platform_toolsets.whatsapp list
# (config.yaml) -- an EXPLICIT list of individual toolset keys, NOT the bare
# "hermes-whatsapp" composite name. This exact shape is what silently hid
# send_message and whatsapp_action even after dc4c07ec2.
_EXPLICIT_WHATSAPP_CONFIG = {
    "platform_toolsets": {
        "whatsapp": [
            "browser", "clarify", "code_execution", "cronjob", "delegation",
            "file", "homeassistant", "image_gen", "memory", "session_search",
        ],
    },
}


def test_send_message_recovered_under_explicit_platform_toolsets_config():
    enabled = _get_platform_tools(_EXPLICIT_WHATSAPP_CONFIG, "whatsapp")
    assert "messaging" in enabled, (
        "send_message's standalone 'messaging' toolset entry was not "
        "recovered onto an explicit platform_toolsets.whatsapp list -- "
        "this is the exact 2026-07-12 regression."
    )


def test_whatsapp_action_recovered_under_explicit_platform_toolsets_config():
    enabled = _get_platform_tools(_EXPLICIT_WHATSAPP_CONFIG, "whatsapp")
    assert "whatsapp_action_recovery" in enabled, (
        "whatsapp_action's standalone toolset entry was not recovered onto "
        "an explicit platform_toolsets.whatsapp list -- the same latent gap "
        "as send_message, independently confirmed 2026-07-12."
    )


def test_send_message_reaches_live_schema_under_explicit_config(
    _tool_gated_and_whatsapp_configured,
):
    """End-to-end: config -> _get_platform_tools -> get_tool_definitions ->
    the actual schema the model receives. This is the real path
    gateway/run.py drives; resolve_toolset("hermes-whatsapp") alone is not
    sufficient proof (that was the previous fix's blind spot)."""
    enabled = sorted(_get_platform_tools(_EXPLICIT_WHATSAPP_CONFIG, "whatsapp"))
    defs = get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True)
    names = {d["function"]["name"] for d in defs}
    assert "send_message" in names


def test_whatsapp_action_reaches_live_schema_under_explicit_config(
    _tool_gated_and_whatsapp_configured,
):
    enabled = sorted(_get_platform_tools(_EXPLICIT_WHATSAPP_CONFIG, "whatsapp"))
    defs = get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True)
    names = {d["function"]["name"] for d in defs}
    assert "whatsapp_action" in names


def test_default_whatsapp_composite_still_includes_both_tools(
    _tool_gated_and_whatsapp_configured,
):
    """Sanity: the simpler no-explicit-config case (bare 'hermes-whatsapp'
    composite) must also still work -- this was the ONLY path the previous
    fix verified, so it must not regress."""
    enabled = _get_platform_tools({}, "whatsapp")
    defs = get_tool_definitions(enabled_toolsets=sorted(enabled), quiet_mode=True)
    names = {d["function"]["name"] for d in defs}
    assert "send_message" in names
    assert "whatsapp_action" in names


def test_messaging_and_whatsapp_action_recovery_not_user_facing_toggles():
    """The two new standalone entries are pure recovery plumbing for
    check_fn-gated core tools -- they must NOT appear as configurable
    checkboxes in `hermes tools` (that would be a real UX regression,
    offering a toggle for a tool the user cannot independently disable)."""
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS

    configurable_keys = {key for key, _, _ in CONFIGURABLE_TOOLSETS}
    assert "messaging" not in configurable_keys
    assert "whatsapp_action_recovery" not in configurable_keys
