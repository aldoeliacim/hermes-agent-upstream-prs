"""profiles.configure ``enabled_toolsets`` must pin the key the runtime reads.

Invariant: whatever ``_save_toolset_pin`` writes, ``_load_enabled_toolsets`` (via ``_get_platform_tools(cfg, "cli")``)
reads back as the enabled set, and ``_describe_toolsets`` reports the same pin. Before the fix the writer used a
``tools.enabled_toolsets`` key that nothing reads, so a pin of ``[web]`` still produced the full default set.

A pin is not absolute: ``_recover_platform_native_toolsets`` re-adds non-configurable
platform-native toolsets (the local ``messaging``/``reactions``/``whatsapp_action_*_recovery``
entries) on top of an explicit list, so messaging survives a pin that omits it. That is the
rule these assertions subtract — the invariant under test is that the pin is READ BACK and
round-trips, not that it silences platform-native capability.
"""

import pytest

from tui_gateway.methods_profiles import _describe_toolsets, _save_toolset_pin


def _native_recovered(resolved: set[str], pinned: set[str] | None) -> set[str]:
    """The non-configurable toolsets recovery re-adds on top of the pin."""
    assert pinned is not None
    return resolved - pinned


def _pin_and_resolve(cfg: dict, names: list[str]) -> tuple[set[str], set[str] | None]:
    from hermes_cli.tools_config import _get_platform_tools

    _save_toolset_pin(cfg, names, save_config=lambda _c: None)
    resolved = set(_get_platform_tools(cfg, "cli", include_default_mcp_servers=False))
    _toolsets, pinned = _describe_toolsets(cfg)
    return resolved, pinned


def test_toolset_pin_is_what_the_runtime_reads():
    cfg: dict = {}
    resolved, pinned = _pin_and_resolve(cfg, ["web"])
    assert pinned == {"web"}
    # The pin is honoured, and the only toolsets beyond it are platform-native
    # recoveries — nothing from the platform default leaked back in.
    assert pinned <= resolved
    assert _native_recovered(resolved, pinned) <= {
        "messaging", "reactions", "whatsapp_action_recovery", "whatsapp_action_admin_recovery",
    }
    assert "file" not in resolved and "terminal" not in resolved, "default composite leaked past the pin"
    assert "enabled_toolsets" not in (cfg.get("tools") or {}), "writer must not leave the phantom key behind"


@pytest.mark.parametrize("second", [["file", "terminal"], []])
def test_toolset_pin_round_trips_a_then_b_then_a(second):
    cfg: dict = {}
    first, first_pinned = _pin_and_resolve(cfg, ["web"])
    _pin_and_resolve(cfg, second)
    again, pinned = _pin_and_resolve(cfg, ["web"])
    assert first_pinned == pinned == {"web"}
    # A->B->A must return to the same resolved set: no bleed from the middle pin.
    assert first == again


def test_empty_pin_clears_the_pin_and_falls_back_to_the_platform_default():
    from hermes_cli.tools_config import _get_platform_tools

    cfg: dict = {}
    _save_toolset_pin(cfg, ["web"], save_config=lambda _c: None)
    _save_toolset_pin(cfg, [], save_config=lambda _c: None)
    _toolsets, pinned = _describe_toolsets(cfg)
    assert pinned is None
    assert set(_get_platform_tools(cfg, "cli", include_default_mcp_servers=False)) != {"web"}
