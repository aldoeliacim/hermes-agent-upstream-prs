"""Every ``GatewayConfig`` field must survive the REAL loader, nested form included.

Background: ``load_yaml_layer`` builds ``gw_data`` FLAT and never forwards the
yaml ``gateway:`` section, so any key missing from ``config_loader._TOPLEVEL_BRIDGE``
was silently discarded on real startup — the operator's value vanished and
``from_dict`` applied its default. That shipped once already
(``gateway.reply_gate_mode: tool`` resolved to ``"prompt"``), so the table is now
DERIVED from ``dataclasses.fields(GatewayConfig)`` and these tests pin the invariant
that makes the next field impossible to lose.
"""

import dataclasses

import pytest

from gateway.config import GatewayConfig, load_gateway_config
from gateway.config_loader import _BRIDGE_EXCLUDE, _TOPLEVEL_BRIDGE


def _write_config(body):
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(body, encoding="utf-8")


def _bridged_keys():
    return {row[0] for row in _TOPLEVEL_BRIDGE}


# --- the invariant ------------------------------------------------------------


def test_every_gateway_config_field_is_bridged():
    """No field may be reachable only through a hand-maintained table row.

    A field is accounted for if it is bridged, or if it is one of the documented
    exclusions (merged by another loader) — which test_exclusions_are_justified
    pins individually.
    """
    fields = {f.name for f in dataclasses.fields(GatewayConfig)}
    unaccounted = fields - _bridged_keys() - _BRIDGE_EXCLUDE

    assert unaccounted == set(), f"GatewayConfig fields dropped by the loader: {sorted(unaccounted)}"


def test_exclusions_are_justified():
    """``platforms``/``stt`` are excluded on purpose — prove they still land."""
    assert _BRIDGE_EXCLUDE == {"platforms", "stt"}

    _write_config("gateway:\n  stt:\n    enabled: false\n")
    # stt_enabled is its own field and IS bridged; the nested stt block is read
    # by from_dict via stt_setting(), which the bridge must not shadow.
    assert load_gateway_config().stt_enabled is False


def test_derived_rows_use_presence_mode_matching_from_dict():
    """Derived rows mirror from_dict's ``pick()`` precedence instead of inventing one.

    Rows in ``_BRIDGE_OVERRIDES`` deliberately keep a non-default mode; every other
    row must be plain "presence" (top-level key wins, else the nested form).
    """
    from gateway.config_loader import _BRIDGE_OVERRIDES

    overridden = {row[0] for row in _BRIDGE_OVERRIDES}
    for yaml_key, gw_key, mode, _accept, _transform in _TOPLEVEL_BRIDGE:
        if yaml_key in overridden or yaml_key in _BRIDGE_EXCLUDE:
            continue
        assert (gw_key, mode) == (yaml_key, "presence"), (
            f"derived row {yaml_key} must use presence mode, got {mode!r}"
        )


# --- end-to-end: nested form actually reaches the config ---------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("reply_gate_mode", "tool"),
        ("session_store_max_age_days", 7),
        ("stt_enabled", False),
        ("loop_watchdog_max_strikes", 9),
        ("thread_sessions_per_user", True),
        ("systemd_watchdog_seconds", 45),
        ("max_concurrent_sessions", 4),
    ],
)
def test_nested_field_reaches_runtime(field, value):
    """The documented ``gateway:`` block form must survive the production loader."""
    _write_config(f"gateway:\n  {field}: {str(value).lower() if isinstance(value, bool) else value}\n")
    assert getattr(load_gateway_config(), field) == value


def test_toplevel_still_wins_over_nested():
    """Regression guard: deriving the table must not change documented precedence."""
    _write_config("gateway:\n  session_store_max_age_days: 7\nsession_store_max_age_days: 21\n")
    assert load_gateway_config().session_store_max_age_days == 21


def test_unrelated_platform_block_still_merges():
    """Platforms are merged by merge_platform_sections; deriving must not shadow it."""
    _write_config("gateway:\n  telegram:\n    enabled: true\n")
    cfg = load_gateway_config()
    assert cfg.platforms, "platforms block stopped merging"