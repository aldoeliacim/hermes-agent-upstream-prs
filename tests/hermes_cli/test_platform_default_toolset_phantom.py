"""A platform's default toolset must actually yield tools.

``_platform_default_toolset`` derives ``hermes-<platform>`` for any platform absent from
``PLATFORMS``. That derived name is only a NAME: ``Platform`` enum values such as
``relay`` and ``msgraph_webhook`` have no matching registry entry, so it resolves to
nothing.

The failure mode is silent and total. Callers hand the result straight to the agent as
``enabled_toolsets`` — a LIST, never ``None`` — and ``model_tools._select_tool_names``
treats ``None`` as "every toolset" but an empty resolution as "no tools at all". A phantom
default therefore yields a non-empty set contributing zero tools, and the session runs
with no tool surface and nothing logged.

These tests pin the observable behavior through ``_get_platform_tools`` (the seam every
caller goes through), not the internal derivation.
"""

from __future__ import annotations

import pytest

from hermes_cli.tools_config import _get_platform_tools
from toolsets import TOOLSETS


def _platform_config_key(platform_value: str) -> str:
    """Mirror gateway/run.py::_platform_config_key (LOCAL is spelled 'cli' in config)."""
    return "cli" if platform_value == "local" else platform_value


def _resolvable(names) -> list:
    return [name for name in names if name in TOOLSETS]


def test_every_platform_yields_a_working_tool_surface():
    """No Platform may end a turn able to call nothing."""
    from gateway.platforms.base import Platform

    broken = []
    for member in Platform:
        key = _platform_config_key(member.value)
        if not _resolvable(_get_platform_tools({}, key)):
            broken.append(f"{member.name} (key={key!r})")

    assert not broken, (
        "platform(s) whose toolsets all fail to resolve; each such session runs with "
        f"ZERO tools and no warning: {', '.join(broken)}"
    )


@pytest.mark.parametrize("platform", ["relay", "msgraph_webhook"])
def test_phantom_platform_default_still_yields_working_tools(platform, caplog):
    """The two known offenders keep a real tool surface instead of an empty one."""
    with caplog.at_level("WARNING"):
        resolved = _get_platform_tools({}, platform)

    assert _resolvable(resolved), (
        f"{platform} resolved only to names absent from the toolset registry "
        f"({sorted(resolved)}) — the session would run with zero tools"
    )
    assert any(
        record.levelname == "WARNING" and platform in record.getMessage()
        for record in caplog.records
    ), f"the phantom default for {platform!r} was not logged"


def test_guard_does_not_flatten_a_registered_platform():
    """A platform that resolves keeps its OWN default — the guard is not a blanket rewrite."""
    from hermes_cli.tools_config import _is_registered_toolset, _platform_default_toolset

    for key in ("telegram", "whatsapp", "cron", "cli"):
        assert _is_registered_toolset(_platform_default_toolset(key)), (
            f"{key} has a registered default and must not be treated as a phantom"
        )