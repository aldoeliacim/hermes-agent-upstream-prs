"""Tests for the explicit cronjob_silent tool and its scheduler wiring.

This is the STRUCTURAL replacement for regex-based cron silence detection
(see incident cron_4fa958ec7c45, 2026-08-12: a job prompt that instructed
the model to reply with the literal Spanish word "SILENCIO" leaked that
word verbatim to a real WhatsApp DM on every non-ready tick, because no
regex-based marker set could ever know that word was supposed to mean
silence -- string-matching free text is a guess, not an enforceable
contract). cronjob_silent gives the agent a real tool call to make instead,
whose availability and semantics are defined by hermes-agent's own code.

The cron-agent-turn marker lives in the cron-owned module
``cron.agent_turn_context`` (not ``gateway.session_context``): the cron
package owns its own turn state.
"""

import json

from cron.agent_turn_context import (
    reset_cron_agent_turn,
    set_cron_agent_turn,
)
from tools.approval import (
    CronSilenceState,
    get_cron_silence_state,
    note_cron_silence_decided,
    reset_cron_silence_state,
    set_cron_silence_state,
)
from tools.cronjob_silent_tool import cronjob_silent_tool


# ---------------------------------------------------------------------------
# Availability: toolset membership, deliberately NOT check_fn. See the
# module docstring in tools/cronjob_silent_tool.py for the full incident
# that made check_fn the wrong mechanism here (process-wide 30s TTL cache
# racing concurrent cron ticks / non-cron callers).
# ---------------------------------------------------------------------------

def test_registered_with_no_check_fn():
    """Pins the deliberate absence of check_fn -- a future edit re-adding
    ``check_fn=is_cron_agent_turn`` (or any check_fn) on this tool would
    reintroduce the exact process-wide TTL cache collision that made a
    real cron agent report "No such tool exists" despite every other piece
    of the wiring being individually correct (2026-08-12 incident)."""
    from tools.registry import registry

    entry = registry.get_entry("cronjob_silent")
    assert entry is not None
    assert entry.check_fn is None


def test_in_hermes_core_tools_so_tool_search_never_defers_it():
    """cronjob_silent must never be deferred into the tool_search-only
    catalog -- a model can simply fail to search for it (confirmed live:
    a verification run searched with unrelated terms and reported the
    tool didn't exist). Being listed in _HERMES_CORE_TOOLS is what keeps
    tools/tool_search.py::is_deferrable_tool_name() from demoting it."""
    from toolsets import _HERMES_CORE_TOOLS

    assert "cronjob_silent" in _HERMES_CORE_TOOLS


def test_resolve_cron_enabled_toolsets_always_injects_cronjob_silent():
    """The real availability gate: cron/scheduler.py must inject
    "cronjob_silent" into every cron job's resolved toolset list,
    regardless of the job's own per-job enabled_toolsets."""
    from cron.scheduler import _resolve_cron_enabled_toolsets

    job = {"id": "x", "enabled_toolsets": ["terminal"]}
    result = _resolve_cron_enabled_toolsets(job, {})
    assert "cronjob_silent" in result


def test_schema_resolves_end_to_end_for_a_cron_turn():
    """Full-stack regression for the actual incident: with the cron-agent-
    turn flag set and cronjob_silent requested via enabled_toolsets, the
    tool must appear in get_tool_definitions()' output. This is the exact
    call shape cron/scheduler.py uses when constructing AIAgent."""
    import model_tools

    token = set_cron_agent_turn(True)
    try:
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["cronjob_silent"], quiet_mode=True
        )
        names = {d["function"]["name"] for d in defs}
        assert "cronjob_silent" in names
    finally:
        reset_cron_agent_turn(token)


def test_schema_resolves_even_with_a_stale_false_check_fn_cache_entry():
    """THE regression this whole redesign exists to prevent, reproduced
    directly: seed the registry's check_fn cache with a stale False verdict
    for is_cron_agent_turn (simulating a concurrent non-cron caller that
    evaluated get_tool_definitions() moments earlier), then confirm
    cronjob_silent STILL resolves -- because, unlike check_fn-gated tools,
    its visibility never consults that cache at all."""
    import time as _time

    from cron.agent_turn_context import is_cron_agent_turn
    from tools.registry import _check_fn_cache, _check_fn_cache_lock
    import model_tools

    with _check_fn_cache_lock:
        _check_fn_cache[(is_cron_agent_turn, None)] = (_time.monotonic(), False)

    try:
        token = set_cron_agent_turn(True)
        try:
            defs = model_tools.get_tool_definitions(
                enabled_toolsets=["cronjob_silent"], quiet_mode=True
            )
            names = {d["function"]["name"] for d in defs}
            assert "cronjob_silent" in names
        finally:
            reset_cron_agent_turn(token)
    finally:
        with _check_fn_cache_lock:
            _check_fn_cache.pop((is_cron_agent_turn, None), None)


# ---------------------------------------------------------------------------
# tool handler
# ---------------------------------------------------------------------------

def test_tool_refused_outside_cron_turn():
    result = json.loads(cronjob_silent_tool({}))
    assert "error" in result
    assert "cron job" in result["error"]


def test_tool_refused_inside_cron_turn_without_bound_state():
    """is_cron_agent_turn() true but no CronSilenceState bound -- a wiring
    gap, not a caller mistake. Must surface loudly, never silently no-op."""
    token = set_cron_agent_turn(True)
    try:
        result = json.loads(cronjob_silent_tool({}))
        assert "error" in result
        assert "wiring gap" in result["error"]
    finally:
        reset_cron_agent_turn(token)


def test_tool_records_decision_when_properly_bound():
    turn_token = set_cron_agent_turn(True)
    state = CronSilenceState()
    state_token = set_cron_silence_state(state)
    try:
        result = json.loads(cronjob_silent_tool({"reason": "nothing new"}))
        assert result == {"success": True, "decision": "silent"}
        assert state.decided is True
        assert state.reason == "nothing new"
    finally:
        reset_cron_silence_state(state_token)
        reset_cron_agent_turn(turn_token)


def test_tool_reason_defaults_to_empty_string():
    turn_token = set_cron_agent_turn(True)
    state = CronSilenceState()
    state_token = set_cron_silence_state(state)
    try:
        result = json.loads(cronjob_silent_tool({}))
        assert result.get("success") is True
        assert state.decided is True
        assert state.reason == ""
    finally:
        reset_cron_silence_state(state_token)
        reset_cron_agent_turn(turn_token)


# ---------------------------------------------------------------------------
# CronSilenceState / note_cron_silence_decided primitives directly
# ---------------------------------------------------------------------------

def test_note_cron_silence_decided_returns_false_when_unbound():
    assert get_cron_silence_state() is None
    assert note_cron_silence_decided("x") is False


def test_note_cron_silence_decided_mutates_bound_state():
    state = CronSilenceState()
    token = set_cron_silence_state(state)
    try:
        assert note_cron_silence_decided("done for now") is True
        assert state.decided is True
        assert state.reason == "done for now"
    finally:
        reset_cron_silence_state(token)


def test_cron_silence_state_default_is_not_decided():
    state = CronSilenceState()
    assert state.decided is False
    assert state.reason == ""


def test_cron_silence_state_is_context_scoped_not_global():
    """A state bound in one context must not leak into a sibling context that
    never bound one -- mirrors the reasoning behind every other ContextVar
    in tools/approval.py (see TurnDeliveryState's module note)."""
    assert get_cron_silence_state() is None
    state = CronSilenceState()
    token = set_cron_silence_state(state)
    try:
        assert get_cron_silence_state() is state
    finally:
        reset_cron_silence_state(token)
    assert get_cron_silence_state() is None
