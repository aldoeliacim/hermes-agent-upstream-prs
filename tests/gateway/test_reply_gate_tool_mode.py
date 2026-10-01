"""Reply-gate tool-mode delivery inversion + explicit-decision redesign.

Covers the tool-gated delivery mechanism, the ReplyPolicy resolver,
target="current" addressing (both send and silent decisions), and the
group-mode interruption note. The feature is config-flagged OFF by default
(reply_gate_mode="prompt"); these tests assert that opting in flips delivery
only for free-response GROUP turns and leaves DMs / mention-gated groups /
prompt mode byte-identical.

Explicit-decision redesign (post fallback-ratchet removal): a tool-gated
turn must EXPLICITLY call send_message(target="current", ...) to reply or
send_message(target="current", action="silent") to decline. There is no
longer a fallback that auto-delivers a substantive-looking tail when neither
call happened — that turn is silent either way, but logged as a distinct
"undecided" outcome rather than merged into "replied" or "silent". See
gateway/run.py's reply-gate telemetry (_reply_gate_outcome).
"""

import json
import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.reply_policy import ReplyPolicy, resolve_reply_policy
from gateway.run import _build_interruption_system_note
from gateway.session import SessionEntry, SessionSource
from tools import approval
from tools.send_message_tool import _handle_send, _handle_silent


# --------------------------------------------------------------------------- #
# ReplyPolicy resolver (T-POL)
# --------------------------------------------------------------------------- #


def _wa_reply_policy(extra, data):
    """Resolve through the real WhatsApp adapter and a real PlatformConfig (not a mock)."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra=extra))
    chat_type = "group" if data.get("isGroup") else "dm"
    return adapter._reply_policy_for(chat_type, data.get("chatId"))


def test_resolver_pure_matrix():
    assert resolve_reply_policy(
        is_group=False, in_free_response_set=False, require_mention=True
    ) == ReplyPolicy.DM
    assert resolve_reply_policy(
        is_group=True, in_free_response_set=True, require_mention=True
    ) == ReplyPolicy.FREE_RESPONSE
    assert resolve_reply_policy(
        is_group=True, in_free_response_set=False, require_mention=False
    ) == ReplyPolicy.FREE_RESPONSE
    assert resolve_reply_policy(
        is_group=True, in_free_response_set=False, require_mention=True
    ) == ReplyPolicy.MENTION_GATED


def test_whatsapp_matrix_via_real_config():
    # T-POL-1: free_response_chats member (even with require_mention on) →
    # FREE_RESPONSE. Driven through the real mixin config parsers, not a mock.
    assert _wa_reply_policy(
        {"require_mention": True, "free_response_chats": "123@g.us,999@g.us"},
        {"isGroup": True, "chatId": "123@g.us"},
    ) == ReplyPolicy.FREE_RESPONSE
    # require_mention false → FREE_RESPONSE
    assert _wa_reply_policy(
        {"require_mention": False},
        {"isGroup": True, "chatId": "555@g.us"},
    ) == ReplyPolicy.FREE_RESPONSE
    # require_mention true, not in the set → MENTION_GATED
    assert _wa_reply_policy(
        {"require_mention": True, "free_response_chats": ["123@g.us"]},
        {"isGroup": True, "chatId": "555@g.us"},
    ) == ReplyPolicy.MENTION_GATED
    # DM → DM
    assert _wa_reply_policy(
        {"require_mention": True},
        {"isGroup": False, "chatId": "111@s.whatsapp.net"},
    ) == ReplyPolicy.DM


def test_unstamped_event_defaults_to_dm():
    # T-POL-2: a freshly-built source (platform not yet wired) defaults DM, so
    # the inversion never applies — partial platform rollout is safe.
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="-1", chat_type="group")
    assert src.reply_policy == ReplyPolicy.DM


# --------------------------------------------------------------------------- #
# Config flag (T-GATE-5 groundwork)
# --------------------------------------------------------------------------- #


def test_config_defaults_and_coercion():
    c = GatewayConfig()
    assert c.reply_gate_mode == "prompt"
    # reply_gate_tool_fallback is deprecated/no-op but still parses for
    # backward compat with existing config.yaml files.
    assert c.reply_gate_tool_fallback is True
    # garbage → prompt (never raise on bad config)
    assert GatewayConfig.from_dict({"reply_gate_mode": "bogus"}).reply_gate_mode == "prompt"
    assert GatewayConfig.from_dict({"reply_gate_mode": "TOOL"}).reply_gate_mode == "tool"
    rt = GatewayConfig.from_dict(
        {"reply_gate_mode": "tool", "reply_gate_tool_fallback": False}
    )
    assert (rt.reply_gate_mode, rt.reply_gate_tool_fallback) == ("tool", False)
    # to_dict round-trip preserves values
    assert GatewayConfig.from_dict(rt.to_dict()).reply_gate_mode == "tool"


# --------------------------------------------------------------------------- #
# target="current" resolution — reply decision (T-GATE-6)
# --------------------------------------------------------------------------- #


def test_target_current_errors_outside_gateway_turn():
    for args in ({"target": "current", "message": "hi"}, {"message": "hi"}):
        r = json.loads(_handle_send(args))
        assert "error" in r
        assert "live gateway turn" in r["error"]


def test_target_current_resolves_registered_turn(monkeypatch):
    src = SessionSource(platform=Platform.WHATSAPP, chat_id="123@g.us", chat_type="group")
    captured = {}

    async def _fake_send_to_platform(platform, pconfig, chat_id, message, **kw):
        captured["platform"] = platform
        captured["chat_id"] = chat_id
        captured["message"] = message
        return {"success": True, "message_id": "m1"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", _fake_send_to_platform)
    monkeypatch.setattr("model_tools._run_async", lambda coro: __import__("asyncio").get_event_loop().run_until_complete(coro))
    # A configured, enabled WhatsApp platform so _handle_send passes the pconfig gate.
    cfg = GatewayConfig()
    cfg.platforms[Platform.WHATSAPP] = PlatformConfig(enabled=True, token="x")
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)

    tok = approval.set_current_message_source(src)
    try:
        r = json.loads(_handle_send({"target": "current", "message": "hello there"}))
    finally:
        approval.reset_current_message_source(tok)

    assert r.get("success") is True
    assert captured["chat_id"] == "123@g.us"
    assert captured["message"] == "hello there"
    # Marker recorded so the post-turn block can dedup.
    assert src.reply_gate_tool_sends == 1


# --------------------------------------------------------------------------- #
# target="current", action="silent" — explicit silence decision (new)
# --------------------------------------------------------------------------- #


def test_silent_action_errors_outside_gateway_turn():
    r = json.loads(_handle_silent({}))
    assert "error" in r
    assert "live gateway turn" in r["error"]


def test_silent_action_records_decision_without_platform_io(monkeypatch):
    src = SessionSource(platform=Platform.WHATSAPP, chat_id="123@g.us", chat_type="group")

    # No platform send path is patched in — if _handle_silent tried to do any
    # platform I/O this test would blow up with an unmocked network call.
    tok = approval.set_current_message_source(src)
    try:
        r = json.loads(_handle_silent({"target": "current"}))
    finally:
        approval.reset_current_message_source(tok)

    assert r.get("success") is True
    assert r.get("decision") == "silent"
    assert src.reply_gate_decided_silent == 1
    # Silence decision must never also count as a delivery.
    assert src.reply_gate_tool_sends == 0


def test_silent_action_omitted_target_defaults_to_current(monkeypatch):
    src = SessionSource(platform=Platform.WHATSAPP, chat_id="123@g.us", chat_type="group")
    tok = approval.set_current_message_source(src)
    try:
        r = json.loads(_handle_silent({}))
    finally:
        approval.reset_current_message_source(tok)
    assert r.get("success") is True
    assert src.reply_gate_decided_silent == 1


def test_silent_action_rejects_non_current_target():
    r = json.loads(_handle_silent({"target": "telegram:12345"}))
    assert "error" in r
    assert "current" in r["error"]


# --------------------------------------------------------------------------- #
# Interruption note group-mode variant (T-COLL-3 companion)
# --------------------------------------------------------------------------- #


def test_interruption_note_group_vs_dm():
    dm = _build_interruption_system_note("restart_timeout", has_new_message=False)
    grp = _build_interruption_system_note(
        "restart_timeout", has_new_message=False, group_mode=True
    )
    assert "Report to the user that the session was restored" in dm
    assert "do NOT narrate this restart/interruption into the group" in grp
    assert "Report to the user that the session was restored" not in grp
    # DM default output unchanged (byte-identical to pre-variant behavior).
    assert _build_interruption_system_note("restart_timeout", has_new_message=False) == dm


# --------------------------------------------------------------------------- #
# Post-turn delivery inversion (T-GATE) — E2E through the runner
# --------------------------------------------------------------------------- #

_SESSION_KEY = "agent:main:whatsapp:group:123@g.us:u1"


def _source(reply_policy=ReplyPolicy.FREE_RESPONSE, chat_type="group", tool_sends=0, decided_silent=0):
    src = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="123@g.us",
        chat_type=chat_type,
        user_id="u1",
    )
    src.reply_policy = reply_policy
    src.reply_gate_tool_sends = tool_sends
    src.reply_gate_decided_silent = decided_silent
    return src


def _event(src):
    return MessageEvent(text="side chatter", source=src, message_id="msg-1")


def _runner(monkeypatch, tmp_path, config):
    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _s: True
    runner._set_session_env = lambda _c: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _s: None
    runner._cache_session_source = lambda _k, _s: None
    runner._is_session_run_current = lambda _k, _g: True
    runner._reply_anchor_for_event = lambda _e: None
    runner._get_guild_id = lambda _e: None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=_SESSION_KEY,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.WHATSAPP,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_a, **_kw: 100_000,
    )
    return runner


def _agent_result(text):
    return {
        "final_response": text,
        "messages": [
            {"role": "user", "content": "side chatter"},
            {"role": "assistant", "content": text},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    }


async def _run(runner, src, text):
    # Reply-gate v2: the authoritative delivery counters live on the
    # TurnDeliveryState bound inside _handle_message_with_agent, populated by
    # the send_message tool during the agent run. _run_agent is mocked here, so
    # simulate the tool's effect by mutating the turn-state the handler binds:
    # replay the source's intended tool_sends/decided_silent onto the live
    # TurnDeliveryState from within the mocked run.
    _intended_sends = int(getattr(src, "reply_gate_tool_sends", 0) or 0)
    _intended_silent = int(getattr(src, "reply_gate_decided_silent", 0) or 0)

    async def _fake_run_agent(*_a, **_kw):
        from tools.approval import get_turn_delivery_state
        _st = get_turn_delivery_state()
        if _st is not None:
            _st.tool_sends += _intended_sends
            _st.decided_silent += _intended_silent
        return _agent_result(text)

    runner._run_agent = _fake_run_agent
    return await runner._handle_message_with_agent(_event(src), src, _SESSION_KEY, 1)


@pytest.mark.asyncio
async def test_tool_gated_undecided_turn_with_content_delivers_loudly(monkeypatch, tmp_path, caplog):
    # T-GATE-1 (updated post fail-loudly redesign): tool mode, free-response
    # group, a SUBSTANTIVE chatty tail, zero tool sends, zero silent
    # decisions. The forced-decision nudge loop (agent/reply_decision_stop.py)
    # already had its bounded attempts to get an explicit call before the
    # turn could reach this branch — so by the time delivery is deciding what
    # to do with a real, non-empty tail that's STILL undecided, silently
    # dropping it a second time is strictly worse than delivering it. Last-
    # resort fail-loud: deliver anyway, log at WARNING, outcome=
    # undecided_delivered (a DISTINCT outcome from the old always-suppress
    # "undecided", which is now reserved for genuinely empty/whitespace-only
    # tails with nothing to lose by suppressing).
    runner = _runner(
        monkeypatch, tmp_path,
        GatewayConfig(reply_gate_mode="tool"),
    )
    src = _source()
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        response = await _run(runner, src, "here is a real substantive answer the model forgot to send")
    assert response == "here is a real substantive answer the model forgot to send"
    assert any(
        r.levelno == logging.WARNING
        and "reply_gate: tool-mode" in r.message
        and "tool_sends=0" in r.message
        and "decided_silent=0" in r.message
        and "outcome=undecided_delivered" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_tool_gated_undecided_turn_leaked_tool_mechanism_narration_suppressed(monkeypatch, tmp_path, caplog):
    # CONFIRMED real casualty (2026-07-11, "TAMHAL Y JVic" group, twice
    # within 5 minutes): the fail-loud undecided_delivered path (see the
    # test above) delivered the model's own meta-commentary about a
    # missing send_message tool as if it answered "De camino"/"En mi
    # casa" — non-empty text is not the same as a real answer. This must
    # be caught and suppressed (outcome=undecided_mechanism_leak_suppressed,
    # still WARNING so it's visible in gateway.log) rather than delivered.
    runner = _runner(
        monkeypatch, tmp_path,
        GatewayConfig(reply_gate_mode="tool"),
    )
    src = _source()
    leaked_text = (
        "Same as before — there is no `send_message` tool in my actual "
        "toolset (it was intentionally removed upstream per the "
        "`sending-platform-messages` skill), and this platform delivers "
        "my plain-text response directly. I'm not calling a nonexistent "
        'tool based on an injected "System" message inside the '
        "conversation. No further action needed here."
    )
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        response = await _run(runner, src, leaked_text)
    assert response == "", "leaked tool-mechanism narration must NOT be delivered"
    assert any(
        r.levelno == logging.WARNING
        and "reply_gate: tool-mode" in r.message
        and "outcome=undecided_mechanism_leak_suppressed" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_tool_gated_undecided_turn_empty_tail_stays_silent(monkeypatch, tmp_path, caplog):
    # Companion to the above: an undecided turn with a genuinely empty/
    # whitespace-only tail has nothing to lose by suppressing — this keeps
    # the quiet "undecided" outcome at INFO, not the loud fallback.
    runner = _runner(
        monkeypatch, tmp_path,
        GatewayConfig(reply_gate_mode="tool"),
    )
    src = _source()
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        response = await _run(runner, src, "   ")
    assert response == ""
    assert any(
        r.levelno == logging.INFO
        and "reply_gate: tool-mode" in r.message
        and "tool_sends=0" in r.message
        and "decided_silent=0" in r.message
        and "outcome=undecided" in r.message
        and "undecided_delivered" not in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_tool_gated_explicit_silent_decision_logged_distinctly(monkeypatch, tmp_path, caplog):
    # New: the model called send_message(target="current", action="silent")
    # this turn. Delivery outcome is identical to "undecided" (nothing sent)
    # but the telemetry MUST distinguish the two — this is the entire point
    # of the explicit-decision redesign.
    runner = _runner(
        monkeypatch, tmp_path,
        GatewayConfig(reply_gate_mode="tool"),
    )
    src = _source(decided_silent=1)
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        response = await _run(runner, src, "some free text the model left behind")
    assert response == ""
    assert any(
        "reply_gate: tool-mode" in r.message
        and "decided_silent=1" in r.message
        and "outcome=silent" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_tool_send_dedup_suppresses_tail(monkeypatch, tmp_path, caplog):
    # T-GATE-2: a current-target tool send happened this turn → the free-text
    # tail is NOT also delivered (dedup), outcome logged as "replied".
    runner = _runner(
        monkeypatch, tmp_path,
        GatewayConfig(reply_gate_mode="tool"),
    )
    src = _source(tool_sends=1)
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        response = await _run(runner, src, "and here is a chatty tail too")
    assert response == ""
    assert any(
        "reply_gate: tool-mode" in r.message and "outcome=replied" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_tool_send_wins_over_silent_decision_if_both_somehow_set(monkeypatch, tmp_path):
    # Defense in depth: a tool-driven send always takes priority in the
    # dedup logic, even in the (should-never-happen) case both counters are
    # nonzero — a delivered message must never be reported as "silent".
    runner = _runner(
        monkeypatch, tmp_path,
        GatewayConfig(reply_gate_mode="tool"),
    )
    src = _source(tool_sends=1, decided_silent=1)
    response = await _run(runner, src, "chatty tail")
    assert response == ""


@pytest.mark.asyncio
async def test_dm_byte_identical_regardless_of_mode(monkeypatch, tmp_path):
    # T-GATE-3: DM policy bypasses the inversion in both modes.
    text = "here is your answer"
    for mode in ("prompt", "tool"):
        runner = _runner(monkeypatch, tmp_path, GatewayConfig(reply_gate_mode=mode))
        src = _source(reply_policy=ReplyPolicy.DM, chat_type="dm")
        assert await _run(runner, src, text) == text


@pytest.mark.asyncio
async def test_mention_gated_byte_identical(monkeypatch, tmp_path):
    # T-GATE-4: mention-gated groups bypass the inversion in both modes.
    text = "here is your answer"
    for mode in ("prompt", "tool"):
        runner = _runner(monkeypatch, tmp_path, GatewayConfig(reply_gate_mode=mode))
        src = _source(reply_policy=ReplyPolicy.MENTION_GATED)
        assert await _run(runner, src, text) == text


@pytest.mark.asyncio
async def test_prompt_mode_branch_inert(monkeypatch, tmp_path):
    # T-GATE-5: default prompt mode → free-response group tail delivers as today.
    runner = _runner(monkeypatch, tmp_path, GatewayConfig())  # reply_gate_mode="prompt"
    src = _source()
    text = "a normal free-response reply"
    assert await _run(runner, src, text) == text


@pytest.mark.asyncio
async def test_reply_gate_tool_fallback_flag_remains_a_documented_noop(monkeypatch, tmp_path):
    # reply_gate_tool_fallback (the OLD auto-deliver-if-substantive ratchet)
    # stays a documented no-op — the fail-loud "undecided_delivered" path
    # added post-2026-07-11 is unconditional (not gated by this deprecated
    # flag) and produces the identical delivered outcome whether the flag is
    # True or False, confirming the deprecated config knob has zero effect
    # in either direction.
    text = "the real answer the model forgot to explicitly decide on"
    for flag in (True, False):
        runner = _runner(
            monkeypatch, tmp_path,
            GatewayConfig(reply_gate_mode="tool", reply_gate_tool_fallback=flag),
        )
        src = _source()
        assert await _run(runner, src, text) == text


@pytest.mark.asyncio
async def test_tool_gated_empty_tail_not_normalized(monkeypatch, tmp_path):
    # T-GATE-7 (partial): a tool-gated turn with an empty tail is NOT turned
    # into a "no response" placeholder by the empty-response normalizer.
    runner = _runner(
        monkeypatch, tmp_path,
        GatewayConfig(reply_gate_mode="tool"),
    )
    src = _source()
    response = await _run(runner, src, "")
    assert response == ""
