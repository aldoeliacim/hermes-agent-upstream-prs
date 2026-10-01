"""E2E: the non-owner rails are routed by READER, through a real GatewayRunner
inbound turn and a wire-capturing adapter.

  1. Final reply on a non-owner DM: the semantic judge runs BEFORE the send,
     for every final, regex or not. A both-LEAK verdict reroutes to the owner
     home; the contact's chat receives nothing.
  2. Platform notice rail (subagent timeout, credit band, no-home nudge): a
     non-owner reader never sees it; the owner home gets it, tagged.

The contact's stored name is a pushname ("Hey!"), so the name regex is blind by
construction, and the leak text is prose with no checkmark bullets, so the
scaffold regex is blind too. Only the judge sees it. The judge lane is mocked
with a verdict oracle (network-free).

Ported from the live layer with neutral placeholders. Entry point is
``_handle_message_with_agent``, the real inbound path where the gate lives, with
``_run_agent`` stubbed to return the scripted final.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import sys

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource

OWNER_HOME = "15550000000@s.whatsapp.net"
CONTACT = "100000000000001@lid"
REQUEST = "por favor agrega la de la película de 1999 en audio latino"

LEAK = (
    "The subagent timed out but left the pipeline mid-flight, not broken: video torrent at 20% "
    "and the Latino donor (voe, verified by frame + ASR as Latin American Spanish) at 40%. "
    "I launched the finisher script on the media host that waits for both, verifies, syncs, muxes, "
    "and promotes. No message to the contact now — he already got \"te aviso cuando quede\"."
)
REPLY = "Va 👍 Ya me puse con la película en latino. Te aviso en cuanto quede."


class WireAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.WHATSAPP)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id=f"m-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    def to(self, chat_id):
        return [m["content"] for m in self.sent if m["chat_id"] == chat_id]


def _source(chat_id, name):
    return SessionSource(platform=Platform.WHATSAPP, chat_id=chat_id, user_id=chat_id, user_name=name, chat_type="dm")


def _event(source, text=REQUEST):
    return MessageEvent(text=text, source=source, message_id="msg-1")


def _runner(monkeypatch, tmp_path, adapter, *, final):
    cfg = GatewayConfig(platforms={
        Platform.WHATSAPP: PlatformConfig(
            enabled=True, token="***",
            home_channel=HomeChannel(platform=Platform.WHATSAPP, chat_id=OWNER_HOME, name="Home"),
        ),
    })
    runner = gateway_run.GatewayRunner(cfg)
    runner.adapters = {Platform.WHATSAPP: adapter}
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
    runner._should_send_voice_reply = lambda *_a, **_k: False
    runner.hooks = MagicMock(); runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="k", session_id="sess", created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.WHATSAPP, chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = []
    runner._run_agent = AsyncMock(return_value={
        "final_response": final,
        "messages": [{"role": "user", "content": REQUEST}, {"role": "assistant", "content": final}],
        "tools": [], "history_offset": 0, "last_prompt_tokens": 0, "api_calls": 1, "failed": False,
    })
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_a, **_k: 100_000)
    return runner


def _oracle(monkeypatch, verdict):
    import gateway.delivery_judge as dj
    calls = []
    def fake(messages, **_):
        calls.append(messages[1]["content"]); return verdict
    monkeypatch.setattr(dj, "ask_lanes", fake)
    return calls


async def _turn(monkeypatch, tmp_path, *, text, verdict, chat_id=CONTACT, name="Hey!"):
    adapter = WireAdapter()
    runner = _runner(monkeypatch, tmp_path, adapter, final=text)
    calls = _oracle(monkeypatch, verdict)
    src = _source(chat_id, name)
    response = await runner._handle_message_with_agent(_event(src), src, f"agent:main:whatsapp:dm:{chat_id}", 1)
    return response, adapter, calls


@pytest.mark.asyncio
async def test_judge_runs_on_every_non_owner_final_and_holds_a_unanimous_leak(monkeypatch, tmp_path):
    response, adapter, calls = await _turn(monkeypatch, tmp_path, text=LEAK, verdict="LEAK")
    # First round asks both framings; an UNCORROBORATED unanimous LEAK (both_leak,
    # regex=none) is then RE-JUDGED to separate a real leak from LLM noise. A stable
    # LEAK (this fixture always holds) is confirmed and stays held — so we see the
    # initial 2 framings plus re-sample rounds (>2 total), never fewer than 2.
    assert len(calls) >= 2, "both framings must be asked, regex or not"
    assert all(REQUEST in c for c in calls), "the judge must see what the contact asked"
    assert not response, f"held reply surfaced as a final: {response!r}"
    assert adapter.to(CONTACT) == [], f"leak reached the contact: {adapter.to(CONTACT)}"
    home = "\n".join(adapter.to(OWNER_HOME))
    assert "subagent timed out" in home.lower() and "Hey!" in home, home


@pytest.mark.asyncio
async def test_judge_lets_a_real_reply_through_untouched(monkeypatch, tmp_path):
    response, adapter, calls = await _turn(monkeypatch, tmp_path, text=REPLY, verdict="DELIVER")
    assert len(calls) == 2, "the judge runs on warm replies too — no regex fast-path"
    assert response == REPLY
    assert adapter.to(OWNER_HOME) == [], "a clean both-DELIVER is silent for the owner"


@pytest.mark.asyncio
async def test_no_lane_answering_delivers_and_alerts_infra_not_leak(monkeypatch, tmp_path):
    # Lanes down + regex blind = NO signal. Fail-open still delivers the reply,
    # but the owner gets a throttled INFRA alert (guard degraded) — NOT the
    # per-message "possible leak" banner with the recipient's reply pasted in.
    # Alert on signal, not on outage.
    response, adapter, _ = await _turn(monkeypatch, tmp_path, text=LEAK, verdict=None)
    assert response == LEAK, "fail-open: a silent judge never withholds a reply"
    home = "\n".join(adapter.to(OWNER_HOME))
    # Infra alert fired...
    assert "degraded" in home.lower() and "no_lane_answered" in home, home
    # ...but it is body-free: the delivered reply must NOT be echoed to the owner,
    # and it must not be framed as a possible leak / ground-truth case.
    assert "ground-truth" not in home.lower(), home
    assert LEAK not in home, "infra alert must not paste the delivered reply body"


@pytest.mark.asyncio
async def test_owner_dm_final_is_never_judged(monkeypatch, tmp_path):
    response, adapter, calls = await _turn(monkeypatch, tmp_path, text=LEAK, verdict="LEAK", chat_id=OWNER_HOME, name="Owner")
    assert calls == [] and response == LEAK


# --------------------------------------------------------------------------- notice rail
@pytest.mark.asyncio
async def test_operator_notice_on_non_owner_dm_goes_to_owner_home(monkeypatch, tmp_path):
    adapter = WireAdapter()
    runner = _runner(monkeypatch, tmp_path, adapter, final="")
    notice = '⚠️ Subagent timed out — "Add the movie (1999, TMDB 10001…": Timed out after 1200.01s (after 1200s)'
    await runner._deliver_platform_notice(_source(CONTACT, "Hey!"), notice)
    assert adapter.to(CONTACT) == [], adapter.sent
    home = "\n".join(adapter.to(OWNER_HOME))
    assert "Subagent timed out" in home and "Hey!" in home, home


@pytest.mark.asyncio
async def test_operator_notice_on_owner_dm_stays_in_place(monkeypatch, tmp_path):
    adapter = WireAdapter()
    runner = _runner(monkeypatch, tmp_path, adapter, final="")
    await runner._deliver_platform_notice(_source(OWNER_HOME, "Owner"), "⚠️ Subagent timed out — x")
    assert adapter.to(OWNER_HOME) == ["⚠️ Subagent timed out — x"]



def _group_turn(monkeypatch, tmp_path, *, text, chat_id="120000000000000001@g.us", trusted=()):
    """Same real inbound path as _turn, but the destination is a shared group."""
    import gateway.run_turn as rt

    monkeypatch.setattr(rt, "_owner_trusted_chat_ids", lambda: frozenset(trusted))

    def _group_source(cid, name, _orig=_source):
        src = _orig(cid, name)
        src.chat_type = "group"
        return src

    monkeypatch.setattr(sys.modules[__name__], "_source", _group_source)
    return _turn(monkeypatch, tmp_path, text=text, verdict="DELIVER", chat_id=chat_id, name="Robin")


GROUP_BACKEND_LEAK = (
    "El servidor está sano. El problema: tu app está probando `localhost:8096` y "
    "`:8920`, que son puertos internos. Jellyfin 10.11.11 responde bien."
)


@pytest.mark.asyncio
async def test_group_final_with_a_deterministic_leak_is_held_and_rerouted(monkeypatch, tmp_path):
    """A shared group is an audience beyond the sender: a deterministic privacy class
    (internal endpoints + backend version) is held and rerouted to the owner, and the
    semantic judge lanes are NOT consulted (register judging is a DM concept)."""
    response, adapter, calls = await _group_turn(monkeypatch, tmp_path, text=GROUP_BACKEND_LEAK)
    assert GROUP_BACKEND_LEAK not in (response or ""), "the leak reached the group"
    home = "\n".join(adapter.to(OWNER_HOME))
    assert "localhost:8096" in home, "the held final must be rerouted to the owner"
    assert calls == [], "group prose must not go through the DM register judge"


@pytest.mark.asyncio
async def test_owner_trusted_group_is_an_entitled_audience(monkeypatch, tmp_path):
    """privacy.owner_trusted_chats exempts a declared group from the leak guard."""
    gid = "120000000000000002@g.us"
    response, adapter, _calls = await _group_turn(
        monkeypatch, tmp_path, text=GROUP_BACKEND_LEAK, chat_id=gid, trusted=(gid,))
    assert response == GROUP_BACKEND_LEAK
    assert adapter.to(OWNER_HOME) == []


@pytest.mark.asyncio
async def test_clean_group_final_is_delivered_untouched(monkeypatch, tmp_path):
    response, adapter, calls = await _group_turn(monkeypatch, tmp_path, text=REPLY)
    assert response == REPLY
    assert adapter.to(OWNER_HOME) == [] and calls == []



@pytest.mark.asyncio
async def test_owner_own_dm_is_never_content_filtered(monkeypatch, tmp_path):
    """The owner is entitled to mechanism/status/backend detail about their own
    infrastructure: an owner DM final with backend substance reaches the owner verbatim.
    An audience-blind filter would silently drop the owner's own answers and hide the
    drop from them."""
    response, adapter, calls = await _turn(
        monkeypatch, tmp_path, text=GROUP_BACKEND_LEAK, verdict="LEAK",
        chat_id=OWNER_HOME, name="Owner")
    assert response == GROUP_BACKEND_LEAK, "the owner's own answer was suppressed"
    assert calls == [], "the owner's own DM is never judged"



async def _silence_turn(monkeypatch, tmp_path, *, chat_type, reply_gate_mode, reply_policy):
    """A bare silence marker from the model, through the real inbound path."""
    from gateway.reply_policy import ReplyPolicy
    from gateway.session import SessionSource

    gid = "120000000000000001@g.us"

    def _src(cid, name):
        return SessionSource(
            platform=Platform.WHATSAPP, chat_id=cid, user_id="80000000000002@lid", user_name=name,
            chat_type=chat_type, reply_policy=getattr(ReplyPolicy, reply_policy) if reply_policy else None,
        )

    monkeypatch.setattr(sys.modules[__name__], "_source", _src)
    adapter = WireAdapter()
    runner = _runner(monkeypatch, tmp_path, adapter, final="NO_REPLY")
    runner.config.reply_gate_mode = reply_gate_mode
    _oracle(monkeypatch, "DELIVER")
    chat_id = gid if chat_type == "group" else CONTACT
    src = _src(chat_id, "Robin")
    response = await runner._handle_message_with_agent(_event(src), src, f"agent:main:whatsapp:{chat_type}:{chat_id}", 1)
    return response, adapter


@pytest.mark.asyncio
async def test_reply_gated_group_silence_stays_silent(monkeypatch, tmp_path):
    """reply_gate_mode "tool" + a free-response group turn: the agent correctly staying out of a
    conversation must post nothing, not a "returned only a silence marker" warning."""
    response, adapter = await _silence_turn(
        monkeypatch, tmp_path, chat_type="group", reply_gate_mode="tool", reply_policy="FREE_RESPONSE")
    assert not (response or "").strip(), f"the group received: {response!r}"
    assert adapter.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type,reply_gate_mode,reply_policy", [
    ("dm", "tool", "DM"),                     # a DM always expects a reply
    ("group", "prompt", "FREE_RESPONSE"),     # gate off: prompt mode
    ("group", "tool", "MENTION_GATED"),       # mention-gated group
])
async def test_silence_where_a_reply_was_expected_gets_the_visible_fallback(
    monkeypatch, tmp_path, chat_type, reply_gate_mode, reply_policy,
):
    from gateway.run_turn import _UNEXPECTED_SILENCE_REPLY

    response, _adapter = await _silence_turn(
        monkeypatch, tmp_path, chat_type=chat_type, reply_gate_mode=reply_gate_mode, reply_policy=reply_policy)
    assert response == _UNEXPECTED_SILENCE_REPLY
