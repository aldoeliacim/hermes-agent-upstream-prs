"""The processing reaction never fires on a reply-gated silence-likely turn.

A free-response group turn under ``reply_gate_mode: tool`` lets the agent decide to stay
silent. A reaction there (start OR outcome) tells the room the bot read and acted, which
breaks that silence. These tests drive the real gate: a real ``GatewayConfig``, a real
``SessionSource`` carrying a real ``ReplyPolicy``, and the runner's own
``_is_reply_tool_gated`` predicate reached through ``adapter.gateway_runner``. Only the
wire (the bridge ``/react`` POST) is replaced, so an assertion of "no call" means no
reaction reached WhatsApp.

The last test is structural: every adapter hook that sets a processing reaction must
consult the suppression gate, so a new adapter cannot quietly skip it.
"""

import ast
import pathlib

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.reply_policy import ReplyPolicy
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource

_REPO = pathlib.Path(__file__).resolve().parents[2]


class _Runner(GatewayTurnMixin):
    """The runner's real reply-gate predicate over a real ``GatewayConfig``."""

    def __init__(self, reply_gate_mode: str) -> None:
        self.config = GatewayConfig.from_dict({"reply_gate_mode": reply_gate_mode})


def _adapter(tmp_path, monkeypatch, *, mode: str = "smart", reply_gate_mode: str = "tool"):
    import yaml

    import gateway.run as run_mod
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({"display": {"processing_reaction": mode}}), encoding="utf-8")
    monkeypatch.setattr(run_mod, "_gateway_config_home", lambda: home)

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"reactions": True}))
    adapter.gateway_runner = _Runner(reply_gate_mode)
    calls: list = []

    async def _wire(chat_id, message_id, emoji):
        calls.append((chat_id, message_id, emoji))
        return True

    monkeypatch.setattr(adapter, "_post_reaction", _wire)
    return adapter, calls


def _event(policy: ReplyPolicy, chat_type: str = "group") -> MessageEvent:
    source = SessionSource(platform=Platform.WHATSAPP, chat_id="120363000000000000@g.us",
                           chat_type=chat_type, user_id="15550001111@s.whatsapp.net",
                           reply_policy=policy)
    return MessageEvent(text="¿alguien sabe cómo se arregla esto?", message_type=MessageType.TEXT,
                        source=source, message_id="MSGID-1")


async def _full_turn(adapter, event, outcome=ProcessingOutcome.SUCCESS):
    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, outcome)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [ProcessingOutcome.SUCCESS, ProcessingOutcome.FAILURE,
                                     ProcessingOutcome.CANCELLED])
async def test_free_response_group_under_tool_gate_gets_no_reaction(tmp_path, monkeypatch, outcome):
    adapter, calls = _adapter(tmp_path, monkeypatch)
    await _full_turn(adapter, _event(ReplyPolicy.FREE_RESPONSE), outcome)
    assert calls == []


@pytest.mark.asyncio
async def test_static_mode_is_gated_too(tmp_path, monkeypatch):
    """The gate is not a smart-mode feature: the fixed 👀 must not leak either."""
    adapter, calls = _adapter(tmp_path, monkeypatch, mode="static")
    await _full_turn(adapter, _event(ReplyPolicy.FREE_RESPONSE))
    assert calls == []


@pytest.mark.asyncio
async def test_mention_gated_group_still_reacts_start_then_outcome(tmp_path, monkeypatch):
    adapter, calls = _adapter(tmp_path, monkeypatch)
    await _full_turn(adapter, _event(ReplyPolicy.MENTION_GATED))
    assert len(calls) == 2
    (_, mid1, start), (_, mid2, done) = calls
    assert mid1 == mid2 == "MSGID-1"
    assert start and done and start != done


@pytest.mark.asyncio
async def test_prompt_mode_free_response_group_still_reacts(tmp_path, monkeypatch):
    """Without the tool gate the agent always answers, so there is no silence to protect."""
    adapter, calls = _adapter(tmp_path, monkeypatch, reply_gate_mode="prompt")
    await _full_turn(adapter, _event(ReplyPolicy.FREE_RESPONSE))
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_dm_reacts_under_tool_gate(tmp_path, monkeypatch):
    adapter, calls = _adapter(tmp_path, monkeypatch)
    await _full_turn(adapter, _event(ReplyPolicy.DM, chat_type="dm"))
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_cancelled_turn_retracts_instead_of_setting_an_outcome(tmp_path, monkeypatch):
    adapter, calls = _adapter(tmp_path, monkeypatch)
    await _full_turn(adapter, _event(ReplyPolicy.DM, chat_type="dm"), ProcessingOutcome.CANCELLED)
    assert [c[2] for c in calls][-1] == ""


@pytest.mark.asyncio
async def test_predicate_failure_fails_closed(tmp_path, monkeypatch):
    adapter, calls = _adapter(tmp_path, monkeypatch)

    def _boom(_source):
        raise RuntimeError("runner torn down mid-turn")

    monkeypatch.setattr(adapter.gateway_runner, "_is_reply_tool_gated", _boom)
    await _full_turn(adapter, _event(ReplyPolicy.DM, chat_type="dm"))
    assert calls == []


@pytest.mark.asyncio
async def test_whatsapp_reactions_default_off_without_explicit_opt_in(tmp_path, monkeypatch):
    """WhatsApp never reacted before this layer; an upgrade must not start doing so unasked."""
    adapter, calls = _adapter(tmp_path, monkeypatch)
    adapter.config.extra.pop("reactions")
    await _full_turn(adapter, _event(ReplyPolicy.DM, chat_type="dm"))
    assert calls == []


@pytest.mark.asyncio
async def test_config_edit_takes_effect_without_restart(tmp_path, monkeypatch):
    import os

    import yaml

    adapter, calls = _adapter(tmp_path, monkeypatch, mode="smart")
    event = _event(ReplyPolicy.DM, chat_type="dm")
    assert adapter._processing_reaction_mode() == "smart"
    cfg = tmp_path / "home" / "config.yaml"
    cfg.write_text(yaml.safe_dump({"display": {"processing_reaction": "off"}}), encoding="utf-8")
    st = cfg.stat()
    os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    await _full_turn(adapter, event)
    assert calls == []


_HOOKS = {"on_processing_start", "on_processing_complete"}
_GATES = {"_processing_reaction_suppressed", "_processing_ack_emoji"}


def _hook_files():
    for rel in ("gateway/platforms", "gateway/relay", "plugins/platforms"):
        yield from sorted((_REPO / rel).rglob("*.py"))


def test_every_reacting_processing_hook_consults_the_suppression_gate():
    """A processing hook that can put a reaction on the wire must route through the gate.

    Hooks that never react (a2a, google_chat, webhook: session bookkeeping only) are
    exempt because they contain no reaction call at all.
    """
    reaction_calls = {"_add_reaction", "_set_reaction", "_send_reaction", "send_reaction",
                      "_react", "_post_reaction", "react", "add"}
    offenders = []
    for path in _hook_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef) or node.name not in _HOOKS:
                continue
            called = {
                (n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None))
                for n in ast.walk(node) if isinstance(n, ast.Call)
            }
            if called & reaction_calls and not called & _GATES:
                offenders.append(f"{path.relative_to(_REPO)}::{node.name}")
    assert offenders == [], f"processing hooks that can react without the suppression gate: {offenders}"
