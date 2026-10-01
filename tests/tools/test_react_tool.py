"""Gate for the ``react`` tool and the uniform adapter reaction contract.

Three things are proven here, each with a positive AND a negative case:

1. **Scoping by construction** — the tool has no target parameter; it reacts
   in the bound turn's chat or not at all.  Outside a live turn it refuses.
2. **Authorization** — owner turns are unrestricted; non-owner turns are
   allowed ONLY under the tool-gated free-response policy.  A non-owner
   prompt-mode turn is refused both at schema level (check_fn) and at call
   time (handler), independently.
3. **The contract is the only path** — an adapter that does not declare
   ``supports_reactions`` is refused even if it happens to have a method
   called ``add_reaction``; an adapter that declares it is called through
   ``react``/``unreact`` with the normalised ``(chat_id, message_id, emoji)``
   shape regardless of its native primitive.

Plus the ``messages_back`` resolution against the transcript.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import react_tool as rt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _Adapter:
    """Minimal adapter honouring the uniform contract, recording calls."""

    supports_reactions = True

    def __init__(self):
        self.calls: list[tuple] = []

    async def react(self, chat_id, message_id, emoji):
        self.calls.append(("react", chat_id, message_id, emoji))
        return True

    async def unreact(self, chat_id, message_id, emoji=""):
        self.calls.append(("unreact", chat_id, message_id, emoji))
        return True


class _LegacyOnlyAdapter:
    """Has the OLD primitive name but never opted into the contract."""

    async def add_reaction(self, chat_id, emoji, message_id):  # pragma: no cover
        raise AssertionError("legacy primitive must never be reached")


def _concrete_base():
    from gateway.platforms.base import BasePlatformAdapter

    class _Concrete(BasePlatformAdapter):
        """Satisfy the ABC; bypass the heavy base __init__."""
        def __init__(self):
            pass
        async def connect(self): return True
        async def disconnect(self): return None
        async def send(self, *a, **k): return None
        async def get_chat_info(self, *a, **k): return {}

    return _Concrete


def _source(**kw):
    # NOTE: no session_id — the real SessionSource has none (it describes the
    # chat, not the transcript). A stub that carries one hides the bug where
    # the tool read it off the source and silently never found earlier rows.
    base = dict(platform="whatsapp", chat_id="5215550000000@s.whatsapp.net",
                user_id="5215550000000", message_id="MSG-TRIGGER", chat_type="dm")
    base.update(kw)
    return SimpleNamespace(**base)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def owner_turn():
    src = _source()
    adapter = _Adapter()
    with patch.object(rt, "get_current_message_source", return_value=src), \
         patch.object(rt, "_live_adapter", return_value=adapter), \
         patch.object(rt, "is_owner_source", return_value=True), \
         patch.object(rt, "is_reply_tool_gated_turn", return_value=False):
        yield src, adapter


# ---------------------------------------------------------------------------
# 1. Scoping
# ---------------------------------------------------------------------------

def test_schema_has_no_target_parameter():
    """The #47856 hazard is a *chosen* target; the tool cannot express one."""
    props = rt.REACT_SCHEMA["parameters"]["properties"]
    assert set(props) == {"emoji", "messages_back"}
    assert "target" not in json.dumps(rt.REACT_SCHEMA)


def test_refused_outside_a_live_turn():
    with patch.object(rt, "get_current_message_source", return_value=None):
        out = json.loads(_run(rt.react_tool("👍")))
    assert "error" in out
    assert rt.check_react_available() is False


def test_reacts_to_the_triggering_message_by_default(owner_turn):
    src, adapter = owner_turn
    out = json.loads(_run(rt.react_tool("❤️")))
    assert out["success"] is True
    assert adapter.calls == [("react", src.chat_id, "MSG-TRIGGER", "❤️")]


def test_empty_emoji_retracts(owner_turn):
    src, adapter = owner_turn
    out = json.loads(_run(rt.react_tool("")))
    assert out["success"] is True and out["emoji"] is None
    assert adapter.calls == [("unreact", src.chat_id, "MSG-TRIGGER", "")]


# ---------------------------------------------------------------------------
# 2. Authorization — positive and NEGATIVE, at both layers
# ---------------------------------------------------------------------------

def test_owner_turn_is_exposed_and_allowed(owner_turn):
    assert rt.check_react_available() is True
    assert json.loads(_run(rt.react_tool("👍")))["success"] is True


def test_non_owner_prompt_mode_turn_is_refused_at_both_layers():
    """A stranger's DM/group in prompt mode: no schema entry AND a refusal."""
    src = _source(user_id="5210000000001")
    adapter = _Adapter()
    with patch.object(rt, "get_current_message_source", return_value=src), \
         patch.object(rt, "_live_adapter", return_value=adapter), \
         patch.object(rt, "is_owner_source", return_value=False), \
         patch.object(rt, "is_reply_tool_gated_turn", return_value=False):
        assert rt.check_react_available() is False
        out = json.loads(_run(rt.react_tool("😂")))
    assert "error" in out
    assert adapter.calls == [], "refusal must happen BEFORE any platform I/O"


def test_non_owner_tool_gated_turn_is_allowed():
    """Reacting is never MORE permissive than speaking — but not less."""
    src = _source(user_id="5210000000001", chat_type="group")
    adapter = _Adapter()
    with patch.object(rt, "get_current_message_source", return_value=src), \
         patch.object(rt, "_live_adapter", return_value=adapter), \
         patch.object(rt, "is_owner_source", return_value=False), \
         patch.object(rt, "is_reply_tool_gated_turn", return_value=True):
        assert rt.check_react_available() is True
        out = json.loads(_run(rt.react_tool("😂")))
    assert out["success"] is True
    assert adapter.calls == [("react", src.chat_id, "MSG-TRIGGER", "😂")]


def test_check_fn_is_uncached():
    """A 30s process-wide cache would leak the owner's verdict into a stranger's turn."""
    from tools.registry import _NO_CACHE_CHECK_FNS
    assert rt.check_react_available in _NO_CACHE_CHECK_FNS


def test_reacting_is_not_a_delivery_decision(owner_turn):
    """The reply gate's one-decision-per-message invariant must survive."""
    from tools.approval import TurnDeliveryState, set_turn_delivery_state, reset_turn_delivery_state
    state = TurnDeliveryState()
    tok = set_turn_delivery_state(state)
    try:
        _run(rt.react_tool("👍"))
    finally:
        reset_turn_delivery_state(tok)
    assert state.tool_sends == 0 and state.decided_silent == 0
    assert state.delivery_decided is False


# ---------------------------------------------------------------------------
# 3. The contract is the only path
# ---------------------------------------------------------------------------

def test_adapter_without_contract_is_refused_even_with_legacy_primitive():
    src = _source()
    with patch.object(rt, "get_current_message_source", return_value=src), \
         patch.object(rt, "_live_adapter", return_value=_LegacyOnlyAdapter()), \
         patch.object(rt, "is_owner_source", return_value=True):
        assert rt.check_react_available() is False
        out = json.loads(_run(rt.react_tool("👍")))
    assert "error" in out


def test_every_reaction_capable_adapter_declares_the_contract():
    """Static sweep: an adapter with a native reaction primitive MUST opt in
    with ``supports_reactions`` so callers never fall back to guessing.

    Walks EVERY .py file under the repo that defines a ``BasePlatformAdapter``
    subclass — not a directory glob. The first version globbed
    ``plugins/platforms/*/adapter.py`` + ``gateway/platforms/*.py`` and missed
    ``gateway/relay/adapter.py`` (a ninth reaction-capable adapter whose ack
    reflex had no off switch at all). A capability that lives at the edges
    must be swept by what it IS, not where it sits.
    """
    import re
    from pathlib import Path
    root = Path(rt.__file__).resolve().parents[1]
    native = re.compile(r"async def (?:_?(?:add|send|set)_reaction|_?react)\(")
    declares = re.compile(r"supports_reactions(?::\s*bool)?\s*=\s*True|def supports_reactions\(")
    offenders, swept = [], 0
    for f in root.rglob("*.py"):
        parts = f.relative_to(root).parts
        if parts[0] in {"tests", "venv", ".venv", "node_modules", "website"} or "/tests/" in str(f):
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        if "BasePlatformAdapter)" not in text and "(BasePlatformAdapter" not in text:
            continue
        if f.name == "base.py" and parts[-2] == "platforms":
            continue
        swept += 1
        if native.search(text) and not declares.search(text):
            offenders.append(str(f.relative_to(root)))
    assert swept >= 9, f"sweep found only {swept} adapter files — the discovery rule regressed"
    assert not offenders, f"adapters with a reaction primitive but no contract: {offenders}"


def test_base_default_delegates_to_add_reaction_shape():
    """Default base impl maps (chat, msg, emoji) → add_reaction(chat, emoji, msg)."""
    _Concrete = _concrete_base()
    calls = []

    class A(_Concrete):
        supports_reactions = True
        name = "test"
        async def add_reaction(self, chat_id, emoji, message_id):
            calls.append((chat_id, emoji, message_id)); return True
        async def remove_reaction(self, chat_id, message_id):
            calls.append(("rm", chat_id, message_id)); return True

    a = A()
    assert _run(a.react("C", "M", "👍")) is True
    assert _run(a.unreact("C", "M")) is True
    assert calls == [("C", "👍", "M"), ("rm", "C", "M")]


def test_base_default_is_a_safe_noop_without_primitive():
    _Concrete = _concrete_base()

    class A(_Concrete):
        name = "bare"

    a = A()
    assert a.supports_reactions is False
    assert _run(a.react("C", "M", "👍")) is False
    assert _run(a.unreact("C", "M")) is False


# ---------------------------------------------------------------------------
# messages_back resolution
# ---------------------------------------------------------------------------

class _DB:
    def __init__(self, ids):
        self.ids = ids
    def get_recent_user_platform_message_ids(self, session_id, limit=5):
        return self.ids[:limit]


def test_messages_back_walks_persisted_user_rows_newest_first():
    src = _source(message_id="M0")
    db = _DB(["M0", "M1", "M2", "M3"])
    assert rt.resolve_target_message_id(src, 0, session_db=db, session_id="s") == "M0"
    assert rt.resolve_target_message_id(src, 1, session_db=db, session_id="s") == "M1"
    assert rt.resolve_target_message_id(src, 2, session_db=db, session_id="s") == "M2"
    assert rt.resolve_target_message_id(src, 9, session_db=db, session_id="s") is None


def test_messages_back_tolerates_trigger_not_yet_persisted():
    """If the trigger row hasn't landed (or has no id), 'one back' is still
    the message before the trigger — never the trigger itself."""
    src = _source(message_id="M0")
    db = _DB(["M1", "M2"])
    assert rt.resolve_target_message_id(src, 1, session_db=db, session_id="s") == "M1"
    assert rt.resolve_target_message_id(src, 2, session_db=db, session_id="s") == "M2"


def test_signal_composite_id_roundtrip():
    from gateway.platforms.signal import SignalAdapter
    assert SignalAdapter._split_message_id("+15551234567:1725000000123") == ("+15551234567", 1725000000123)
    assert SignalAdapter._split_message_id("uuid-with:colons:1725000000123") == ("uuid-with:colons", 1725000000123)
    assert SignalAdapter._split_message_id("garbage") is None
    assert SignalAdapter._split_message_id("author:notanumber") is None


def test_messages_back_end_to_end_reads_session_from_turn_env(owner_turn):
    """messages_back must resolve through the turn's session env, not the source."""
    src, adapter = owner_turn
    assert not hasattr(src, "session_id")
    fake_db = _DB(["MSG-TRIGGER", "MSG-PREV", "MSG-PREV2"])
    fake_db.close = lambda: None
    with patch("gateway.session_context.get_session_env", return_value="sess-1"), \
         patch("hermes_state.SessionDB", return_value=fake_db):
        out = json.loads(_run(rt.react_tool("😂", messages_back=1)))
    assert out["success"] is True
    assert adapter.calls == [("react", src.chat_id, "MSG-PREV", "😂")]


def test_messages_back_without_session_env_fails_closed(owner_turn):
    src, adapter = owner_turn
    with patch("gateway.session_context.get_session_env", return_value=""):
        out = json.loads(_run(rt.react_tool("😂", messages_back=1)))
    assert "error" in out and adapter.calls == []
