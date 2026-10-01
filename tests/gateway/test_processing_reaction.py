"""Tests for the deterministic contextual processing-reaction picker.

Covers: whole-token Spanish + English classification (including the 11 sentences the old
substring matcher got wrong), positive per-category cases (es + en), the rotating anti-repeat
cursor (start and success outcome walk the whole pool before repeating), the per-chat bounded
LRU, outcome (varied success / fixed failure), the "every pool emoji is Telegram-allowed"
invariant asserted against python-telegram-bot's ``ReactionEmoji`` (not the module's own
constant), and the base-adapter mode gate resolved from a real ``PlatformConfig``/``GatewayConfig``.
"""

import pytest

from gateway.platforms.event import MessageType
from gateway.processing_reaction import (
    TELEGRAM_ALLOWED_REACTIONS,
    STATIC_ACK_EMOJI,
    STATIC_OK_EMOJI,
    STATIC_FAIL_EMOJI,
    SUCCESS_POOL,
    classify,
    pick_processing_reaction,
    pick_outcome_reaction,
    ChatReactionMemory,
    _POOLS,
)


# ── Pool integrity: derive the allowed set from python-telegram-bot, not our own constant ────

def _ptb_allowed() -> frozenset:
    """Authoritative Telegram free-reaction set from the REAL python-telegram-bot.

    The gateway test conftest installs a MagicMock ``telegram`` into ``sys.modules`` at import
    time, so a plain ``from telegram.constants import ReactionEmoji`` resolves the mock (which
    iterates empty, and whose ``StringEnum`` can't be subclassed). Temporarily evict every
    ``telegram*`` module so the genuine package imports from site-packages, read the constant,
    then restore the mock so the rest of the gateway suite still sees its stub.
    """
    import importlib
    import sys

    saved = {name: mod for name, mod in sys.modules.items()
             if name == "telegram" or name.startswith("telegram.")}
    for name in saved:
        del sys.modules[name]
    try:
        real = importlib.import_module("telegram.constants")
        members = list(getattr(real, "ReactionEmoji", []))
        if not members or not hasattr(real, "__file__"):
            pytest.skip("real python-telegram-bot ReactionEmoji not importable")
        return frozenset(e.value.replace("\ufe0f", "") for e in members)
    finally:
        for name in [n for n in sys.modules
                     if n == "telegram" or n.startswith("telegram.")]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_all_pool_emojis_are_telegram_allowed_per_ptb():
    allowed = _ptb_allowed()
    for category, pool in _POOLS.items():
        assert pool, f"pool {category} is empty"
        for emoji in pool:
            assert emoji.replace("\ufe0f", "") in allowed, \
                f"{category}: {emoji!r} not in telegram.constants.ReactionEmoji"


def test_success_and_static_outcomes_are_telegram_allowed_per_ptb():
    allowed = _ptb_allowed()
    for emoji in (*SUCCESS_POOL, STATIC_ACK_EMOJI, STATIC_OK_EMOJI, STATIC_FAIL_EMOJI):
        assert emoji.replace("\ufe0f", "") in allowed, f"{emoji!r} not TG-allowed per ptb"


def test_module_constant_matches_ptb():
    # The hand-written constant is only kept because it is a verified subset of ptb's set.
    allowed = _ptb_allowed()
    for emoji in TELEGRAM_ALLOWED_REACTIONS:
        assert emoji.replace("\ufe0f", "") in allowed, f"{emoji!r} claims TG-allowed but isn't"


def test_intel_emoji_is_not_telegram_allowed():
    # Regression: 🧐 was in an old pool but is NOT a free Telegram reaction (setMessageReaction
    # would fail). It must not reappear in any pool.
    assert "\U0001f9d0" not in _ptb_allowed()
    for pool in _POOLS.values():
        assert "\U0001f9d0" not in pool


# ── The 11 sentences the raw-substring classifier got wrong ─────────────────────────────────
# Each is annotated with the bogus substring the old matcher fired on. A Spanish speaker reads
# these mostly as question / default / task — never the old category.

@pytest.mark.parametrize("text,expected,old_bug", [
    ("¿viste el episodio de anoche?", "question", "'odio' inside 'episodio' → bad_news"),
    ("whatever, mándalo", "task", "'odio'? → bad_news; now task via 'mandalo'"),
    ("por supuesto", "default", "'sup' → greeting"),
    ("yo quiero ver Dune", "default", "'yo ' → greeting"),
    ("de hecho ya lo vi", "default", "'hecho' → done"),
    ("me quedo en casa", "default", "'quedo' → done"),
    ("el mes pasado", "default", "'mal'/'sad' → bad_news"),
    ("revisa mi correo", "task", "'corre' → urgent; now task via 'revisa'"),
    ("todo normal", "default", "'mal' → bad_news"),
    ("Graciela dice hola", "greeting", "'graciela' listed → gratitude; now greeting via 'hola'"),
    ("pretty cool", "default", "'ty ' → gratitude"),
])
def test_no_substring_misclassification(text, expected, old_bug):
    assert classify(text) == expected, f"{text!r}: {old_bug}"


# ── Positive classification: Spanish ────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("gracias!", "gratitude"),
    ("mil gracias, te lo agradezco", "gratitude"),
    ("jajaja no manches", "humor"),
    ("¿cómo hago esto?", "question"),
    ("cuál es el mejor", "question"),
    ("porfa arregla el bug", "task"),
    ("puedes ayudarme", "task"),
    ("revisa el correo por favor", "task"),
    ("urge ya", "urgent"),
    ("apúrate está urgente", "urgent"),
    ("no funciona nada", "bad_news"),
    ("el sitio está roto", "bad_news"),
    ("odio este error", "bad_news"),
    ("ya quedó listo", "done"),
    ("hola buenas tardes", "greeting"),
    ("qué onda", "greeting"),
])
def test_classify_spanish(text, expected):
    assert classify(text) == expected


# ── Positive classification: English ────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("thanks a lot", "gratitude"),
    ("thank you so much", "gratitude"),
    ("lol that's hilarious", "humor"),
    ("how do I do this?", "question"),
    ("can you fix the bug", "task"),
    ("please build me a script", "task"),
    ("this is urgent, asap", "urgent"),
    ("the build is broken", "bad_news"),
    ("I hate this crash", "bad_news"),
    ("all done, works now", "done"),
    ("hey there", "greeting"),
    ("hello", "greeting"),
])
def test_classify_english(text, expected):
    assert classify(text) == expected


def test_accent_folding():
    assert classify("¿CÓMO?") == "question"
    assert classify("APÚRATE") == "urgent"


def test_media_type_without_text():
    assert classify("", MessageType.PHOTO) == "media"
    assert classify(None, MessageType.VOICE) == "media"


def test_media_with_specific_caption_wins_text():
    # A caption with a strong signal beats the media default.
    assert classify("gracias por la foto", MessageType.PHOTO) == "gratitude"


# ── Rotation anti-repeat: START walks the whole pool before repeating ───────────────────────

def test_start_rotation_cycles_whole_pool():
    mem = ChatReactionMemory()
    text = "puedes arreglar el bug"  # task
    pool = _POOLS["task"]
    picks = [mem.pick("chatA", text) for _ in range(len(pool) * 2)]
    # First N picks are the full pool in order; then it repeats the same cycle.
    assert picks[:len(pool)] == list(pool)
    assert picks[len(pool):] == list(pool)
    # No immediate repeats within a cycle.
    for i in range(1, len(pool)):
        assert picks[i] != picks[i - 1]


def test_start_rotation_is_per_chat_and_per_category():
    mem = ChatReactionMemory()
    # Different chats keep independent cursors.
    a = mem.pick("chatA", "puedes arreglar esto")   # task, cursor 0
    b = mem.pick("chatB", "puedes arreglar esto")   # task, cursor 0 (fresh chat)
    assert a == b == _POOLS["task"][0]
    # Different category in the same chat has its own cursor.
    q = mem.pick("chatA", "¿cómo funciona?")        # question, cursor 0
    assert q == _POOLS["question"][0]


def test_six_task_turns_do_not_collapse_to_one_pair():
    # The original bug: start(🫡)+outcome(👍) shared one slot → six task turns produced
    # '🫡👍 🫡👍 …'. Now start rotates and outcome rotates on a SEPARATE cursor.
    mem = ChatReactionMemory()
    starts, outcomes = [], []
    for _ in range(6):
        starts.append(mem.pick("c", "arregla el bug porfa"))   # task
        outcomes.append(mem.pick_outcome("c", True))
    # Starts are not all identical; the first pool cycle is distinct.
    assert len(set(starts[:len(_POOLS["task"])])) == len(_POOLS["task"])
    # Success outcomes vary too (not the same emoji six times).
    assert len(set(outcomes[:len(SUCCESS_POOL)])) == len(SUCCESS_POOL)


# ── Rotation anti-repeat: SUCCESS OUTCOME walks the whole success pool ───────────────────────

def test_success_outcome_rotation_cycles_whole_pool():
    mem = ChatReactionMemory()
    picks = [mem.pick_outcome("chatZ", True) for _ in range(len(SUCCESS_POOL) * 2)]
    assert picks[:len(SUCCESS_POOL)] == list(SUCCESS_POOL)
    assert picks[len(SUCCESS_POOL):] == list(SUCCESS_POOL)


def test_failure_outcome_is_always_fixed():
    mem = ChatReactionMemory()
    for _ in range(5):
        assert mem.pick_outcome("chatZ", False) == STATIC_FAIL_EMOJI
    assert pick_outcome_reaction(False) == STATIC_FAIL_EMOJI


def test_start_and_outcome_cursors_are_independent():
    # Interleaving start and outcome must not perturb each other's rotation.
    mem = ChatReactionMemory()
    s0 = mem.pick("c", "hazme un script")       # task cursor 0
    o0 = mem.pick_outcome("c", True)            # success cursor 0
    s1 = mem.pick("c", "hazme un script")       # task cursor 1
    o1 = mem.pick_outcome("c", True)            # success cursor 1
    assert (s0, s1) == (_POOLS["task"][0], _POOLS["task"][1])
    assert (o0, o1) == (SUCCESS_POOL[0], SUCCESS_POOL[1])


# ── Bounded LRU ─────────────────────────────────────────────────────────────────────────────

def test_memory_is_bounded():
    mem = ChatReactionMemory(maxsize=8)
    for i in range(100):
        mem.pick(f"chat{i}", "hola")
        mem.pick_outcome(f"chat{i}", True)
    # Two maps, each capped at maxsize.
    assert len(mem) <= 8 * 2


def test_pure_pickers_are_deterministic():
    assert pick_processing_reaction("¿cómo?", MessageType.TEXT, 0) == _POOLS["question"][0]
    assert pick_processing_reaction("¿cómo?", MessageType.TEXT, 1) == _POOLS["question"][1]
    assert pick_outcome_reaction(True, 0) == SUCCESS_POOL[0]
    assert pick_outcome_reaction(True, 2) == SUCCESS_POOL[2]


# ── Base-adapter mode gate resolved from a REAL PlatformConfig / GatewayConfig ──────────────

def _make_adapter():
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter

    class _Probe(BasePlatformAdapter):
        async def connect(self, *, is_reconnect: bool = False) -> bool:
            return True

        async def disconnect(self) -> None:
            ...

        async def send(self, chat_id, content, reply_to=None, *a, **k):
            ...

        async def get_chat_info(self, chat_id):
            return {}

    return _Probe(PlatformConfig(enabled=True), Platform.TELEGRAM)


def _write_config(tmp_path, monkeypatch, display: dict):
    import yaml
    cfg_dir = tmp_path / "home"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump({"display": display}), encoding="utf-8")
    # _load_gateway_config reads _gateway_config_home()/config.yaml.
    import gateway.run as run_mod
    monkeypatch.setattr(run_mod, "_hermes_home", cfg_dir, raising=False)
    monkeypatch.setattr(run_mod, "_gateway_config_home", lambda: cfg_dir, raising=False)


def test_mode_defaults_to_static_with_real_platformconfig():
    # A bare PlatformConfig has no .display attr at all (it has .extra) — the old code returned
    # 'static' by accident but could never reach 'smart'. Default must be 'static'.
    adapter = _make_adapter()
    # No config file wired → _load_gateway_config fails-open to {} → default static.
    assert adapter._processing_reaction_mode() == "static"


def test_smart_mode_is_reachable_via_display_config(tmp_path, monkeypatch):
    adapter = _make_adapter()
    _write_config(tmp_path, monkeypatch, {"processing_reaction": "smart"})
    assert adapter._processing_reaction_mode() == "smart"


def test_off_mode_via_display_config(tmp_path, monkeypatch):
    adapter = _make_adapter()
    _write_config(tmp_path, monkeypatch, {"processing_reaction": "off"})
    assert adapter._processing_reaction_mode() == "off"


def test_per_platform_override_wins(tmp_path, monkeypatch):
    adapter = _make_adapter()  # Platform.TELEGRAM
    _write_config(tmp_path, monkeypatch, {
        "processing_reaction": "off",
        "platforms": {"telegram": {"processing_reaction": "smart"}},
    })
    assert adapter._processing_reaction_mode() == "smart"


def test_unknown_mode_degrades_to_static(tmp_path, monkeypatch):
    adapter = _make_adapter()
    _write_config(tmp_path, monkeypatch, {"processing_reaction": "bogus"})
    assert adapter._processing_reaction_mode() == "static"


def test_smart_ack_emoji_is_contextual_and_off_is_none(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.config import Platform

    adapter = _make_adapter()
    _write_config(tmp_path, monkeypatch, {"processing_reaction": "smart"})
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1")
    event = MessageEvent(source=src, text="¿cómo funciona esto?", message_type=MessageType.TEXT)
    assert adapter._processing_ack_emoji(event) == _POOLS["question"][0]

    _write_config(tmp_path / "x", monkeypatch, {"processing_reaction": "off"})
    assert adapter._processing_ack_emoji(event) is None
