"""Deterministic, zero-cost contextual picker for the *processing-start* reaction ack.

When a platform adapter begins working on an inbound message it may set a single emoji
reaction on that message as a read-receipt ("I saw this, I'm on it"). Historically every
adapter hardcoded 👀. This module turns that into a **contextual, non-repetitive** choice
without any model call and without any extra network round-trip beyond the single reaction
the adapter was already going to send.

Design constraints (see gateway/AGENTS.md, CONTRIBUTING.md):

* **Zero LLM cost, zero extra network.** Pure string classification + a per-chat LRU. No I/O.
* **One reaction slot per message, start→outcome lifecycle preserved.** The adapter sets a
  start emoji, then *replaces* it (never stacks) with a success/failure emoji at completion —
  the sane lifecycle upstream already ships. This module makes the *start* choice contextual
  and the *success* choice varied, while keeping failure unambiguously readable as failure.
* **Bilingual (Spanish + English).** The owner and contacts are Mexican; ``¿cómo``, ``porfa``,
  ``gracias``, ``jaja``, ``urge``, ``ya quedó`` all classify correctly.
* **Whole-token matching.** Classification tokenizes (accent-fold + lowercase, split on
  non-letters) and matches whole tokens / exact multi-word phrases on token boundaries — never
  raw substrings. ``episodio`` does not match ``odio``; ``por supuesto`` does not match ``sup``.
* **Anti-repeat by rotation.** Each category maps to a *pool*; a per-chat, per-category cursor
  rotates through the whole pool before any emoji repeats. Start and outcome keep **separate**
  memories so a start pick never overwrites the outcome pick's rotation (or vice versa).
* **Reaction-safe emojis only.** Every emoji here is a member of Telegram's standard free
  reaction set (``telegram.constants.ReactionEmoji``) — a non-member makes ``setMessageReaction``
  fail — and every one also renders fine as a WhatsApp reaction. The test suite asserts pool
  membership against python-telegram-bot directly; ``TELEGRAM_ALLOWED_REACTIONS`` mirrors it.

This complements the platform-specific processing-status work in #29354 (WhatsApp 👀/✅/❌) and
#14667 (Feishu content-aware reactions) by providing the *generic, platform-agnostic* picker
none of them offered; it does not replace their per-platform lifecycle.
"""

from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from typing import Iterable, Optional

from gateway.platforms.event import MessageType

__all__ = [
    "TELEGRAM_ALLOWED_REACTIONS",
    "STATIC_ACK_EMOJI",
    "STATIC_OK_EMOJI",
    "STATIC_FAIL_EMOJI",
    "SUCCESS_POOL",
    "classify",
    "pick_processing_reaction",
    "pick_outcome_reaction",
    "ChatReactionMemory",
]

# Telegram's standard free reaction set (Bot API ``ReactionTypeEmoji`` /
# ``telegram.constants.ReactionEmoji``), U+FE0F stripped. A reaction outside this set makes
# ``setMessageReaction`` raise, so every pool below draws only from here. This is also a
# WhatsApp-safe subset (WhatsApp accepts any emoji as a reaction). The test suite derives the
# authoritative set from python-telegram-bot and asserts every pool member ∈ it, so this
# constant is kept only because it is test-verified against ptb — never hand-trusted.
TELEGRAM_ALLOWED_REACTIONS = frozenset({
    "\U0001f44d",  # 👍
    "\U0001f44e",  # 👎
    "\u2764",       # ❤
    "\U0001f525",  # 🔥
    "\U0001f970",  # 🥰
    "\U0001f44f",  # 👏
    "\U0001f601",  # 😁
    "\U0001f914",  # 🤔
    "\U0001f92f",  # 🤯
    "\U0001f631",  # 😱
    "\U0001f62d",  # 😭
    "\U0001f389",  # 🎉
    "\U0001f929",  # 🤩
    "\U0001f4af",  # 💯
    "\U0001f923",  # 🤣
    "\u26a1",       # ⚡
    "\U0001f440",  # 👀
    "\U0001f64f",  # 🙏
    "\U0001f44c",  # 👌
    "\U0001f4a9",  # 💩
    "\U0001f60d",  # 😍
    "\U0001fae1",  # 🫡
    "\U0001f60e",  # 😎
})

# The emoji used before this change: also the ``static`` mode value and the safe default when a
# category has no more specific choice would otherwise be inappropriate.
STATIC_ACK_EMOJI = "\U0001f440"  # 👀

# Outcome emojis. In smart mode the *success* reaction is drawn from a varied pool (anti-repeat)
# so a chatty conversation doesn't get the identical ✅ every turn; the *failure* reaction stays
# a single unambiguous marker — a failure must always read as a failure, never as flavour.
STATIC_OK_EMOJI = "\U0001f44d"    # 👍  (upstream Telegram/Photon success)
STATIC_FAIL_EMOJI = "\U0001f44e"  # 👎  (upstream Telegram/Photon failure)

# Varied success pool (all Telegram-allowed). Head is the neutral upstream 👍 so a rotated pick
# still reads as approval.
SUCCESS_POOL = ("\U0001f44d", "\U0001f525", "\U0001f4af", "\U0001f44c", "\U0001f389")  # 👍 🔥 💯 👌 🎉

# ── Category → pool. Order inside a pool is the deterministic rotation order. Every entry is a
#    member of ``TELEGRAM_ALLOWED_REACTIONS`` (asserted in tests against ptb). Keep pools short;
#    the rotating picker needs 2+ to avoid immediate repeats.
_POOLS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict((
    ("gratitude", ("\U0001f970", "\U0001f64f", "\u2764")),          # 🥰 🙏 ❤
    ("humor",     ("\U0001f923", "\U0001f601", "\U0001f929")),      # 🤣 😁 🤩
    ("greeting",  ("\U0001f60e", "\U0001f970", "\U0001f601")),      # 😎 🥰 😁
    ("bad_news",  ("\U0001fae1", "\U0001f64f", "\U0001f62d")),      # 🫡 🙏 😭
    ("urgent",    ("\u26a1", "\U0001f525", "\U0001fae1")),          # ⚡ 🔥 🫡
    ("done",      ("\U0001f525", "\U0001f4af", "\U0001f44c")),      # 🔥 💯 👌  ("ya quedó" praise)
    ("question",  ("\U0001f914", "\U0001f440", "\U0001f60e")),      # 🤔 👀 😎
    ("task",      ("\U0001fae1", "\U0001f44c", "\U0001f440")),      # 🫡 👌 👀
    ("media",     ("\U0001f440", "\U0001f60d", "\U0001f929")),      # 👀 😍 🤩
    ("default",   ("\U0001f440", "\U0001f44c", "\U0001fae1")),      # 👀 👌 🫡
))

# Media MessageType → category. Photo/video/sticker read as "look", voice/audio/doc as "media".
_MEDIA_TYPES = {
    MessageType.PHOTO, MessageType.VIDEO, MessageType.STICKER,
    MessageType.VOICE, MessageType.AUDIO, MessageType.DOCUMENT,
}

# ── Keyword tables. All matching is whole-token or exact multi-word phrase on token
#    boundaries, after accent-fold + lowercase. Single-word entries here are matched as whole
#    tokens (``odio`` never matches inside ``episodio``); multi-word entries (containing a
#    space) are matched as an ordered phrase of adjacent tokens. Over-broad needles that only
#    worked as raw substrings ('mal','sad','hecho','quedo','sup','yo','ty','corre','graciela')
#    are intentionally gone.

_GRATITUDE = frozenset({
    "gracias", "agradezco", "agradecido", "agradecida", "thanks", "thank", "thx",
    "appreciate", "appreciated",
})
_GRATITUDE_PHRASES = (
    ("te", "lo", "agradezco"), ("mil", "gracias"), ("muchas", "gracias"), ("thank", "you"),
)

_HUMOR = frozenset({
    "jaja", "jajaja", "jajajaja", "jeje", "jejeje", "jiji", "jojo", "lol", "lmao", "rofl",
    "haha", "hahaha", "hehe", "\U0001f602", "\U0001f923",
})
_HUMOR_PHRASES = (
    ("que", "risa"), ("me", "matas"), ("es", "un", "chiste"),
)

_GREETING = frozenset({
    "hola", "holi", "buenas", "wenas", "hey", "hi", "hello", "hey",
})
_GREETING_PHRASES = (
    ("buenos", "dias"), ("buenas", "tardes"), ("buenas", "noches"), ("que", "onda"),
    ("que", "tal"), ("que", "pex"),
)

_BAD_NEWS = frozenset({
    "roto", "rota", "fallo", "falla", "fallando", "error", "errores", "problema", "problemas",
    "terrible", "horrible", "odio", "harto", "harta", "frustrado", "frustrada", "molesto",
    "molesta", "enojado", "enojada", "broken", "failed", "failing", "crash", "crashed",
    "hate", "annoying", "frustrated", "frustrating", "angry", "upset", "triste",
    "\U0001f622", "\U0001f62d",
})
_BAD_NEWS_PHRASES = (
    ("no", "funciona"), ("no", "sirve"), ("no", "jala"), ("esta", "roto"), ("se", "rompio"),
    ("doesn't", "work"), ("does", "not", "work"), ("not", "working"),
)

_URGENT = frozenset({
    "urge", "urgente", "urgent", "asap", "ahorita", "rapido", "emergencia", "emergency",
    "immediately", "hurry", "prisa", "apurate",
})
_URGENT_PHRASES = (
    ("de", "inmediato"), ("right", "now"), ("ya", "mero"),
)

_DONE = frozenset({
    "listo", "terminado", "termine", "resuelto", "solved", "finished", "ready",
})
_DONE_PHRASES = (
    ("ya", "quedo"), ("quedo", "listo"), ("ya", "esta"), ("ya", "funciona"),
    ("works", "now"), ("ya", "jala"), ("ya", "sirve"), ("all", "done"),
)

# Task / request imperatives and modal-request markers (Spanish + English).
_TASK = frozenset({
    "porfa", "porfis", "please", "pls", "necesito", "puedes", "podrias", "hazme", "arregla",
    "revisa", "manda", "mandalo", "mandala", "envia", "crea", "genera", "busca", "dame",
    "descarga", "instala", "configura",
})
_TASK_PHRASES = (
    ("por", "favor"), ("me", "ayudas"), ("me", "puedes", "ayudar"), ("quiero", "que"),
    ("help", "me"), ("can", "you"), ("could", "you"), ("look", "into"), ("set", "up"),
    ("i", "need"), ("i", "want", "you", "to"),
)

# Interrogative openers (whole tokens) that mark a question absent a '?' mark.
_QUESTION_TOKENS = frozenset({
    "como", "cuando", "donde", "porque", "cual", "cuales", "quien", "quienes",
    "how", "what", "when", "where", "why", "which", "who",
})
_QUESTION_PHRASES = (
    ("por", "que"), ("can", "i"), ("should", "i"), ("is", "it"), ("are", "you"), ("do", "you"),
)

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _fold(s: str) -> str:
    """Lowercase + strip combining accents so ``¿Cómo`` folds to ``como``."""
    if not s:
        return ""
    decomposed = unicodedata.normalize("NFD", s)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower()


def _tokenize(folded: str) -> list[str]:
    """Split accent-folded text into whole word tokens (letters only; digits/punct are seps)."""
    return _TOKEN_RE.findall(folded)


def _has_phrase(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    """True when ``phrase`` appears as an ordered run of adjacent tokens."""
    n = len(phrase)
    if n == 0 or n > len(tokens):
        return False
    first = phrase[0]
    for i in range(len(tokens) - n + 1):
        if tokens[i] == first and tuple(tokens[i:i + n]) == phrase:
            return True
    return False


def _matches(tokens: list[str], token_set, phrases: Iterable[tuple[str, ...]] = ()) -> bool:
    """True when any whole token is in ``token_set`` or any multi-word phrase matches."""
    tset = set(tokens)
    if tset & token_set:
        return True
    return any(_has_phrase(tokens, p) for p in phrases)


def classify(text: Optional[str], message_type: MessageType = MessageType.TEXT) -> str:
    """Classify a message into one of the pool categories. Pure; no side effects.

    Matching is whole-token / exact-phrase on token boundaries (never raw substring). Order
    matters: gratitude/humor/bad-news/urgent/done are checked before the broad question/task/
    greeting buckets so a "gracias, pero no funciona" style message lands on the more salient
    sentiment. Media type wins only when there is no strong textual signal.
    """
    raw = text or ""
    folded = _fold(raw)
    tokens = _tokenize(folded)
    has_text = bool(tokens)

    # Strong sentiment / intent signals first.
    if _matches(tokens, _GRATITUDE, _GRATITUDE_PHRASES):
        return "gratitude"
    if _matches(tokens, _HUMOR, _HUMOR_PHRASES):
        return "humor"
    if _matches(tokens, _BAD_NEWS, _BAD_NEWS_PHRASES):
        return "bad_news"
    if _matches(tokens, _DONE, _DONE_PHRASES):
        return "done"
    if _matches(tokens, _URGENT, _URGENT_PHRASES):
        return "urgent"

    # Media with no strong sentiment reads as "here's something to look at".
    if message_type in _MEDIA_TYPES and not has_text:
        return "media"

    if _matches(tokens, _TASK, _TASK_PHRASES):
        return "task"

    # Questions: explicit Spanish inverted mark, trailing '?', or interrogative openers.
    if "?" in raw or "\u00bf" in raw:
        return "question"
    if _matches(tokens, _QUESTION_TOKENS, _QUESTION_PHRASES):
        return "question"

    if _matches(tokens, _GREETING, _GREETING_PHRASES):
        return "greeting"

    # Media alongside a caption that didn't match anything more specific.
    if message_type in _MEDIA_TYPES:
        return "media"

    return "default"


def _rotate(pool: tuple[str, ...], cursor: int) -> str:
    """Return the pool member at ``cursor`` (mod pool size)."""
    return pool[cursor % len(pool)]


def pick_processing_reaction(
    text: Optional[str],
    message_type: MessageType = MessageType.TEXT,
    cursor: int = 0,
) -> str:
    """Pick a contextual reaction emoji by rotating through the category pool.

    ``cursor`` is a per-chat, per-category counter (see :class:`ChatReactionMemory`): N
    consecutive same-category turns walk the whole pool before repeating. Fully deterministic
    for a given ``(text, type, cursor)``.
    """
    category = classify(text, message_type)
    pool = _POOLS.get(category) or _POOLS["default"]
    return _rotate(pool, cursor)


def pick_outcome_reaction(success: bool, cursor: int = 0) -> str:
    """Pick the completion (outcome) reaction that replaces the start emoji.

    On **success** a varied emoji from ``SUCCESS_POOL``, rotated by ``cursor`` so a busy chat
    doesn't collapse to the identical ✅ every turn. On **failure** always ``STATIC_FAIL_EMOJI``
    — a failure must be unmistakably a failure, so it is never varied for flavour.
    """
    if not success:
        return STATIC_FAIL_EMOJI
    return _rotate(SUCCESS_POOL, cursor)


class ChatReactionMemory:
    """Bounded per-chat, per-category rotation cursors, for anti-repeat.

    Two *separate* cursor maps — one for start acks, one for success outcomes — so a start pick
    never overwrites the outcome rotation (the original single-slot bug where 🫡 start then 👍
    outcome overwrote each other, collapsing six task turns to '🫡👍 🫡👍 …'). Each map is an
    LRU capped at ``maxsize`` (chat, category) keys (oldest evicted first) so a plain dict can't
    grow unbounded. Not thread-safe by design — the gateway drives one adapter turn at a time
    per chat, and a lock on a purely cosmetic hot path is not worth it.
    """

    __slots__ = ("_start", "_outcome", "_maxsize")

    def __init__(self, maxsize: int = 512) -> None:
        self._start: "OrderedDict[tuple[str, str], int]" = OrderedDict()
        self._outcome: "OrderedDict[str, int]" = OrderedDict()
        self._maxsize = max(1, int(maxsize))

    def _next(self, lru: "OrderedDict", key) -> int:
        """Return the current cursor for ``key`` and advance it (LRU-bounded)."""
        cursor = lru.get(key, 0)
        lru[key] = cursor + 1
        lru.move_to_end(key)
        while len(lru) > self._maxsize:
            lru.popitem(last=False)
        return cursor

    def pick(
        self,
        chat_id: Optional[str],
        text: Optional[str],
        message_type: MessageType = MessageType.TEXT,
    ) -> str:
        """Pick the *start* ack, rotating this chat's cursor for the message's category."""
        category = classify(text, message_type)
        key = (chat_id or "", category)
        cursor = self._next(self._start, key)
        return pick_processing_reaction(text, message_type, cursor)

    def pick_outcome(self, chat_id: Optional[str], success: bool) -> str:
        """Pick the *outcome* ack (varied success / fixed failure), rotating a separate cursor."""
        if not success:
            return STATIC_FAIL_EMOJI
        cursor = self._next(self._outcome, chat_id or "")
        return pick_outcome_reaction(True, cursor)

    def __len__(self) -> int:  # test helper: assert the LRU stays bounded
        return len(self._start) + len(self._outcome)
