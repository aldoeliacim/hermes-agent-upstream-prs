"""Gateway response filtering helpers.

These decide whether a completed agent turn should be delivered to the chat,
not what should be persisted in conversation history.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from typing import Any, Optional

# Exact whole-response markers meaning "the agent intentionally chose not to
# reply". Keep small and explicit; arbitrary empty output remains an
# error/empty-response path, not silence. A lane that does not think in English
# translates the sentinel rather than dropping it, and the whole control token
# then reaches the user as content, so the translated forms are carried here
# too. zh-Hans is the only non-English locale this project ships documentation
# for, which is where the list stops.
LIVE_GATEWAY_SILENT_MARKERS = frozenset({
    "[SILENT]", "SILENT", "NO_REPLY", "NO REPLY",
    "[静默]", "静默", "[沉默]", "沉默",
})

# Bracketed markers drive the autonomous lane's prefix rule ("[SILENT] nothing
# new this tick"). Derived from the set above so a new marker cannot be added to
# one rule and forgotten in the other.
_BRACKETED_SILENCE_MARKERS = tuple(
    sorted(m for m in LIVE_GATEWAY_SILENT_MARKERS if m.startswith("["))
)

# The persisted user-row kind of a self-injected MessageEvent(internal=True) turn — the only
# machinery kind the gateway produces; only these may vanish on a bare silence marker.
INTERNAL_NOTIFICATION_DISPLAY_KIND = "internal_notification"
MACHINERY_DISPLAY_KINDS = frozenset({INTERNAL_NOTIFICATION_DISPLAY_KIND})

# Longer than any marker could plausibly be, even with stray punctuation.
_MARKER_LENGTH_CAP = 64


def _canonical_silence_candidate(text: str) -> str:
    return " ".join(text.strip().upper().split())


def _is_edge_punctuation(ch: str) -> bool:
    # Square brackets stay structural so malformed ``[SILENT`` cannot become ``SILENT``.
    return ch not in "[]" and unicodedata.category(ch).startswith("P")


def _strip_edge_silence_punctuation(text: str) -> str:
    """Strip stray edge punctuation (``.NO_REPLY``, ``*NO_REPLY*``) without erasing marker structure."""
    start, end = 0, len(text)
    while start < end and _is_edge_punctuation(text[start]):
        start += 1
    while end > start and _is_edge_punctuation(text[end - 1]):
        end -= 1
    return text[start:end].strip()


def _canonical_silence_candidates(text: Any) -> tuple[str, ...]:
    """Canonical forms of a short marker-sized response; ``()`` when not a candidate at all."""
    stripped = text.strip() if isinstance(text, str) else ""
    if not 0 < len(stripped) <= _MARKER_LENGTH_CAP:
        return ()
    depunctuated = _strip_edge_silence_punctuation(stripped)
    forms = (stripped,) if depunctuated == stripped else (stripped, depunctuated)
    return tuple(_canonical_silence_candidate(f) for f in forms)


def is_intentional_silence_response(response: Any) -> bool:
    """True when ``response`` is an intentional-silence signal.

    Two cases suppress delivery:
      1. The whole canonicalized response is an exact silence marker
         (``NO_REPLY`` / ``[SILENT]`` / ...), tolerant of stray edge
         punctuation. Prose that merely mentions a marker is delivered
         normally; a blank response is the empty-response failure path, not
         silence.
      2. The model narrated its stay-silent DECISION instead of emitting the
         token (leaked reply-gate reasoning such as "Banter grupal, no dirigido
         a mí. Me quedo callado.") — a high-precision backstop, see
         ``_is_leaked_silence_narration``.
    """
    if any(c in LIVE_GATEWAY_SILENT_MARKERS for c in _canonical_silence_candidates(response)):
        return True
    return isinstance(response, str) and _is_leaked_silence_narration(response)


def suppression_reason(text: Any) -> Optional[str]:
    """Return a short reason string if ``text`` must NOT be delivered.

    Returns ``None`` when the text is safe to deliver. The reason is a stable
    token intended for logs and tests, never for display:

        ``"empty"``               — nothing (or only whitespace) to send
        ``"silence_marker"``      — an explicit ``[SILENT]`` decision
        ``"silence_narration"``   — the model *described* staying silent
        ``"tool_mechanism"``      — the model narrated its own tooling

    Deliberately NOT included: failure-alert routing. That is a *destination*
    decision (send the alert to the operator instead of the contact), not a
    suppression decision, and lives with the caller that knows the job.
    """
    if not isinstance(text, str):
        return "empty"
    stripped = text.strip()
    if not stripped:
        return "empty"
    if is_intentional_silence_response(stripped):
        return "silence_marker"
    if _is_leaked_silence_narration(stripped):
        return "silence_narration"
    if is_leaked_tool_mechanism_narration(stripped):
        return "tool_mechanism"
    return None


def is_autonomous_silence_response(response: Any) -> bool:
    """Loose silence matcher for autonomous lanes (cron, webhook).

    Models reliably bracket ``[SILENT]`` with a short note, so unlike the
    interactive EXACT rule this also suppresses when a marker sits on its own
    first/last line or the bracketed sentinel opens the response (``[SILENT] No
    changes detected``).  A token buried mid-sentence is still delivered.
    Shares :data:`LIVE_GATEWAY_SILENT_MARKERS` so the two sets cannot drift.
    """
    stripped = response.strip() if isinstance(response, str) else ""
    if not stripped:
        return False
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    # Bracketed form only for the prefix rule, so a bare "Silent retry succeeded" is NOT swallowed.
    # Same de-punctuating forms as the interactive rule, so ``【静默】`` / ``静默。`` cannot
    # be suppressed in chat yet delivered by cron.
    return stripped.upper().startswith(_BRACKETED_SILENCE_MARKERS) or any(
        is_intentional_silence_response(c) for c in (stripped, lines[0], lines[-1])
    )


def is_intentional_silence_agent_result(agent_result: dict | None, response: Any) -> bool:
    """Silence markers suppress delivery only for successful agent turns."""
    return isinstance(agent_result, dict) and not agent_result.get("failed") and is_intentional_silence_response(response)


def display_kind_for_event(event: Any) -> str | None:
    """The persisted user-row kind for a gateway turn: only self-injected events are machinery.

    A scheduled heartbeat prompt is self-injected too (``_heartbeat_session_id`` is stamped only
    by the gateway poller, never inferred from inbound text), but it deliberately stays
    non-internal so authorization and the emergency stop still apply to it.
    """
    if getattr(event, "internal", False) or getattr(event, "_heartbeat_session_id", None):
        return INTERNAL_NOTIFICATION_DISPLAY_KIND
    return None


def is_machinery_display_kind(display_kind: Any) -> bool:
    """Only a machinery turn may vanish on a bare silence marker; a human turn gets a visible fallback.

    The caller passes the current turn's persisted display kind instead of inferring it from the
    transcript: the inbound user row is not persisted yet, and a previous internal row must never
    authorize silence on a human turn.
    """
    return display_kind in MACHINERY_DISPLAY_KINDS


def is_partial_silence_marker(text: Any) -> bool:
    """True while streamed ``text`` could still resolve to a silence marker.

    A buffer whose canonical form is a non-empty *prefix* of a marker (``"NO"`` on
    the way to ``"NO_REPLY"``, or an exact marker not yet terminated by stream-end)
    is held back so a raw marker is never shown and then retracted.  Divergence
    from every marker, or exceeding the cap, resumes normal streaming.
    """
    return any(
        c and any(marker.startswith(c) for marker in LIVE_GATEWAY_SILENT_MARKERS)
        for c in _canonical_silence_candidates(text)
    )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

SILENT_REPLY_TOKEN = "NO_REPLY"
# ---- END PLUGIN-COMPAT ----


def markdown_to_prose(text: Any) -> str:
    """Strip cosmetic Markdown so register guards judge prose, not formatting.

    Order matters: thematic breaks and block prefixes are line-anchored and run
    first, then links collapse to their labels, then inline emphasis. Every
    removal substitutes a SPACE rather than deleting, so two adjacent tokens can
    never be glued into a word that was not in the original
    (``**Sam**Lee`` must not become ``SamLee``).

    Verdict-input only — see the module comment above. Never use this to
    transform text that will actually be delivered.
    """
    if not isinstance(text, str):
        return ""
    out = _THEMATIC_BREAK.sub(" ", text)
    out = _BLOCK_PREFIX.sub(" ", out)
    out = _LINK.sub(r" \1 ", out)
    out = _REF_LINK.sub(r" \1 ", out)
    return _EMPHASIS_RUN.sub(" ", out)

def mentions_recipient_third_person(text: Any, recipient_name: Any, aliases: Any = None) -> bool:
    """True when a reply refers to its OWN recipient in the third person.

    ``aliases`` (optional) is an extra iterable of names/nicknames to test
    ALONGSIDE ``recipient_name`` — the defense-in-depth fix (2026-09-09) for
    silos whose stored display name is not a usable first name: a pushname
    ("Hey!"=Dana), a nickname that differs ("Sunny" vs "Sunflowerpop 🌸…"), an
    emoji-glued name ("Morgan🦉"), a handle ("jdoe88"=Jane), or a
    2-char token ("P."=Robin). Each alias is run through the SAME third-person
    frame logic, so passing "Dana" as an alias re-arms this guard for a silo
    the display name alone left blind. Empty/short aliases are ignored exactly
    as the primary name is.

    The 2026-08-22 leak: HAL, mid a heavy report-to-owner session, wrote a
    status report about the contact ("la TV de Dana no reproduce…", "Le digo
    que intente de nuevo") and delivered it straight TO Dana. The real tell is
    REGISTER: an owner-report is third-person narration ABOUT the contact with
    no second-person address TO them. So the guard's primary discriminator is
    grammatical person, not the recipient's name:

      0. if the text addresses the reader in the SECOND person (tú/usted/te/
         dijiste-style/you/your), it is a direct reply → never a leak → False.
         This is the root-cause check; it retires the brittle name+punctuation
         proxy that flagged real replies like "Directo, Sam, porque tú mismo…".

    Only when nobody is addressed does the name fallback run (a report that
    merely NAMES the contact, e.g. "la tele de Dana no reproduce"):
      1. take the recipient's FIRST name only (display names carry surnames the
         reply never uses);
      2. reduce the text to PROSE (``markdown_to_prose``) so formatting can
         never sit between a clause anchor and the name — see the bug-class
         comment above that function;
      3. flag ONLY when the name sits inside a THIRD-PERSON FRAME
         (``_third_person_frames``): a preposition/article in front of it
         ("la tele *de* Dana", "le avisé *a* Dana", "*con* Dana") or the
         name opening a clause as the SUBJECT of an agent verb ("Dana *no
         puede* reproducir", "Lee *pidió* el video", "Sam *dice que*").

    DESIGN — default-PASS, not default-flag (2026-09-05). The previous version
    did the inverse: it stripped the vocative shapes it *recognised* (a fixed
    greeting list, comma/colon/dash separators) and flagged any name that
    survived. That makes the false-positive rate proportional to the size of
    an OPEN set — every greeting, interjection, or separator nobody enumerated
    ("Holaaa Lee 👋 ¿Qué se cuenta?", "¡Lee! ¿Qué onda?", "Buenas
    Lee 🙌", "Jajaja Lee…") was held as an owner-report. Measured:
    70% of a greeting×separator×tail matrix was held. The 2026-08-31 Markdown
    incident was the same flaw with a different open set. Third-person frames
    are a CLOSED set (grammar: prepositions, articles, clitics, a bounded verb
    list), so keying on them makes a new greeting/emoji/punctuation unable to
    change a verdict by construction. Ambiguous shapes ("Lee ya quedó" —
    vocative or subject?) deliberately PASS: this guard blocks delivery, so it
    is the precision half; recall belongs to the async semantic auditor
    (``gateway/register_audit.py``).

    Returns False on empty/short names (< 3 chars: initials, ambiguous).
    """
    if not isinstance(text, str) or not isinstance(recipient_name, str):
        return False
    # Root-cause discriminator: a second-person address is a reply, not a report.
    if addresses_recipient_second_person(text):
        return False
    # Candidate first-names to test: the stored display name's first token, plus
    # any explicit aliases (nicknames/real names) for silos whose display name is
    # not a usable name. Each is reduced to its first token and length-filtered.
    candidates = [recipient_name]
    if aliases:
        for a in aliases:
            if isinstance(a, str):
                candidates.append(a)
    firsts = []
    for name in candidates:
        tok = name.strip().split()[0] if name.strip() else ""
        if len(tok) >= 3:
            firsts.append(tok)
    if not firsts:
        return False
    # Judge prose: headings/quotes/bullets/links/emphasis must not decide this.
    prose = markdown_to_prose(text)
    for first in firsts:
        oblique, subject = _third_person_frames(first)
        if oblique.search(prose) is not None or subject.search(prose) is not None:
            return True
    return False

# Tokens that never stand in for a person's usable first name: honorifics and
# generic address words. A camel/handle split that yields only these is dropped.
_ALIAS_STOPWORDS = frozenset({
    "don", "dona", "sr", "sra", "senor", "senora", "lic", "ing", "dr", "dra",
    "mtro", "mtra", "profe", "mr", "mrs", "ms", "the", "el", "la",
})

# A run of separators, digits, or non-letter noise between name fragments.
_ALIAS_SEP_RE = re.compile(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ]+")
# camelCase / PascalCase boundary and letter/digit boundary, for handle splits.
_ALIAS_CAMEL_RE = re.compile(r"(?<=[a-zà-öø-ÿ])(?=[A-ZÀ-Ö])")


def _strip_symbols_and_marks(value: str) -> str:
    """Drop emoji, pictographs, and combining marks, keeping letters/spaces.

    Generic Unicode-category filter — no per-recipient data. Symbol (``S*``),
    combining-mark (``M*``), and private-use (``Co``) codepoints become spaces,
    so an emoji-glued or emoji-wrapped display name reduces to its letters.
    """
    out = []
    for ch in value:
        cat = unicodedata.category(ch)
        if cat[0] in ("S", "M") or cat == "Co":
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def recipient_aliases_for(chat_id: Any, display_name: Any = None) -> tuple[str, ...]:
    """Derive candidate first-name aliases for a recipient, for the leak guard.

    Zero private data lives in this module: the aliases are *derived generically*
    from the ``display_name`` present at runtime, not looked up from a hardcoded
    id→name table. This re-arms the third-person guard for silos whose stored
    display name is not a clean first name — a pushname, a handle, a nickname, an
    emoji-glued or emoji-wrapped name, or a "P. Robin"-style abbreviated token —
    by extracting the usable letter tokens the name still contains:

      1. strip emoji / pictographs / combining marks (``_strip_symbols_and_marks``);
      2. split on separator noise AND camelCase/handle boundaries;
      3. keep tokens of length ≥ 3 that are not honorifics/address stopwords.

    ``chat_id`` keys the operator's ``privacy.recipient_aliases`` table (config, never
    source): names a recipient is addressed by that cannot be derived from the stored
    name. No recipient identifiers or names are embedded in source. The stored
    ``display_name`` is already tested by the guard directly, so tokens equal to
    the trimmed display name are not re-emitted. Returns a de-duplicated tuple,
    or ``()`` when nothing usable can be derived.
    """
    if not isinstance(display_name, str) or not display_name.strip():
        return _configured_recipient_aliases(chat_id)
    cleaned = _strip_symbols_and_marks(display_name)
    raw_tokens: list[str] = []
    for chunk in _ALIAS_SEP_RE.split(cleaned):
        if not chunk:
            continue
        raw_tokens.extend(t for t in _ALIAS_CAMEL_RE.split(chunk) if t)
    primary = display_name.strip()
    seen: set[str] = set()
    aliases: list[str] = []
    for tok in raw_tokens:
        if len(tok) < 3 or tok.lower() in _ALIAS_STOPWORDS:
            continue
        if tok == primary:
            continue
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        aliases.append(tok)
    # Operator-declared names the recipient is addressed by (privacy.recipient_aliases).
    # Derivation cannot recover a name that is absent from the stored display name
    # (stored "Morgan", addressed "Robin"); only the operator knows that mapping.
    for tok in _configured_recipient_aliases(chat_id):
        if tok == primary or tok.lower() in seen:
            continue
        seen.add(tok.lower())
        aliases.append(tok)
    return tuple(aliases)


def _configured_recipient_aliases(chat_id: Any) -> tuple[str, ...]:
    """Names from ``privacy.recipient_aliases`` for ``chat_id`` (config, never source).

    Shape: ``privacy.recipient_aliases: {"<chat id>": ["Name", ...]}``; a bare string
    value is accepted as a one-name list. Read fresh per call, like
    ``privacy.owner_trusted_chats``, so edits apply without a restart. Empty by default,
    and any config error yields ``()``: the aliases only widen the name guard and give
    the judges context, so a missing table degrades to derivation alone, never to a
    wider leak.
    """
    key = str(chat_id or "").strip()
    if not key:
        return ()
    try:
        from gateway.run import _load_gateway_config

        table = (_load_gateway_config().get("privacy") or {}).get("recipient_aliases") or {}
        raw = table.get(key) if isinstance(table, dict) else None
    except Exception:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(s for s in (str(v).strip() for v in raw) if s)


def recipient_label_for_judge(chat_id: Any, display_name: Any) -> str:
    """How the semantic register judges are told who the recipient is.

    The judges only see a name. When the stored name differs from the name the
    recipient is actually addressed by (stored "Morgan", addressed "Robin"), a
    judge reads "¡Perfecto, Robin!" as a message ABOUT a third person and votes LEAK,
    stably, so re-judging cannot rescue it. Give the judges the same alias knowledge
    the deterministic name guard uses (recipient_aliases_for).
    """
    name = str(display_name or "").strip()
    aliases = [a for a in recipient_aliases_for(chat_id, name) if a.lower() != name.lower()]
    if not aliases:
        return name
    also = ", ".join(aliases)
    return f"{name} (also addressed as: {also})" if name else also


# ──────────────────────────────────────────────────────────────────────────
# Re-homed from the local layer (2026-09-17 structured port): a custom tool-call exercise
# ──────────────────────────────────────────────────────────────────────────

_HONORIFIC = r"(?:(?:don|doña|dona|sr\.?|sra\.?|señor|señora|senor|senora|lic\.?|ing\.?|dr\.?|dra\.?|mtro\.?|mtra\.?|profe|mr\.?|mrs\.?|ms\.?)[ \t]+)?"

_ES_OBLIQUE = (
    r"\b(?:de|del|a|al|con|para|por|sobre|sin|hacia|desde|según|segun|ante|"
    r"entre|contra|tras|hasta|el|la|los|las|este|esta|ese|esa|aquel|aquella)"
)

_EN_OBLIQUE = (
    # Prepositions/determiners, plus English verbs that take a PERSON as direct
    # object with no preposition ("notifies Robin", "told Lee", "ping
    # Sam"). Spanish needs no such list: the personal "a" is mandatory before
    # a human object, so "a Lee" already covers "avisarle a Lee".
    r"\b(?:to|for|with|about|from|of|at|towards?|regarding|the|this|that|"
    r"notif(?:y|ies|ied|ying)|tell(?:s|ing)?|told|ask(?:s|ed|ing)?|"
    r"remind(?:s|ed|ing)?|ping(?:s|ed|ing)?|messag(?:e|es|ed|ing)|"
    r"text(?:s|ed|ing)?|email(?:s|ed|ing)?|call(?:s|ed|ing)?|"
    r"contact(?:s|ed|ing)?|inform(?:s|ed|ing)?|let|send(?:s|ing)?|sent|"
    r"giv(?:e|es|ing)|gave|help(?:s|ed|ing)?|show(?:s|ed|ing)?|"
    r"updat(?:e|es|ed|ing)|thanked|nudg(?:e|es|ed|ing)|"
    r"warn(?:s|ed|ing)?|alert(?:s|ed|ing)?|answer(?:s|ed|ing)?|"
    r"reach(?:es|ed|ing)?|forward(?:s|ed|ing)?[ \t]+(?:it[ \t]+)?to)"
)

_ES_ADV_CLI = (
    r"(?:no|ya|nunca|jamás|jamas|también|tambien|tampoco|todavía|todavia|aún|"
    r"aun|siempre|apenas|casi|ni|solo|sólo|me|se|le|lo|la|nos|les|los|las)"
)

_ES_CLI = r"(?:me|se|le|lo|la|nos|les|los|las)"

_ES_V3_BARE = (
    r"puede(?![ \t]+(?:ser|que)\b)|quiere(?![ \t]+decir\b)|dice|"
    r"tiene(?![ \t]+(?:que|sentido|caso|chiste|remedio)\b)|necesita|sabe|"
    r"prefiere|suele|ha(?![ \t]+(?:sido|habido|estado(?![ \t]+\w+ndo\b))\b)|"
    r"pidió|pidio|dijo|contestó|contesto|respondió|respondio|mandó|mando|"
    r"envió|envio|pagó|pago|confirmó|confirmo|avisó|aviso|preguntó|pregunto|"
    r"comentó|comento|escribió|escribio|marcó|marco|llamó|llamo|aceptó|acepto|"
    r"rechazó|rechazo|prefirió|prefirio|entendió|entendio|creyó|creyo|pensó|"
    r"penso|recibió|recibio|solicitó|solicito|reportó|reporto|mencionó|"
    r"menciono|insistió|insistio|logró|logro|consiguió|consiguio|quiso|"
    r"pudo(?![ \t]+ser\b)|tuvo(?![ \t]+que\b)|supo|vino|vio|leyó|leyo|olvidó|"
    r"olvido|tomó|tomo|usó|uso|decidió|decidio|canceló|cancelo|compró|compro|"
    r"vendió|vendio|activó|activo|borró|borro|eliminó|elimino|instaló|instalo|"
    r"configuró|configuro|agregó|agrego|entregó|entrego|firmó|firmo|autorizó|"
    r"autorizo|aprobó|aprobo|negó|nego|reclamó|reclamo|agradeció|agradecio|"
    r"saludó|saludo|hizo|puso|trajo|llevó|llevo|dejó(?![ \t]+de\b)|"
    r"dejo(?![ \t]+de\b)|subió|subio|bajó|bajo|descargó|descargo|abrió|abrio|"
    r"cerró|cerro|depositó|deposito|transfirió|transfirio|renovó|renovo"
)

_ES_V3_NEGATED = (
    r"pide|recibe|contesta|responde|manda|envía|envia|paga|confirma|avisa|"
    r"pregunta|entiende|cree|piensa|acepta|lee|logra|consigue|viene|usa|"
    r"quiere|puede|tiene|necesita|sabe|dice|ha|prefiere|escribe|marca|llama|"
    r"comenta|reporta|menciona|insiste|solicita|firma|entrega|autoriza|aprueba"
)

_EN_ADV = r"(?:also|already|still|never|just|now|then|only|\w+ly)"

_EN_V3 = (
    r"is(?![ \t]+(?:it|this|that|there)\b)|was(?![ \t]+(?:it|this|that|there)\b)|"
    r"has|hasn't|says|said|wants|wanted|needs|needed|asked|asks|told|tells|"
    r"doesn't|didn't|isn't|wasn't|gets|got|sent|seems|seemed|keeps|kept|"
    r"replied|confirmed|paid|requested|mentioned|reported|prefers|thinks|"
    r"thought|knows|knew"
)

@functools.lru_cache(maxsize=256)
def _third_person_frames(first_name: str) -> "tuple[re.Pattern, re.Pattern]":
    """Compile the (oblique, subject) frame patterns for one first name."""
    name = re.escape(first_name)
    oblique = re.compile(
        rf"(?:{_ES_OBLIQUE}|{_EN_OBLIQUE})[ \t]+{_HONORIFIC}{name}\b|\b{name}'s\b",
        re.IGNORECASE,
    )
    subject = re.compile(
        rf"\b{name}[ \t]+"
        rf"(?:(?:{_ES_ADV_CLI}[ \t]+)*(?:{_ES_V3_BARE})\b"
        rf"|(?:{_ES_ADV_CLI}[ \t]+)*(?:no|nunca|jamás|jamas|tampoco|ni)[ \t]+"
        rf"(?:{_ES_CLI}[ \t]+)*(?:{_ES_V3_NEGATED})\b"
        rf"|(?:{_EN_ADV}[ \t]+)*(?:{_EN_V3})\b)",
        re.IGNORECASE,
    )
    return oblique, subject

_REPORT_LABEL_HEADING = re.compile(
    r"(?mi)^\s*[*_]{1,2}\s*"
    r"(?:done|to-?do|pending|status|completed|complete|in[ -]?progress|"
    r"blockers?|next[ -]?steps?|summary|results?|what[ '’]?s?[ ]?(?:i[ ]?)?did|"
    r"what[ '’]?s?[ ]?(?:i[ ]?)?done|remaining|notes?|recap|"
    r"hecho|hechos|pendiente|pendientes|estado|completado|completados|"
    r"en[ ]?(?:curso|proceso)|resumen|resultados?|siguientes[ ]?pasos|"
    r"lo[ ]?que[ ]?(?:hice|falta|queda)|bloqueos?|por[ ]?hacer|notas?)"
    r"\s*:?\s*[*_]{1,2}\s*:?\s*$"
)

_CHECKLIST_BULLET = re.compile(
    r"(?m)^\s*(?:[-*•·]\s*)?[✅✔☑❌✗🔲⏳🔄☐]\s*\S"
)

_OWNER_REPORT_PHRASE = re.compile(
    r"(?i)(?:"
    r"everything\s+is\s+set\s+up\s+and\s+running\s+autonomously|"
    r"here'?s?\s+where\s+things?\s+stand|"
    r"running\s+autonomously|"
    r"no\s+requiere\s+(?:ninguna\s+)?acci[oó]n(?:\s+(?:adicional|de\s+tu\s+parte|de\s+mi\s+parte))?|"
    r"no\s+(?:se\s+)?requiere\s+intervenci[oó]n|"
    r"(?:le|les)\s+avisar[eé]\s+cuando\s+(?:termine|acabe|est[eé]\s+list[oa])|"
    r"queda\s+corriendo\s+(?:en\s+)?(?:segundo\s+plano|de\s+forma\s+aut[oó]noma)|"
    r"de\s+forma\s+aut[oó]noma|"
    r"todo\s+(?:est[aá]\s+)?(?:configurado|listo)\s+y\s+corriendo"
    r")"
)

def is_owner_report_scaffold(text: Any) -> bool:
    """True when ``text`` has the STRUCTURE of an owner-facing status report.

    Name-independent, high-precision backstop for the delivery-blocking path.
    Fires only on scaffolding a genuine chat reply does not carry:

      * a *bolded* status-label heading ("*Done:*", "*Pendiente:*"), OR
      * TWO OR MORE checkmark/status bullets (a checklist board), OR
      * an autonomy / report-to-owner phrase ("running autonomously",
        "no requiere acción de mi parte").

    A warm reply with a single "listo ✅" or one emoji trips none of these.

    ROOT-CAUSE GATE (2026-09-10, Robin grammar-lesson false positive). An
    owner-report is BY DEFINITION third-person narration NOT addressed to the
    contact — so a reply that addresses its reader in the second person is a
    genuine reply and can never be one, whatever glyphs it happens to carry.
    This is the SAME discriminator ``mentions_recipient_third_person`` applies
    internally (see its step 0); it is hoisted here so the structural sibling
    honours it too and the class cannot reappear through a different marker.
    The tell: a teaching reply ("✅ She *went* out." / "❌ She *didn't went*
    out.", "¿Quieres que te ponga unas frasecitas y me las contestas?") uses
    ✅/❌ as CORRECT/INCORRECT answer markers and reads exactly like a ≥2-bullet
    status board — but it talks TO the reader, so it is not a report.
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    # The bold-label heading and the ≥2-bullet board are the AMBIGUOUS markers:
    # both are ordinary structure in a long second-person reply. 2026-09-10:
    # a grammar lesson used ✅/❌ as CORRECT/INCORRECT glyphs. 2026-09-11: a
    # tuition comparison to Robin carried a "*Resumen*" section header over
    # cost bullets and was held — verdict flipped by a heading label with the
    # register unchanged ("Si quieres te las redacto"). Both are gated behind
    # the root-cause discriminator: a reply that addresses its reader in the
    # second person is a genuine reply, never an owner-report, whatever
    # scaffolding it carries. Only the autonomy phrase stands on its own — a
    # mechanism narration ("corriendo de forma autónoma… te aviso") is still a
    # leak even though it says "te".
    if _REPORT_LABEL_HEADING.search(stripped) or len(_CHECKLIST_BULLET.findall(stripped)) >= 2:
        if not addresses_recipient_second_person(stripped):
            return True
    if _OWNER_REPORT_PHRASE.search(stripped):
        return True
    return False



# ---------------------------------------------------------------------------
# Reply-gate leak backstops: second-person address discriminator + leaked
# stay-silent narration + leaked tool-mechanism narration. Ported for the
# tool-gated reply layer (issue 921). These screen a would-be reply that is
# actually the model's own decision-log / mechanism voice leaking into chat.
# ---------------------------------------------------------------------------

_SECOND_PERSON_ADDRESS = re.compile(
    # Whole-word 2nd-person markers a reply TO someone carries and an
    # owner-report ABOUT them structurally cannot: subject/oblique pronouns,
    # possessives, the clitic "te", and English you/your. Word-boundaried so
    # "intente"/"cuente" don't match the "te" clitic.
    r"\b(?:t[úu]|usted(?:es)?|vos(?:otros)?|ustedes|ti|contigo|te|tu|tus|"
    r"tuy[oa]s?|vuestr[oa]s?|you|your|yours|you're|u're)\b",
    re.IGNORECASE,
)

def addresses_recipient_second_person(text: Any) -> bool:
    """True when the text addresses its reader directly in the second person.

    A message written TO someone ("¿ya terminaste?", "tú mismo lo pusiste",
    "your account") is a genuine reply — never the owner-report register the
    third-person guard exists to catch. This is the real invariant; the name
    heuristic below is only a fallback for messages that name the contact
    without addressing anyone.
    """
    if not isinstance(text, str):
        return False
    # Judge PROSE, not markup: "**tú** mismo" addresses the reader exactly like
    # "tú mismo", and markup between the \b anchors must not hide the pronoun.
    return _SECOND_PERSON_ADDRESS.search(markdown_to_prose(text)) is not None


_REPLY_GATE_JARGON_RE = re.compile(r"\breply[\s\-]?gate\b", re.IGNORECASE)

# Tier A — first-person / definitive stay-silent decision (the leaked CoT voice).
# Includes Spanish equivalents: the model's decision-log voice follows the chat
# language, so a Spanish-chat leak ("Silencio deliberado", "nada que responder",
# "no respondo") must be caught too — English-only tells let real leaks through.
_SILENCE_DECISION_TIER_A_RES = (
    re.compile(r"\b(?:i|i'?ll|i\s+will)\s+(?:stay|remain)\s+(?:silent|quiet)\b", re.IGNORECASE),
    re.compile(r"\bno\s+message\s+(?:sent|is\s+sent)\b", re.IGNORECASE),
    re.compile(r"\bsent\s+no\s+message\b", re.IGNORECASE),
    re.compile(r"\b(?:nobody|no\s+one)\s+addressed\s+me\b", re.IGNORECASE),
    re.compile(r"\bnot\s+addressed\s+to\s+me\b", re.IGNORECASE),
    re.compile(r"\bwas\s*n'?t\s+addressed\b", re.IGNORECASE),
    re.compile(r"\b(?:i\s+)?(?:do\s+not|don'?t|won'?t|will\s+not)\s+reply\b", re.IGNORECASE),
    re.compile(r"\b(?:i\s*'?\s*m|i\s+am)\s+not\s+(?:reply|respond)ing\b", re.IGNORECASE),
    re.compile(r"\bi\s+stay\s+silent\b", re.IGNORECASE),
    # Impersonal-English reply-decision narration: the model reports its own
    # no-reply verdict without a first-person "I", so the patterns above miss it.
    # "No reply needed", "no action pending/required", "ending quietly", "no
    # response needed" are decision-log phrasing — a real message to a human
    # never announces that no reply is needed.
    re.compile(r"\bno\s+(?:reply|response)\s+(?:needed|required|necessary|warranted)\b", re.IGNORECASE),
    re.compile(r"\bno\s+action\s+(?:needed|required|pending|necessary)\b", re.IGNORECASE),
    re.compile(r"\bending\s+quietly\b", re.IGNORECASE),
    re.compile(r"\b(?:staying|stay)\s+quiet\b", re.IGNORECASE),
    # Spanish first-person / definitive stay-silent decision.
    re.compile(r"\bsilencio\s+deliberado\b", re.IGNORECASE),
    re.compile(r"\bnada\s+(?:que|a)\s+responder\b", re.IGNORECASE),
    re.compile(r"\bnada\s+accionable\s+que\s+responder\b", re.IGNORECASE),
    re.compile(r"\b(?:no\s+respondo|no\s+voy\s+a\s+responder|no\s+contesto)\b", re.IGNORECASE),
    re.compile(r"\b(?:me\s+)?(?:quedo|permanezco)\s+(?:en\s+)?silencio\b", re.IGNORECASE),
    # "Callado/callada" is the other half of the Spanish idiom and was missing entirely:
    # "Me quedo callado", "mejor me quedo callada", "me mantengo callado". The patterns above
    # only knew the noun "silencio", so this whole phrasing scored ZERO signals on every tier
    # and shipped verbatim into a live group.
    re.compile(r"\b(?:me\s+)?(?:quedo|mantengo|permanezco)\s+(?:en\s+)?callad[oa]\b", re.IGNORECASE),
    # "No dirigido a mí" / "no va dirigido a mí" — the reason-half of the same decision log,
    # the Spanish twin of "not addressed to me", which Tier A already carries in English.
    re.compile(r"\bno\s+(?:va\s+|iba\s+|está\s+|estaba\s+)?dirigid[oa]\s+a\s+m[ií]\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:me\s+)?(?:mencion|interpel|aludi|invoc)\w*\s*(?:a\s+m[ií])?\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:hay\s+)?mensaje\s+(?:que\s+)?(?:enviar|entregar)\b", re.IGNORECASE),
)

# Tier B — contextual support; never decisive on its own.
_SILENCE_DECISION_TIER_B_RES = (
    re.compile(r"\bhuman[\s\-]to[\s\-]human\b", re.IGNORECASE),
    re.compile(r"\bobvious\s+recipient\b", re.IGNORECASE),
    re.compile(r"\bstay(?:ing)?\s+silent\b", re.IGNORECASE),
    re.compile(r"\bremain\s+silent\b", re.IGNORECASE),
    re.compile(r"\bambient\s+(?:conversation|banter|chatter)\b", re.IGNORECASE),
    # Spanish contextual support.
    re.compile(r"\bsin\s+texto(?:\s+acompa\w+)?\b", re.IGNORECASE),
    re.compile(r"\b(?:reacci[oó]n|reacciona\w*)\s+\S+\s+de\s+\w+", re.IGNORECASE),
    re.compile(r"\b(?:banter|charla|conversaci[oó]n)\s+(?:entre|ambient\w*)\b", re.IGNORECASE),
    re.compile(r"\bentre\s+(?:otras\s+)?personas\b", re.IGNORECASE),
)

# The verdict-as-LABEL leak. The Tier-A patterns above all look for the decision
# as a *sentence* ("no respondo", "me quedo en silencio"). But the leaked
# decision-log just as often states the verdict as a bare LABEL wrapping the
# reasoning, and that shape slipped three leaks into a live group in one session:
#
#   "Silencio: Bruno etiquetó a Ana, no a mí. …"       <- prefix label
#   "… responder solo pesaría. Silencio."              <- terminal verdict
#
# Both are structural, not lexical: the giveaway is the verdict word standing
# ALONE in verdict POSITION (opening label before the reason, or the final word
# closing the explanation), never as a noun inside a sentence. Matching the
# position rather than the word keeps "el silencio de los corderos" and "un
# minuto de silencio" deliverable — they never put the bare word in either slot.
_SILENCE_VERDICT_WORD = r"(?:silencio|silence)"

# Opening label: "Silencio:" / "Silencio." / "Silencio —" at the very start,
# followed by the reasoning. Requires the delimiter, so a sentence that merely
# BEGINS with the noun ("Silencio absoluto en la sala") is untouched.
_SILENCE_LABEL_PREFIX_RE = re.compile(
    rf"^\W*{_SILENCE_VERDICT_WORD}\s*(?::|\.|\u2014|-|\u2013)\s+\S",
    re.IGNORECASE,
)

# Terminal verdict: the text ENDS on the bare word as its own final sentence,
# i.e. preceded by sentence-ending punctuation. "…solo pesaría. Silencio."
# matches; "…prefiero el silencio." does not (no sentence break before it).
_SILENCE_VERDICT_SUFFIX_RE = re.compile(
    rf"(?:[.!?\u2026\u2014]|\n)\s*{_SILENCE_VERDICT_WORD}\s*[.!?]?\W*$",
    re.IGNORECASE,
)


def _has_silence_verdict_marker(text: str) -> bool:
    """True when the stay-silent verdict appears as a standalone LABEL — opening
    the message or closing it — rather than as a word inside real prose."""
    return bool(
        _SILENCE_LABEL_PREFIX_RE.search(text)
        or _SILENCE_VERDICT_SUFFIX_RE.search(text)
    )


# The decision-OPENS-the-message shape. The label rule above keys on the verdict
# NOUN ("Silencio: …"); this one keys on the verdict CLAUSE in first position:
#
#   "No respondo — reacción entre ellos."
#   "No respondo — audio al grupo, no va dirigido a mí."
#   "I'm not replying: this is between the two of them."
#
# Structurally identical to the label form — verdict first, reason after — but it
# slipped the Tier-A/B calculus because one Tier-A hit alone needs corroboration
# (tier_a + tier_b >= 3) and a terse "decision + short reason" never reaches
# three.
#
# A genuine reply to a human never OPENS by announcing that it is not a reply:
# the sentence is addressed to no one and answers nothing. So a first-person
# stay-silent decision in first position is decisive — but only when the text
# ALSO addresses nobody. A real answer may legitimately decline one thing and
# then answer ("No respondo por Ana — ella TE dirá; el partido es a las 8"), and
# the tell that it is a real answer is that it speaks TO someone. Length alone is
# not that tell: a terse genuine reply and a terse leak are the same size.
# Second-person address is the discriminator, so the rule keys on it.
def _opens_with_silence_decision(text: str) -> bool:
    """True when the message BEGINS with a first-person stay-silent decision
    and addresses nobody — i.e. it is a verdict, not a reply."""
    if addresses_recipient_second_person(text):
        return False
    for rx in _SILENCE_DECISION_TIER_A_RES:
        m = rx.search(text)
        # Allow only leading quotes/dashes/whitespace before the decision.
        if m and not text[: m.start()].strip(" \t\"'`*_-–—…("):
            return True
    return False


# Only scan short responses — a leaked decision-log is terse. A genuine long
# message that merely mentions silence is never the bug.
_SILENCE_NARRATION_MAX_LEN = 800


def _is_leaked_silence_narration(text: str) -> bool:
    """True when `text` is the model narrating a stay-silent decision instead of
    a real message — high precision so legitimate prose is never suppressed."""
    stripped = text.strip()
    if not stripped or len(stripped) > _SILENCE_NARRATION_MAX_LEN:
        return False
    tier_a = sum(1 for rx in _SILENCE_DECISION_TIER_A_RES if rx.search(stripped))
    verdict_label = _has_silence_verdict_marker(stripped)
    opening_decision = _opens_with_silence_decision(stripped)
    if tier_a == 0 and not verdict_label:
        # No first-person stay-silent decision and no verdict label -> not
        # leaked reasoning. An owner explanation of the mechanism (Tier-B /
        # "reply gate" only) lands here and is delivered untouched.
        return False
    tier_b = sum(1 for rx in _SILENCE_DECISION_TIER_B_RES if rx.search(stripped))
    has_jargon = bool(_REPLY_GATE_JARGON_RE.search(stripped))
    return (
        verdict_label                               # verdict stated as a label == decision log
        or opening_decision                         # decision in first position == decision log
        or tier_a >= 2                              # two decision statements == decision log
        or (has_jargon and tier_a >= 1)             # mechanism + a decision == leaked CoT
        or (tier_a >= 1 and tier_a + tier_b >= 3)   # one decision, heavily corroborated
    )


_TOOL_MECHANISM_NARRATION_RES = (
    re.compile(r"\bno\s+`?send_message`?\s+tool\b", re.IGNORECASE),
    re.compile(r"\b(?:nonexistent|non-existent)\s+(?:tool|function)\b", re.IGNORECASE),
    re.compile(r"\binjected\s+[\"']?system[\"']?\s+message\b", re.IGNORECASE),
    re.compile(r"\bintentionally\s+removed\b", re.IGNORECASE),
    re.compile(r"\bnot\s+calling\s+a\s+(?:nonexistent|non-existent)\b", re.IGNORECASE),
    re.compile(r"\bno\s+further\s+action\s+needed\s+here\b", re.IGNORECASE),
    re.compile(r"\bmy\s+(?:actual\s+)?toolset\b", re.IGNORECASE),
    re.compile(r"\bthis\s+platform\s+delivers\s+my\s+plain[\s\-]?text\b", re.IGNORECASE),
    # Cron-agent variants.  These were derived by mining the real session
    # history (state.db) rather than invented: a scheduled job's model,
    # pushed by the reply gate, narrates that it cannot reach a delivery
    # tool and/or that the scheduler will deliver for it.  Such text shipped
    # verbatim to a non-technical third party's chat.
    #
    # Precision note: legitimate engineering replies routinely *mention*
    # send_message (debugging send_message_tool.py, naming a test).  What
    # separates a leak is FIRST PERSON possession/availability language about
    # the agent's own tools, or describing the delivery mechanism as the
    # reason for not acting.  Patterns below target that, not the bare noun.
    re.compile(
        r"\b(?:i\s+)?(?:don'?t|do\s+not|doesn'?t)\s+have\s+(?:a|an|the|access\s+to)\s+"
        r"[`\"']?[\w:.]+[`\"']?\s+tool\b",
        re.IGNORECASE,
    ),
    # "<tool> is not available in this tool environment / in my function set /
    #  in my available toolset / in my schema"
    re.compile(
        r"\b(?:is\s+)?not\s+(?:available|present|included)\s+in\s+(?:this|my|the)\s+"
        r"(?:tool\s+)?(?:environment|function\s+set|toolset|schema|available\s+tools)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\btool\s+available\s+in\s+(?:this|my)\s+(?:environment|function\s+set|toolset)\b", re.IGNORECASE),
    re.compile(r"\bin\s+my\s+(?:available\s+)?(?:toolset|function\s+set|schema)\b", re.IGNORECASE),
    re.compile(r"\bisn'?t\s+in\s+my\s+(?:available\s+)?toolset\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:such\s+)?(?:tool|messaging\s+tool)\s+(?:actually\s+)?exists\b", re.IGNORECASE),
    re.compile(r"\bno\s+messaging\s+tool\s+i\s+can\s+call\b", re.IGNORECASE),
    # Delivery-mechanism narration: the model explaining the scheduler's job.
    re.compile(r"\b(?:cron|scheduler)\s+(?:system\s+)?will\s+deliver\b", re.IGNORECASE),
    re.compile(r"\bauto[\s\-]?delivered\s+by\s+the\s+system\b", re.IGNORECASE),
    re.compile(
        r"\b(?:is|are|will\s+be)\s+(?:auto(?:matically)?[\s\-]?)?delivered\s+"
        r"(?:automatically\s+)?to\s+(?:the|that|its)\s+(?:job'?s\s+)?configured\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bconfigured\s+(?:target|destination)\b", re.IGNORECASE),
    re.compile(r"\bthis\s+is\s+a\s+(?:scheduled\s+)?cron\s+(?:job|delivery\s+context)\b", re.IGNORECASE),
    re.compile(r"\bcron\s+delivery\s+context\b", re.IGNORECASE),
    re.compile(r"\bper\s+the\s+original\s+(?:job\s+)?instructions\b", re.IGNORECASE),
    re.compile(r"\bmy\s+original\s+instructions\s+were\b", re.IGNORECASE),
    re.compile(r"\b(?:to\s+)?satisfy\s+this\s+gate\b", re.IGNORECASE),
    re.compile(r"\bstands\s+as\s+(?:that|the\s+final)\s+response\b", re.IGNORECASE),
    re.compile(r"\bis\s+the\s+final\s+deliverable\b", re.IGNORECASE),
    re.compile(r"\b(?:cannot|can'?t)\s+deliver\s+this\s+to\b", re.IGNORECASE),
    # --- Spanish equivalents -------------------------------------------------
    # The agent answers in the language of the chat, so a Spanish-speaking
    # deployment leaks in Spanish and an English-only pattern set is blind to
    # it (observed 2026-07-26: "No tengo acceso a una función `send_message`
    # en este entorno..." passed a 27-pattern English filter untouched).
    # Precision guard: "no tengo acceso a tu PC" / "no se puede desde mis
    # herramientas" are legitimate, frequent replies to contacts -- so these
    # patterns require TOOL/DELIVERY-MECHANISM context, never bare "no tengo
    # acceso".
    re.compile(
        r"\bno\s+tengo\s+acceso\s+a\s+(?:una?\s+)?(?:función|funcion|herramienta)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:aparece|figura|est[áa])\s+en\s+mi\s+lista\s+de\s+herramientas\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bherramientas\s+disponibles\s+(?:en\s+)?(?:este\s+)?entorno\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:existe|cuento\s+con)\s+(?:un\s+)?mecanismo\s+de\s+env[íi]o\b", re.IGNORECASE),
    re.compile(r"\bse\s+entrega\s+autom[áa]ticamente\s+al\s+destino\b", re.IGNORECASE),
    re.compile(r"\bdestino\s+configurado\b", re.IGNORECASE),
    re.compile(r"\b(?:mis\s+)?instrucciones\s+originales\s+de\s+sistema\b", re.IGNORECASE),
    re.compile(r"\beste\s+es\s+un\s+cron\s+job\b", re.IGNORECASE),
    re.compile(r"\bmi\s+respuesta\s+(?:final\s+)?(?:de\s+texto\s+)?se\s+entrega\b", re.IGNORECASE),
    re.compile(r"\bno\s+requiere\s+(?:una\s+)?acci[óo]n\s+adicional\s+de\s+mi\s+parte\b", re.IGNORECASE),
)

_TOOL_MECHANISM_NARRATION_MAX_LEN = 800

def is_leaked_tool_mechanism_narration(text: Any) -> bool:
    """True when ``text`` is the model narrating confusion/commentary about
    its own tool availability or the reply-gate mechanism, rather than
    answering the conversation — high precision (>=2 independent hits
    required) so an on-topic reply that legitimately discusses tools/schemas
    is never suppressed."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > _TOOL_MECHANISM_NARRATION_MAX_LEN:
        return False
    hits = sum(1 for rx in _TOOL_MECHANISM_NARRATION_RES if rx.search(stripped))
    return hits >= 2



# ---------------------------------------------------------------------------
# Foundation constants for markdown_to_prose().
# Module-level names resolve at call time, so defining them here is sufficient
# for the helpers defined above. Ported verbatim from the reply-gate source.
# ---------------------------------------------------------------------------
_EMPHASIS_RUN = re.compile(r"[*~`]+|(?<![A-Za-z0-9])_+|_+(?![A-Za-z0-9])")

_BLOCK_PREFIX = re.compile(
    r"^[ \t]*(?:>[ \t]*)*(?:#{1,6}[ \t]+|[-*+][ \t]+|\d+[.)][ \t]+)?",
    re.MULTILINE,
)

_THEMATIC_BREAK = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$", re.MULTILINE)

_LINK = re.compile(r"!?\[([^\]]*)\]\((?:[^)]*)\)")

_REF_LINK = re.compile(r"!?\[([^\]]*)\]\[[^\]]*\]")
