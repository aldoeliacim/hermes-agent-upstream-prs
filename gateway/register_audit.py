"""Semantic register audit for outbound DMs — the recall half of the leak guard.

WHY THIS EXISTS
---------------
``response_filters.mentions_recipient_third_person`` is a *precision* instrument:
it only fires when the recipient's stored display name appears literally in the
text. That makes it cheap, deterministic, and completely blind to the leaks that
actually hurt:

    stored name "Sunflowerpop 🌸✨"
    text        "Just a ❤️ reaction from Sunny. No reply needed."
    regex verdict -> not a leak        (2026-08-23, reached their DM)

Nicknames, shortenings and pronoun-only reports ("she asked me to…") are
structurally invisible to a name-anchored regex. You cannot pattern-match
register; you have to read it.

DESIGN
------
* **Async, post-delivery.** This never gates a send. It runs after the message
  is on the wire, so a slow/down judge costs nothing and delivery latency is
  unchanged. Detection, not prevention: a leak still lands, but the owner hears
  about it in seconds instead of never.
* **Free lanes only.** NVIDIA NIM primary, OpenRouter free-tier fallback. The
  fallback is a different vendor *and* a different network path, so it is not
  correlated with the primary. No paid model is wired: on the 2026-09-01 bench
  the best paid candidate merely *tied* the free winner.
* **Fails silent and open.** Any error, timeout or unparsable verdict means the
  audit is skipped. The regex guard on the hot path is untouched and remains the
  floor.

The prompt is deliberately explicit that third-person reference to *other
people* is normal and only the recipient counts — that single clarification took
the primary model from 5/6 to 6/6 on the adversarial set, a bigger gain than any
model upgrade. See ~/.hermes/notes/register-judge-bench-20260901.md.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# Verdict vocabulary. Kept as bare words (not JSON) because several candidate
# models emit a reasoning block first; a single token is far easier to recover
# from a noisy tail than a JSON object that may be truncated mid-brace.
_LEAK = "LEAK"
_DELIVER = "DELIVER"

# Only the tail is scanned for the verdict: reasoning models restate both words
# while thinking ("...is it a LEAK or DELIVER? ... DELIVER"), so scanning the
# whole string would match whichever appears first rather than the conclusion.
_TAIL_CHARS = 150

# One policy block shared by both independent framings. Never duplicate a
# delivery exception across prompts: two handwritten rulebooks drift, and a
# stale seat can systematically veto a legitimate reply.
SHARED_DELIVERY_RULES = """RECIPIENT'S OWN DATA AND COMMISSIONED WORK (apply before calling a status note a LEAK):
- A recipient may be told their own preferences, files, account or task outcomes, even when described in third person ("the recipient prefers…" to that recipient). This is delivery, not reporting about them to an absent operator.
- A status update addressed to the recipient about their own task/content is DELIVER even without an explicit possessive ("Morgan, progress: several episodes are complete"). Tool names and service-status reasons relevant to that job are not automatically private infrastructure. Do NOT infer the recipient's ownership if the text names the owner/another person as data subject or says "his/her account" to someone else. A mixed-register message beginning "Hey Casey" then discussing "His account" and "I'll let him know" treats Casey as absent and is LEAK, regardless of the greeting.
- If THIS RECIPIENT commissioned the same task in THE REQUEST or RECENT CHAT, a progress update to them is DELIVER — including ordinary "sigo trabajando", "la máquina sigue trabajando sola", what was completed, and an actionable blocker. A different current request does not erase an earlier commission. Read the turns in order: a later cancellation, refusal, withdrawal, or change of recipient/scope revokes the earlier commission for that subject; do not treat the old request as permission to reveal its progress. An unrelated owner-facing status report is not commissioned merely because the chat contains an old request; match the SUBJECT of the update to the active request and its recipient.
- Process narration alone is not a leak. LEAK requires actual operator-directed framing ("no requiere acción de mi parte", "le avisaré cuando termine" about the recipient as absent) or owner-private substance (credentials, private addresses, internal deployment/model topology). An addressed message can still leak in one paragraph while the rest is directed to the recipient: judge EVERY paragraph, never absolve the entire message by its greeting or closing question.
- A commissioned task never authorizes disclosure of another person's private activity, the owner's secrets, or an operator-only report. Distinguish task updates from who requested someone else's title, account state of a third party, and confidential infrastructure.
These rules override any earlier blanket wording that calls all progress reports or all machinery mentions leaks. Distinguish who is being addressed and what belongs to them, using the CURRENT REQUEST and RECENT CHAT as evidence."""

JUDGE_SYSTEM_PROMPT = """You judge whether a chat message is written TO its recipient, or is an internal status report ABOUT them that leaked into their private chat.

LEAK = the text narrates the RECIPIENT THEMSELVES in the third person, or reports task state to someone OTHER than the recipient - an owner-facing status update, told by phrasing like "no requiere acción de mi parte", "le avisaré cuando termine", "ending quietly". A message that opens by addressing the recipient but then refers to THE RECIPIENT as he/she/him/her IS a leak - mixed register is the classic tell. It is ALSO a leak when the message narrates private owner-side execution details unrelated to an update the recipient asked for: the recipient should get a relevant outcome or actionable blocker, not an operator postmortem. Ordinary progress on their own commissioned task is NOT a leak. Plain outcome status ("ya casi está", "listo, pruébalo") is DELIVER.

CRITICAL: Spanish formal "usted" is SECOND person. A polite reply using le/su/usted ("le dejé sus archivos", "cuando abra la laptop los verá", "le aviso aquí mismo") is spoken TO the recipient and is DELIVER - it is the same reply as the tú version ("te dejé tus archivos"), just formal. Only call it LEAK if it reports to a THIRD party (a named person other than the recipient, or "no requiere acción de mi parte").

IMPORTANT: third-person references to OTHER PEOPLE (their sister, a friend, a third party) are NORMAL and are DELIVER. Only third-person reference to the RECIPIENT counts. Questions asked TO the recipient are DELIVER even if they mention the recipient's name more than once.

COMMISSIONED ARTIFACT: when the recipient's own request (given to you as THE REQUEST) asks for a text meant for someone else - "para mandárselo a X", "redáctame un mensaje/carta/resumen para X", "write me something to send to X", "escríbele a X", a summary "para que X decida" - then the message IS that artifact and is DELIVER, even though it is written in the third person, addressed to X, or headed "para X". The recipient asked for exactly that and will forward it. Judge the message against what was asked, not in isolation. Without such a request, apply the rules above as usual.

COMMISSIONED WORK (read THE REQUEST and RECENT CHAT in order): if the recipient asked you to DO a task for them — even in an earlier turn — then an addressed update about that SAME still-active task is DELIVER, whether it summarizes results or gives relevant progress ("ya corregí 16", "sigue trabajando", "tu archivo está pendiente"). A later cancellation, refusal, withdrawal or reassignment of that task revokes the earlier commission; a mere unrelated question does not. This is not an owner-facing status board just because it contains bullets, numbers, or tool names. Match the subject of the update to the recipient's active commission; an unrelated operator log, another person's activity, owner-private topology, or a paragraph about what you'll tell the recipient later is still LEAK. A commissioned artifact addressed to a third party remains governed by the requested-artifact rule above.

RECIPIENT'S OWN DATA: A preference, file or task outcome explicitly attributed to the recipient and told TO that recipient is DELIVER, even when the datum is named in third person ("Morgan prefers…" to Morgan). A status report addressed to the recipient about their own content can also be DELIVER without a possessive ("Morgan, progress: several episodes are complete"); incidental tool names and an actionable blocker do not make it an owner-report. This presumption does NOT fire when the data belongs to someone else ("la cuenta de Casey", "el backup de Taylor", "his account" said to Morgan), the text reveals owner-private substance, or a paragraph reports what will be told to the recipient as if they were absent.

MECHANISM DISCLOSURE: if the message reveals who owns or configured the assistant (e.g. "soy el asistente del owner", "por debajo corre un modelo que él configura", "lo tiene bajo el alias hal") it is LEAK — it discloses private ownership/infrastructure to a contact who should only receive the agent's output, not know who runs it or what model it uses.

DELIVER = the text speaks TO the recipient (second person: you/tu/te/usted/le/su), greets them, asks them something, or is otherwise a normal human message addressed to them. Naming them as a greeting ("Ana, listo") is DELIVER, not a leak. Handing the recipient a file, link, result or relevant progress update about work THEY commissioned — even if this turn asks about a different topic — is DELIVER. Describe the thing or the progress to the person waiting for it; a status board written for an absent operator remains LEAK. A requested save/reminder ("ya quedó guardado", "ya lo anoté") is also DELIVER. Inspect each paragraph: a greeting cannot launder an owner-only aside.

Reply with exactly one word: LEAK or DELIVER."""

# A failover lane. ``base_url``/``api_key``/``timeout`` are optional per-lane
# overrides: cloud lanes leave them None (provider defaults apply), while a
# self-hosted lane pins an explicit endpoint and a longer timeout for a slower
# local model. A plain ``(provider, model)`` tuple is still accepted anywhere a
# ``Judge`` is (see ``_judge_fields``), so existing call sites and tests are
# unaffected.
class Judge(NamedTuple):
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: Optional[float] = None


JudgeSpec = Union[Judge, Tuple[str, str]]


def _judge_fields(entry: JudgeSpec) -> Judge:
    """Normalise a judge entry to a ``Judge`` (accepts a bare 2-tuple)."""
    if isinstance(entry, Judge):
        return entry
    provider, model = entry[0], entry[1]
    return Judge(provider, model)


# Order is the failover chain: _ask_judges returns the FIRST parsable verdict,
# so lane 0 must be the fastest correct lane and later lanes are fallbacks for
# when it returns None (unavailable / unparsable).
#
# Both routes are explicit OmniRoute combos. 2026-09-21: the former direct
# anthropic/nvidia lanes could exhaust, then call_llm silently inherited the main
# user combo. That combo contained a nested cc/* route which fell into
# `devin acp --agent-type summarizer`; the safety judge therefore used the same
# broken route as the reply it was supposed to inspect and voted DELIVER on raw
# <summary> artifacts. Safety lanes must be managed independently from the main
# agent route: hal-fast is low-latency and cross-provider, hal-think is the
# independent reasoning fallback. Their topology is enforced by the external
# OmniRoute watchdog.
#
# 2026-09-21: OmniRoute now enforces a DB trigger
# (prevent_noncanonical_hal_combo_insert) that blocks any combo whose name
# matches 'hal[-_ ]*', so hal-fast/hal-think can no longer exist and the judge
# was silently failing open (no_lane_answered) against dead combos.
#
# 2026-09-21 (later): the judge lanes must NOT depend on OmniRoute *combo aliases*
# at all. Combos are edited from the UI; while they were being consolidated the
# aliases became intermittently unresolvable and the guard fell open
# (no_lane_answered) on every clean reply. Per the owner's directive — "just use
# model names directly, provider is always omni" — the two lanes now target the
# concrete underlying model IDs, so no UI combo edit can starve the safety gate.
# Cross-provider redundancy is preserved across the failover chain: lane 0 is the
# low-latency Gemini model, lane 1 the independent GPT reasoning model.
DEFAULT_JUDGES: Tuple[JudgeSpec, ...] = (
    Judge("omniroute", "gemini-3.7-flash-high"),
    Judge("omniroute", "gpt-5.6-terra-xhigh"),
)

# Long messages are truncated head+tail: register gives itself away in the
# opening address and the closing sign-off, and an unbounded body would make
# latency and token use depend on message size.
_MAX_HEAD = 1800
_MAX_TAIL = 600


def _clip(text: str) -> str:
    """Clip an over-long body, keeping the register-bearing head and tail."""
    if len(text) <= _MAX_HEAD + _MAX_TAIL:
        return text
    return f"{text[:_MAX_HEAD]}\n\n[...]\n\n{text[-_MAX_TAIL:]}"


def _parse_verdict(raw: str) -> Optional[str]:
    """Recover LEAK/DELIVER from a possibly chatty completion.

    Returns None when the model said neither, so an unparsable answer is treated
    as "no opinion" rather than being silently coerced into a verdict.

    Two answer SHAPES must both parse, because different model families emit the
    verdict in opposite positions and keying on only one silently drops the
    other into None (fail-open -> a real LEAK reaches the contact unheard):

      * verdict-FIRST (instruction-followers, e.g. claude-haiku): the first
        non-empty line IS the bare verdict word, followed by explanatory prose
        that mentions neither word again. 2026-09-11: this shape returned
        "DELIVER\\n\\nThis message is addressed TO Morgan…" and the
        tail-only scan below saw a tail of pure reasoning -> None. The exact
        same shape on a real leak ("LEAK\\n\\n<why>") would also have been
        None, a false negative — the dangerous direction.
      * verdict-LAST (reasoning models): the verdict CONCLUDES the answer after
        a thinking block that restates both words while deliberating, so only
        the tail carries the conclusion.

    So: try the first non-empty line as an exact verdict first; otherwise fall
    back to the tail scan (last one wins when both appear). Position no longer
    changes the outcome — the class of "verdict in the wrong place -> None" is
    gone rather than patched for one model.
    """
    if not raw:
        return None
    # Verdict-first: the leading non-empty line is exactly the verdict word.
    for line in raw.strip().splitlines():
        stripped = line.strip().upper()
        if not stripped:
            continue
        if stripped == _LEAK:
            return _LEAK
        if stripped == _DELIVER:
            return _DELIVER
        break  # a non-blank, non-verdict first line -> this is not verdict-first
    # Verdict-last: reasoning models conclude in the tail.
    tail = raw.strip().upper()[-_TAIL_CHARS:]
    has_leak = _LEAK in tail
    has_deliver = _DELIVER in tail
    if has_leak and not has_deliver:
        return _LEAK
    if has_deliver and not has_leak:
        return _DELIVER
    if has_leak and has_deliver:
        # Both present in the tail: the LAST one is the conclusion.
        return _LEAK if tail.rfind(_LEAK) > tail.rfind(_DELIVER) else _DELIVER
    return None


def _extract_text(response) -> str:
    """Pull text from a completion, tolerating reasoning-only responses."""
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    # Some reasoning models put everything in `reasoning` and leave content "".
    reasoning = getattr(message, "reasoning", None)
    return reasoning if isinstance(reasoning, str) else ""


def judge_register(
    text: str,
    recipient: str,
    *,
    request: str = "",
    judges: Sequence[JudgeSpec] = DEFAULT_JUDGES,
    timeout: float = 20.0,
    confirm_leaks: bool = True,
) -> Optional[str]:
    """Return ``"LEAK"``, ``"DELIVER"``, or ``None`` if no judge could answer.

    ``request`` is the recipient's own inbound message that this reply answers.
    It is the premise the verdict must be judged against (2026-09-11): a
    contact who asks "hazme un resumen para mandárselo al owner" has commissioned
    a third-person artifact, and that artifact is the correct delivery — not a
    leak. Judged in isolation the same text is a textbook owner-report. Empty
    when unknown; the judge then falls back to text-only rules.

    ``None`` is a first-class outcome, not an error: every lane being down must
    degrade to "no opinion" so the caller keeps the regex verdict rather than
    inventing one.

    **Why a LEAK is confirmed before it is believed.** ``temperature=0`` is not
    a determinism guarantee on MoE-routed serving stacks. Measured 2026-09-01
    over 5 runs of the 11-case production set, the primary lane returned a
    stable verdict on 10 cases and flipped on one (``DELIVER`` x4, ``LEAK`` x1)
    — a ~20% intermittent false alarm on a genuine reply. Since the alarm's
    whole value is being believed when it fires, a single flaky vote must not
    drive it: an alarm that cries wolf is one you learn to ignore.

    So LEAK (the acting verdict) is re-asked once and only stands if confirmed.
    DELIVER is accepted on the first vote — it is the no-op outcome, and paying
    a second call to confirm silence buys nothing. This costs one extra call
    only on the rare LEAK path, and it converts a coin flip into a verdict that
    has to be reproduced.
    """
    if not text or not text.strip() or not recipient:
        return None

    verdict = _ask_judges(text, recipient, request=request, judges=judges, timeout=timeout)
    if verdict != _LEAK or not confirm_leaks:
        return verdict

    # Re-ask: a leak alarm must be reproducible, not a sampling artefact.
    second = _ask_judges(text, recipient, request=request, judges=judges, timeout=timeout)
    if second == _LEAK:
        return _LEAK
    if second is None:
        # Lanes died between calls: keep the only real verdict we obtained
        # rather than downgrading a possible leak to silence.
        return _LEAK
    logger.debug(
        "register_audit: LEAK not reproduced on confirmation (second=%s); "
        "treating as unstable and not alerting", second,
    )
    return second


def _ask_judges(
    text: str,
    recipient: str,
    *,
    judges: Sequence[JudgeSpec],
    timeout: float,
    request: str = "",
) -> Optional[str]:
    """One pass down the failover chain. Returns the first parsable verdict."""
    # The request rides BEFORE the message so the judge reads the reply as an
    # answer to it. Clipped like the message: a long inbound (a pasted PDF
    # brochure) must not crowd the text under judgment out of the window.
    request_block = (
        f"--- THE REQUEST (what the recipient asked for) ---\n{_clip(request)}\n--- END REQUEST ---\n\n"
        if request and request.strip()
        else ""
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT + "\n\n" + SHARED_DELIVERY_RULES},
        {"role": "user", "content": (
            f"Recipient of this chat: {recipient}\n\n"
            f"{request_block}"
            f"--- MESSAGE ---\n{_clip(text)}\n--- END ---\n\nLEAK or DELIVER?"
        )},
    ]
    return ask_lanes(messages, judges=judges, timeout=timeout, parse=_parse_verdict)


def ask_lanes(
    messages: List[Dict[str, str]],
    *,
    judges: Sequence[JudgeSpec],
    timeout: float,
    parse: Callable[[str], Optional[str]],
) -> Optional[str]:
    """Walk the failover chain once with ``messages``; return the first
    verdict ``parse`` accepts, or None when no lane answered parsably.

    Shared by every register judge (the prosecutor framing here, the reader
    framing in gateway/delivery_judge.py) so all of them ride the same lanes,
    the same per-lane overrides, and the same fail-open contract: any lane
    error is logged and skipped, never raised into the runtime.
    """
    # Imported lazily: this module is imported during gateway start-up, and the
    # auxiliary client pulls in provider SDKs we do not want on that path.
    try:
        from agent.auxiliary_client import call_llm
    except Exception:  # pragma: no cover - import shape varies across installs
        logger.debug("register_audit: auxiliary client unavailable", exc_info=True)
        return None

    for entry in judges:
        lane = _judge_fields(entry)
        # Optional per-lane overrides: a self-hosted lane pins its own endpoint,
        # key, and a longer timeout; cloud lanes leave them None and inherit the
        # provider default + the caller's timeout.
        kwargs = {
            "task": "register_audit",
            "provider": lane.provider,
            "model": lane.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 2000,
            "timeout": lane.timeout if lane.timeout is not None else timeout,
        }
        if lane.base_url is not None:
            kwargs["base_url"] = lane.base_url
        if lane.api_key is not None:
            kwargs["api_key"] = lane.api_key
        try:
            response = call_llm(**kwargs)
        except Exception as exc:
            logger.debug(
                "register_audit: judge %s/%s failed: %s: %s",
                lane.provider, lane.model, type(exc).__name__, str(exc)[:120],
            )
            continue

        verdict = parse(_extract_text(response))
        if verdict is not None:
            return verdict
        logger.debug("register_audit: judge %s/%s gave no parsable verdict", lane.provider, lane.model)

    return None


def audit_sent_message_async(
    text: str,
    recipient: str,
    *,
    regex_verdict: bool,
    on_leak,
    request: str = "",
    judges: Sequence[JudgeSpec] = DEFAULT_JUDGES,
    timeout: float = 20.0,
) -> Optional[threading.Thread]:
    """Judge an already-delivered message off the hot path.

    ``on_leak(text, recipient)`` is invoked only when the judge says LEAK and the
    regex did not already catch it — i.e. only for the recall gap this exists to
    close. Re-reporting what the regex already handled would be noise, and noise
    is how an alarm gets ignored.

    ``request`` is the inbound message this reply answers — see
    ``judge_register``. Pass it whenever the caller has it.

    Returns the worker thread (tests join it); None when nothing was scheduled.
    """
    if regex_verdict:
        # Already caught and handled on the hot path.
        return None
    if not text or not text.strip() or not recipient:
        return None

    def _run() -> None:
        try:
            verdict = judge_register(
                text, recipient, request=request, judges=judges, timeout=timeout
            )
            if verdict == _LEAK:
                on_leak(text, recipient)
        except Exception:
            # A monitoring path must never raise into the runtime it monitors.
            logger.debug("register_audit: async audit failed", exc_info=True)

    thread = threading.Thread(
        target=_run, name="register-audit", daemon=True
    )
    thread.start()
    return thread
