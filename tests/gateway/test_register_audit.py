"""Tests for the semantic register auditor (gateway/register_audit.py).

Two layers:

1. **Offline** (default) — verdict parsing, failover, fail-open, and the
   contract that the auditor never runs when the regex already caught the leak.
   No network.

2. **Live** (opt-in via HERMES_TEST_REGISTER_JUDGE=1) — the production eval set
   against the real free judge lanes. Kept opt-in so CI never depends on a
   third-party endpoint, but preserved in-repo so the eval that justified the
   model choice can be re-run after any prompt or model change.
"""

from __future__ import annotations

import os
import sys
import threading
import types
from typing import Any, Optional

import pytest

# Repo root on sys.path so `gateway.*` imports resolve. This file lives in
# tests/gateway/, so the root is two levels up — the earlier "..\" pointed at
# tests/ itself, and inserting THAT at position 0 shadowed the real
# `gateway.platforms.*` package for every module collected afterwards.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from gateway.register_audit import (  # noqa: E402
    _clip,
    _parse_verdict,
    audit_sent_message_async,
    judge_register,
)


def _started(thread: Optional[threading.Thread]) -> threading.Thread:
    """Assert a worker was actually scheduled, and narrow the Optional."""
    assert thread is not None, "expected the auditor to schedule a worker"
    return thread


# --------------------------------------------------------------------------
# Verdict parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("LEAK", "LEAK"),
    ("DELIVER", "DELIVER"),
    ("  deliver  ", "DELIVER"),
    ("The message is a LEAK.", "LEAK"),
    # Reasoning models restate the question before answering; the CONCLUSION is
    # the last verdict word, not the first.
    ("Is this a LEAK or DELIVER? ... Final answer: DELIVER", "DELIVER"),
    ("Considering DELIVER, but the report register means LEAK", "LEAK"),
    # Verdict-FIRST shape (2026-09-11): instruction-following models
    # (claude-haiku) emit the bare verdict on line 1, then explanatory prose
    # that names neither word again. The tail-only scan returned None here,
    # which fail-opens — a real "LEAK\n\n<why>" would have been a silent false
    # negative that reached the contact. The leading verdict line must win.
    ("DELIVER\n\nThis message is addressed TO Morgan using formal usted.", "DELIVER"),
    ("LEAK\n\nMixed register: opens addressed then reports about her.", "LEAK"),
    ("leak\nthird-person narration about the recipient", "LEAK"),
    # No opinion must stay None rather than defaulting to a verdict.
    ("I'm not sure about this one", None),
    ("", None),
    (None, None),
])
def test_parse_verdict(raw, expected):
    assert _parse_verdict(raw) == expected


def test_parse_verdict_reads_the_tail_not_the_head():
    """A long preamble mentioning the wrong word must not win over the answer."""
    raw = "LEAK " * 200 + " Therefore the correct classification is DELIVER"
    assert _parse_verdict(raw) == "DELIVER"


# --------------------------------------------------------------------------
# Clipping
# --------------------------------------------------------------------------

def test_clip_preserves_head_and_tail():
    body = "A" * 5000 + "TAIL_MARKER"
    out = _clip(body)
    assert out.startswith("A" * 100)
    assert "TAIL_MARKER" in out, "sign-off carries register and must survive"
    assert len(out) < len(body)


def test_clip_leaves_short_text_untouched():
    assert _clip("hola") == "hola"


# --------------------------------------------------------------------------
# Failover / fail-open
# --------------------------------------------------------------------------

class _FakeCallLLM:
    """call_llm stub. `script` maps model -> verdict string or Exception."""

    def __init__(self, script: dict):
        self.script = script
        self.calls: list[str] = []

    def __call__(self, *, task, provider, model, messages, **kw) -> Any:
        self.calls.append(model)
        outcome = self.script.get(model)
        if isinstance(outcome, Exception):
            raise outcome
        msg = types.SimpleNamespace(content=outcome, reasoning=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


@pytest.fixture
def aux(monkeypatch):
    """Install a fake auxiliary_client so no test touches the network."""
    def _install(script: dict) -> _FakeCallLLM:
        call = _FakeCallLLM(script)
        fake = types.ModuleType("agent.auxiliary_client")
        setattr(fake, "call_llm", call)
        monkeypatch.setitem(sys.modules, "agent.auxiliary_client", fake)
        return call
    return _install


JUDGES = (("nvidia", "primary"), ("openrouter", "fallback"))  # legacy combo aliases — kept for parse/failover unit tests that mock the lanes by these names


def test_primary_verdict_is_used(aux):
    call = aux({"primary": "LEAK"})
    assert judge_register("x", "Ana", judges=JUDGES) == "LEAK"
    # A LEAK is asked twice (confirmation); the point here is that the FALLBACK
    # lane is never consulted while the primary keeps answering.
    assert set(call.calls) == {"primary"}, "fallback must not be used when primary answers"


def test_falls_over_to_second_lane_on_error(aux):
    call = aux({"primary": RuntimeError("503"), "fallback": "DELIVER"})
    assert judge_register("x", "Ana", judges=JUDGES) == "DELIVER"
    assert call.calls == ["primary", "fallback"]


def test_unparsable_primary_falls_through(aux):
    """A lane that answers but says neither word must not end the chain."""
    call = aux({"primary": "hmm, hard to say", "fallback": "LEAK"})
    assert judge_register("x", "Ana", judges=JUDGES) == "LEAK"
    # Two passes (verdict + confirmation), each walking primary -> fallback.
    assert call.calls == ["primary", "fallback", "primary", "fallback"]


def test_all_lanes_down_returns_none(aux):
    aux({"primary": RuntimeError("down"), "fallback": RuntimeError("down")})
    assert judge_register("x", "Ana", judges=JUDGES) is None


# --------------------------------------------------------------------------
# Leak confirmation — an alarm must be reproducible, not a sampling artefact
# --------------------------------------------------------------------------

class _SequenceCallLLM:
    """call_llm stub returning a scripted SEQUENCE of verdicts per call."""

    def __init__(self, sequence: list):
        self.sequence = list(sequence)
        self.calls: list[str] = []

    def __call__(self, *, task, provider, model, messages, **kw) -> Any:
        self.calls.append(model)
        outcome = self.sequence.pop(0) if self.sequence else "DELIVER"
        if isinstance(outcome, Exception):
            raise outcome
        msg = types.SimpleNamespace(content=outcome, reasoning=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


@pytest.fixture
def aux_seq(monkeypatch):
    def _install(sequence: list) -> _SequenceCallLLM:
        call = _SequenceCallLLM(sequence)
        fake = types.ModuleType("agent.auxiliary_client")
        setattr(fake, "call_llm", call)
        monkeypatch.setitem(sys.modules, "agent.auxiliary_client", fake)
        return call
    return _install


def test_leak_must_be_reproduced_to_stand(aux_seq):
    """Measured 2026-09-01: the primary lane flipped DELIVER x4 / LEAK x1 on a
    genuine reply. A one-off LEAK vote must not raise an alarm."""
    call = aux_seq(["LEAK", "DELIVER"])
    assert judge_register("x", "Ana", judges=JUDGES) == "DELIVER"
    assert len(call.calls) == 2, "a LEAK must be re-asked before it is believed"


def test_confirmed_leak_stands(aux_seq):
    call = aux_seq(["LEAK", "LEAK"])
    assert judge_register("x", "Ana", judges=JUDGES) == "LEAK"
    assert len(call.calls) == 2


def test_deliver_is_not_re_asked(aux_seq):
    """DELIVER is the no-op outcome; confirming silence buys nothing."""
    call = aux_seq(["DELIVER"])
    assert judge_register("x", "Ana", judges=JUDGES) == "DELIVER"
    assert len(call.calls) == 1, "must not spend a second call to confirm a no-op"


def test_leak_stands_when_confirmation_is_impossible(aux_seq):
    """If the lanes die between calls, keep the only real verdict obtained
    rather than silently downgrading a possible leak."""
    call = aux_seq(["LEAK", RuntimeError("down"), RuntimeError("down")])
    assert judge_register("x", "Ana", judges=JUDGES) == "LEAK"
    assert len(call.calls) == 3


def test_missing_auxiliary_client_is_not_fatal(monkeypatch):
    """Import failure must degrade to no-opinion, never raise into the gateway."""
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", None)
    assert judge_register("x", "Ana", judges=JUDGES) is None


@pytest.mark.parametrize("text,recipient", [
    ("", "Ana"), ("   ", "Ana"), ("hola", ""),
])
def test_empty_inputs_short_circuit(aux, text, recipient):
    call = aux({"primary": "LEAK"})
    assert judge_register(text, recipient, judges=JUDGES) is None
    assert call.calls == [], "must not spend a call on empty input"


# --------------------------------------------------------------------------
# The async contract — the part that protects delivery
# --------------------------------------------------------------------------

def test_no_audit_when_regex_already_caught_it(aux):
    """The auditor exists for the RECALL gap. Re-reporting a caught leak is
    noise, and a noisy alarm is an ignored alarm."""
    call = aux({"primary": "LEAK"})
    fired = []
    thread = audit_sent_message_async(
        "whatever", "Ana", regex_verdict=True,
        on_leak=lambda *a: fired.append(a), judges=JUDGES,
    )
    assert thread is None
    assert call.calls == []
    assert fired == []


def test_alerts_on_the_recall_gap(aux):
    """Regex said clean, judge says leak -> the owner must hear about it."""
    aux({"primary": "LEAK"})
    fired = []
    thread = _started(audit_sent_message_async(
        "Just a heart reaction from Robin. No reply needed.", "Robinpop",
        regex_verdict=False, on_leak=lambda *a: fired.append(a), judges=JUDGES,
    ))
    thread.join(timeout=5)
    assert len(fired) == 1
    assert fired[0][1] == "Robinpop"


def test_silent_when_judge_agrees_it_is_clean(aux):
    aux({"primary": "DELIVER"})
    fired = []
    thread = _started(audit_sent_message_async(
        "Ana, ya quedó 👍", "Ana", regex_verdict=False,
        on_leak=lambda *a: fired.append(a), judges=JUDGES,
    ))
    thread.join(timeout=5)
    assert fired == []


def test_callback_exception_never_escapes(aux):
    """A monitoring path must not be able to crash the runtime it monitors."""
    aux({"primary": "LEAK"})

    def _boom(*_a):
        raise RuntimeError("notification backend is down")

    thread = _started(audit_sent_message_async(
        "x", "Ana", regex_verdict=False, on_leak=_boom, judges=JUDGES,
    ))
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_audit_thread_is_daemon(aux):
    """A hung judge must never hold up interpreter shutdown."""
    aux({"primary": "DELIVER"})
    thread = _started(audit_sent_message_async(
        "x", "Ana", regex_verdict=False, on_leak=lambda *a: None, judges=JUDGES,
    ))
    assert thread.daemon is True
    thread.join(timeout=5)


# --------------------------------------------------------------------------
# The request is the premise (commissioned-artifact false positive)
# --------------------------------------------------------------------------
# A contact asked "hazme un resumen para mandárselo a Sam". The reply — a
# third-person comparison headed "para Sam" — IS the artifact she asked for,
# and was judged in isolation as an owner-report. The judge must see what was
# asked. These pin the MECHANISM: the request reaches the judge, ahead of the
# message, clipped, and through every entry point; and its absence changes
# nothing for existing callers.

class _CapturingCallLLM:
    def __init__(self, verdict="DELIVER"):
        self.verdict = verdict
        self.prompts: list[str] = []

    def __call__(self, *, task, provider, model, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        msg = types.SimpleNamespace(content=self.verdict, reasoning=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


@pytest.fixture
def aux_capture(monkeypatch):
    def _install(verdict="DELIVER") -> _CapturingCallLLM:
        call = _CapturingCallLLM(verdict)
        fake = types.ModuleType("agent.auxiliary_client")
        setattr(fake, "call_llm", call)
        monkeypatch.setitem(sys.modules, "agent.auxiliary_client", fake)
        return call
    return _install


_REQ = "Haz un resumen de ambas para mandárselo a Sam y decidir en conjunto"
_ART = "Resumen para Sam:\n\n*Escuela A vs Escuela B*\n• Escuela A ~$132k\n• Escuela B ~$61k"


def test_request_reaches_the_judge_before_the_message(aux_capture):
    call = aux_capture()
    judge_register(_ART, "P. Jesse", request=_REQ, judges=JUDGES)
    prompt = call.prompts[0]
    assert "THE REQUEST" in prompt and _REQ in prompt
    assert prompt.index(_REQ) < prompt.index("--- MESSAGE ---"), (
        "the request must precede the message so the reply is read as an answer to it"
    )
    assert _ART in prompt


def test_no_request_means_no_request_block(aux_capture):
    """Existing callers pass nothing; the prompt must be byte-identical to before."""
    call = aux_capture()
    judge_register(_ART, "P. Jesse", judges=JUDGES)
    assert "THE REQUEST" not in call.prompts[0]
    call2 = aux_capture()
    judge_register(_ART, "P. Jesse", request="   ", judges=JUDGES)
    assert "THE REQUEST" not in call2.prompts[0], "whitespace-only request is no request"


def test_request_is_clipped_like_the_message(aux_capture):
    """A pasted brochure must not push the judged text out of the window."""
    from gateway.register_audit import _MAX_HEAD, _MAX_TAIL
    call = aux_capture()
    huge = "x" * (_MAX_HEAD + _MAX_TAIL + 5000)
    judge_register(_ART, "P. Jesse", request=huge, judges=JUDGES)
    prompt = call.prompts[0]
    assert "[...]" in prompt.split("--- MESSAGE ---")[0], "request was not clipped"
    assert _ART in prompt, "message under judgment was crowded out"


def test_request_rides_through_confirmation_pass(aux_capture):
    """Both votes must be judged against the same premise."""
    call = aux_capture(verdict="LEAK")
    judge_register(_ART, "P. Jesse", request=_REQ, judges=JUDGES)
    assert len(call.prompts) == 2
    assert all(_REQ in p for p in call.prompts)


def test_request_rides_through_async_entry_point(aux_capture):
    call = aux_capture()
    thread = _started(audit_sent_message_async(
        _ART, "P. Jesse", regex_verdict=False, on_leak=lambda *a: None,
        request=_REQ, judges=JUDGES,
    ))
    thread.join(timeout=5)
    assert call.prompts and _REQ in call.prompts[0]


def test_gateway_judge_call_site_passes_the_inbound_as_request():
    """Structural gate on the ONE production caller.

    The judge is only as good as its wiring: if gateway/run.py calls it
    without ``request=``, every unit test here stays green and the judge is
    back to judging in isolation (paging the owner for commissioned
    artifacts). Grep the CALL form, not the parameter name.
    """
    import re
    from pathlib import Path
    src = Path(__file__).resolve().parents[2].joinpath("gateway", "run_turn.py").read_text(encoding="utf-8")
    calls = re.findall(r"judge_delivery\((.*?)\)\n", src, flags=re.S)
    calls = [c for c in calls if "response" in c]  # the production call, not the import
    assert calls, "no judge_delivery call found in gateway/run.py"
    for args in calls:
        assert "request=" in args, (
            "gateway/run.py calls judge_delivery without request=; the judge "
            "will see the reply in isolation and page the owner for commissioned "
            "artifacts. Pass the inbound user text."
        )

# --------------------------------------------------------------------------
# Live eval — opt-in
# --------------------------------------------------------------------------

# Production cases. Each is a real message that was sent, or a near-miss variant
# of one. These are the regression fixtures for any prompt/model change.
LIVE_CASES = [
    # The 2026-08-23 leak the REGEX CANNOT SEE: stored name "Robinpop 🌸✨",
    # text says "Robin". This single case is why the judge exists.
    ("Just a ❤️ reaction from Robin. No reply needed. Ending quietly.",
     "Robinpop 🌸✨", "LEAK"),
    # Pronoun-only report: no name at all, so name-anchored regexes are blind.
    ("Confirmado disponible. No requiere acción adicional de mi parte.",
     "Alex", "LEAK"),
    # Mixed register: opens addressed, drifts into third person.
    ("Hey Casey, quick update.\n\nHis account was reactivated this morning and "
     "the transcode issue is resolved. I'll let him know once it finishes.",
     "Casey", "LEAK"),
    # Genuine replies that must NEVER be flagged.
    ("Lista, Alex 🏔️ la película ya está disponible en Chill, "
     "en Full HD con audio 5.1. Disfrútala 🍿", "Alex", "DELIVER"),
    ("Casey, ya lo revisé 👍 tu cuenta quedó al corriente, dale de nuevo.",
     "Casey", "DELIVER"),
    # Third person about SOMEONE ELSE is normal conversation, not a leak.
    ("Oye, tu hermana me escribió preguntando por la serie. Le dije que ya "
     "estaba lista, pero mejor confírmale tú.", "Alex", "DELIVER"),
    # Long technical prose, but written TO her.
    ("Te explico qué pasó con la película, porque estuvo raro 🏔️ Llevaba desde "
     "marzo atorada. El sistema te mostraba 'Solicitado', pero en realidad "
     "estaba trabada por un filtro de calidad. Lo forcé a mano y verifiqué el "
     "audio pista por pista. Si vuelves a ver algo así, avísame.",
     "Alex", "DELIVER"),
    # Media hand-off to the requester: delivery confirmation with the file's own
    # specs (duration/size/validity) and no pronoun/greeting. 2026-09-04: the
    # judge flagged this LEAK because "reports on work done as a status update"
    # swallowed it. Describing the thing you are delivering IS the delivery.
    ("Listo. MP3 de 3:34, audio válido, 5.6 MB.\n\n"
     "FILE:/tmp/hoy_me_atrevo_a_soltar.mp3\n\n"
     "Hermanos Pancardo – Hoy Me Atrevo a Soltar 🎵",
     "Quetzalcoatl", "DELIVER"),
    # Formal-USTED register (2026-09-11). Two real FALSE POSITIVES reached
    # contacts' DMs as owner-alerts: Spanish formal usted ("su laptop", "le
    # pongo", "le aviso", "va a revisar") is grammatically 3rd-person, and the
    # old prompt (+ its "su cuenta/le avise" LEAK markers) taught the judge to
    # read polite address AS narration about the recipient. Flipping usted->tú
    # made the same text DELIVER, proving register — not content — was the
    # trigger. These are the exact messages that leaked; they MUST be DELIVER.
    ("¡Listo! Ya lo dejé programado 🙌\n\nEn cuanto su laptop se reconecte "
     "(puede ser cuando la abra o tenga buena señal de internet), yo le pongo "
     "los 4 PDFs directo en el escritorio y le aviso aquí mismo. Va a revisar "
     "cada ratito durante las próximas horas — usted no",
     "Morgan", "DELIVER"),
    ("Todavía no se reconecta ese programita de su laptop, Robin — por eso aún "
     "no los puedo poner en el escritorio. No es nada grave, solo que no está "
     "'despierto' para dejarme entrar.\n\nYa lo tengo vigilado: en cuanto se "
     "conecte, yo le pongo los 4 PDFs en el escritorio y le aviso aquí mismo.",
     "Robin", "DELIVER"),
    # ...but usted must NOT become a laundering channel: a report aimed at a
    # THIRD party (the owner) or narrating the recipient as absent is still a
    # LEAK even in polite register. Recall floor for the usted class.
    ("La cuenta de Morgan sigue impagada; le aviso cuando pague. No requiere "
     "acción de mi parte.", "Morgan", "LEAK"),
    # Recipient's own preference reported back to them in the third person.
    ("Morgan's preference is: dubbed audio when available, otherwise original "
     "audio with subtitles.", "Morgan", "DELIVER"),
    # Recipient's own content: work details relevant to their request are
    # not automatically a private operator report.
    ("Morgan, progress: 16 of 88 episodes have corrected subtitles. The next "
     "batch needs a subtitle download; the server cannot reach the source. "
     "I can fetch the files elsewhere and transfer them if you want.",
     "Morgan", "DELIVER"),
    # Own data plus private mechanism disclosure must still be held.
    ("Your audio preference is stored in my private model infrastructure.",
     "Morgan", "LEAK"),
    # Third-party private infrastructure is never the recipient's own data.
    ("Taylor's private backup is stored in my cloud account.", "Morgan", "LEAK"),
    # True positive: third-party data, pronoun does NOT fire Branch B
    ("Hey Casey, quick update. His account was reactivated this morning and the "
     "transcode issue is resolved. I'll let him know once it finishes.",
     "Casey", "LEAK"),
]

# Request-aware cases (2026-09-11). (text, recipient, request, expected).
# The commissioned artifact: judged alone it is a textbook owner-report;
# judged against what was asked it is exactly the delivery. And the request
# must not become a universal excuse — a real leak stays a leak under an
# innocent request.
LIVE_CASES_WITH_REQUEST = [
    # The verbatim shape that paged the owner 2026-09-11 17:49.
    ("Con los datos nuevos, esto cambia la cuenta. Resumen actualizado para Sam:\n\n"
     "*Comparativa Licenciatura en Psicología en línea — Escuela A vs Escuela B*\n"
     "*Objetivo:* dar consulta psicológica.\n"
     "• Escuela A: ~$132,750 en 3 años, 4 clases/semana, FIMPES.\n"
     "• Escuela B: ~$61,000 en 3 años, 2 clases/semana fijas 7-9 pm.\n\n"
     "*Recomendación de Hal:* Escuela A si el presupuesto lo permite.\n\n"
     "¿Se lo mando yo a Sam o se lo reenvías tú?",
     "P. Jesse",
     "Haz un resumen bien planteado de ambas, sobre todos los temas que vimos "
     "para mandárselo a Sam y tenga los datos necesarios y tomar una decisión en conjunto",
     "DELIVER"),
    # A letter to a third party, commissioned.
    ("Estimado Lic. Ramírez:\n\nPor medio de la presente solicito la constancia "
     "laboral de la C. Alex Doe Example, quien labora en esta empresa "
     "desde 2021.\n\nAtentamente,",
     "P. Jesse", "redáctame una carta para mi jefe pidiendo mi constancia laboral",
     "DELIVER"),
    # Real leaks under an innocent request: the request must NOT launder them.
    ("Hola Alex. Su cuenta sigue impagada. Se le envió recordatorio. "
     "No requiere acción de mi parte.",
     "P. Jesse", "¿ya quedó lo de mi cuenta?", "LEAK"),
    ("Hey Alex, quick update. Her laptop was reset and I'll let her know once "
     "the installs finish.",
     "P. Jesse", "did you fix my laptop?", "LEAK"),
]


@pytest.fixture
def live_credentials(monkeypatch):
    """Restore real provider credentials for the opt-in live eval.

    ``tests/conftest.py`` deliberately blanks every credential-shaped env var so
    local and CI runs match and no test can accidentally bill a real provider.
    That is correct and must not be weakened — so the live eval re-supplies the
    keys explicitly, for itself only, by reading the canonical secret store.
    """
    env_path = os.path.expanduser("~/.hermes/.env")
    if not os.path.exists(env_path):
        pytest.skip("no ~/.hermes/.env — cannot run the live eval")

    # Every lane in DEFAULT_JUDGES needs its key here, or the live eval
    # silently runs on whichever fallback happens to have one — and passes
    # against a lane that is not the one production uses. Anthropic is the
    # primary since 2026-09-11 (haiku-4-5); it may be OAuth-only, in which
    # case the key is absent and the auxiliary client resolves the sub itself.
    wanted = {"ANTHROPIC_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OMNIROUTE_API_KEY"}
    found = set()
    with open(env_path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            if name in wanted:
                monkeypatch.setenv(name, value.strip().strip('"').strip("'"))
                found.add(name)

    if not found:
        pytest.skip("no judge credentials in ~/.hermes/.env")
    return found


@pytest.mark.skipif(
    os.environ.get("HERMES_TEST_REGISTER_JUDGE") != "1",
    reason="live judge eval: set HERMES_TEST_REGISTER_JUDGE=1 to run",
)
@pytest.mark.parametrize("text,recipient,expected", LIVE_CASES)
def test_live_judge_matches_production_ground_truth(
    text, recipient, expected, live_credentials
):
    assert judge_register(text, recipient) == expected


@pytest.mark.skipif(
    os.environ.get("HERMES_TEST_REGISTER_JUDGE") != "1",
    reason="live judge eval: set HERMES_TEST_REGISTER_JUDGE=1 to run",
)
@pytest.mark.parametrize("text,recipient,inbound,expected", LIVE_CASES_WITH_REQUEST)
def test_live_judge_honours_the_request(
    text, recipient, inbound, expected, live_credentials
):
    assert judge_register(text, recipient, request=inbound) == expected
