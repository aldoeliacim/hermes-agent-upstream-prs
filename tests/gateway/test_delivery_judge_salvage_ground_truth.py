"""Ground-truth for the ephemeral-infra-teardown leak class and the salvage path.

A live incident: the model copied a file to a contact by spinning up a throwaway
HTTP server, tore it down, and PREPENDED an internal teardown self-note to an
otherwise clean, second-person reply:

    "El servidor temporal está apagado (conexión rechazada). Todo quedó limpio.

     ¡Listo, Robin! Ya te copié el documento a la laptop ..."

The whole reply was HELD, so the contact got silence — a legitimate answer lost
to protect against a stray line. The correct behaviour is to STRIP the flagged
line (deterministic class, fail-closed) and DELIVER the clean remainder, routing
only the stripped note to the owner.

Two invariants are locked here, both deterministic (no LLM lanes needed):

1. The teardown/self-bookkeeping class is detected, and legitimate service-status
   answers a contact actually asked about are NOT (fail-closed detector must not
   swallow real replies).
2. Salvage never lets a flagged block reach the delivered body, refuses to
   stitch across an interior hole, and refuses when the hold is diffuse (no
   deterministic block to safely strip).
"""
from __future__ import annotations

import pytest

from gateway import delivery_judge as dj


INFRA_LINE = "El servidor temporal está apagado (conexión rechazada). Todo quedó limpio."
CONTACT_BODY = (
    "¡Listo, Robin! Ya te copié el documento a la laptop. Lo encuentras en tu "
    "Escritorio con el nombre: material.pdf\n\n"
    "Ahí lo abres e imprimes directo para tus alumnos."
)
CONTACT_FULL = INFRA_LINE + "\n\n" + CONTACT_BODY


# ---------------------------------------------------------------------------
# 1. The ephemeral-infra-teardown class is detected.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        INFRA_LINE,
        "Levanté un servidor temporal para pasarte el archivo.",
        "temporary http server is down now",
        "connection refused, ya lo apagué",
        "El puerto quedó liberado, todo quedó limpio.",
        "handle proc_a1b2c3d4 terminado, pid 8123 muerto",
        # Terse teardown self-note (2026-09-22 class): bare "servidor apagado"
        # with no "temporal"/"conexión rechazada" — the earlier regex missed it,
        # so salvage refused and the whole reply was held (contact got silence).
        "Servidor apagado. Todo listo.",
        "Proceso detenido, ya está.",
    ],
)
def test_ephemeral_infra_teardown_is_detected(text):
    assert dj._is_ephemeral_infra_teardown(text)


def test_ephemeral_infra_reason_is_stable():
    assert dj.deterministic_privacy_leak_reason(INFRA_LINE) == "ephemeral_infra_teardown"


# ---------------------------------------------------------------------------
# 2. Legitimate service-status / everyday answers are NOT flagged. The detector
#    is fail-closed, so a false positive here would silently swallow a real
#    reply to a contact.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "Sí, tu servidor de Minecraft ya está prendido, entra cuando quieras.",
        "Listo, ya te copié el documento a la laptop, está en tu Escritorio.",
        "La descarga va al 80%, en un rato queda lista.",
        "Tu página ya está en línea, ábrela en el navegador.",
        "Apagué la tele antes de salir, no te preocupes.",
        "Tu servidor está apagado, préndelo desde el panel para poder jugar.",
    ],
)
def test_legit_status_answers_are_not_ephemeral_infra(text):
    assert not dj._is_ephemeral_infra_teardown(text)


# ---------------------------------------------------------------------------
# 3. Salvage invariants — no LLM lanes, deterministic taint authority only.
#    We stub the LLM backstop to DELIVER so we isolate the stripping logic.
# ---------------------------------------------------------------------------
@pytest.fixture
def _stub_deliver_backstop(monkeypatch):
    def _always_deliver(text, recipient, **__):
        return dj.Verdict(False, "DELIVER", "DELIVER", "stub_backstop")

    monkeypatch.setattr(dj, "judge_delivery", _always_deliver)


def test_salvage_strips_leading_infra_line(_stub_deliver_backstop):
    s = dj.salvage_deliverable_span(CONTACT_FULL, "Robin", request="pásame el documento")
    assert s.ok
    assert INFRA_LINE not in s.body  # the leak NEVER reaches the contact
    assert s.body.startswith("¡Listo, Robin!")
    assert s.stripped == INFRA_LINE


def test_salvage_strips_trailing_infra_line(_stub_deliver_backstop):
    full = CONTACT_BODY + "\n\n" + INFRA_LINE
    s = dj.salvage_deliverable_span(full, "Robin")
    assert s.ok
    assert INFRA_LINE not in s.body
    assert s.stripped == INFRA_LINE


def test_salvage_recovers_terse_teardown_incident(_stub_deliver_backstop):
    # 2026-09-22: model prepended a bare "Servidor apagado. Todo listo." to a
    # clean reply. Before the terse-teardown regex, the detector missed it,
    # salvage had no authority to strip, and the whole reply was HELD -> the
    # contact got silence. Now the terse note is stripped and the body ships.
    terse = "Servidor apagado. Todo listo."
    full = terse + "\n\n" + CONTACT_BODY
    s = dj.salvage_deliverable_span(full, "Robin")
    assert s.ok, "terse teardown note must be strippable so the contact isn't silenced"
    assert terse not in s.body
    assert s.stripped == terse
    assert s.body.startswith("¡Listo, Robin!")


def test_salvage_refuses_interior_leak(_stub_deliver_backstop):
    # clean - leak - clean: stripping would stitch across a hole and reorder.
    clean_a = "¡Hola Robin! Aquí está tu documento listo en el Escritorio."
    clean_b = "¡Cualquier cosa me avisas y con gusto te ayudo! 😊"
    full = clean_a + "\n\n" + INFRA_LINE + "\n\n" + clean_b
    s = dj.salvage_deliverable_span(full, "Robin")
    assert not s.ok


def test_salvage_refuses_when_hold_is_diffuse(_stub_deliver_backstop):
    # No block is a deterministic leak class -> nothing safe to strip -> hold whole.
    full = "un párrafo cualquiera\n\notro párrafo cualquiera"
    s = dj.salvage_deliverable_span(full, "Robin")
    assert not s.ok


def test_salvage_refuses_when_backstop_holds_remainder(monkeypatch):
    # Even with a clean deterministic strip, if the LLM backstop vetoes the
    # remainder, salvage refuses (reject-only backstop still has teeth).
    def _always_hold(text, recipient, **__):
        return dj.Verdict(True, "LEAK", "LEAK", "stub_hold")

    monkeypatch.setattr(dj, "judge_delivery", _always_hold)
    s = dj.salvage_deliverable_span(CONTACT_FULL, "Robin")
    assert not s.ok



# ---------------------------------------------------------------------------
# 6. Incident: the self-note was GLUED to the greeting in one paragraph
#    ("Servidor apagado. ¡Listo, Robin! ✅"). Block-level salvage stripped the
#    whole paragraph, so the greeting that belonged to the contact went to the
#    owner. Salvage now peels only the flagged EDGE sentences of the block
#    touching the clean span; interior notes still refuse.
# ---------------------------------------------------------------------------
GLUED_FULL = (
    "Servidor apagado. ¡Listo, Robin! ✅\n\n"
    "Ya te dejé en el escritorio de tu laptop la guía ilustrada con las fotos, en dos versiones:\n\n"
    "📄 Trajes tipicos - guia ilustrada.pdf — para imprimir y repartir a tus alumnos."
)


def test_glued_leading_note_is_peeled_and_greeting_kept(_stub_deliver_backstop):
    s = dj.salvage_deliverable_span(GLUED_FULL, "Robin")
    assert s.ok
    assert s.stripped == "Servidor apagado."
    assert s.body.startswith("¡Listo, Robin! ✅")
    assert "Servidor apagado" not in s.body


def test_glued_note_in_a_single_paragraph_is_peeled(_stub_deliver_backstop):
    text = "Servidor apagado. ¡Listo, Robin! Ya te dejé el PDF en tu escritorio para imprimir."
    s = dj.salvage_deliverable_span(text, "Robin")
    assert s.ok and s.stripped == "Servidor apagado."
    assert s.body == "¡Listo, Robin! Ya te dejé el PDF en tu escritorio para imprimir."


def test_glued_trailing_note_is_peeled(_stub_deliver_backstop):
    text = "Ya te dejé el PDF en tu escritorio para imprimir, Robin. ¡Listo! Servidor apagado."
    s = dj.salvage_deliverable_span(text, "Robin")
    assert s.ok and s.stripped == "Servidor apagado."
    assert "Servidor apagado" not in s.body


def test_interior_glued_note_still_refuses(_stub_deliver_backstop):
    text = "¡Hola Robin! Servidor apagado. Ya quedó tu PDF en el escritorio para imprimir."
    assert not dj.salvage_deliverable_span(text, "Robin").ok


def test_short_unaddressed_remainder_stays_with_the_note(_stub_deliver_backstop):
    """'Servidor apagado. Todo listo.' is the self-note's own status clause, not a reply:
    salvage must not deliver the dangling fragment on its own."""
    s = dj.salvage_deliverable_span("Servidor apagado. Todo listo.", "Robin")
    assert not s.ok
