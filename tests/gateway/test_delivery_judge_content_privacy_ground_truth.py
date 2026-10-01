"""Ground-truth privacy classes from recent non-owner WhatsApp deliveries.

These are not register mistakes: every message is grammatically addressed to the
recipient. The leaked CONTENT is the problem, so the existing two-seat
"is this written TO you?" judge is structurally incapable of catching them.
They must be deterministic fail-closed classes before either LLM seat runs.
"""
from __future__ import annotations

import pytest

from gateway import delivery_judge as dj


REQUEST_HISTORY_LEAK = (
    'Encontrado. La serie *The Twilight Zone (1959)* (tmdbId 1400) '
    'la pidió **Robin** el 15 de diciembre de 2025.'
)
REQUEST_HISTORY_ASK = 'Hal quien pidió la serie de the Twilight zone 1959?'

BACKEND_LEAK_PORTS = (
    'El servidor está sano, Alex. No es él. El problema: tu app está '
    'probando `localhost:8096` y `:8920`, que son puertos internos. '
    'Jellyfin 10.11.11 responde bien.'
)
BACKEND_LEAK_SOURCING = (
    'La serie sí está en Chill, pero se bajó en inglés. La versión doblada '
    'existe pero está carísima de conseguir porque casi nadie la comparte.'
)
BACKEND_LEAK_GROUP_DNS = (
    'El CV vivía en el apex example.org. Agregué cv.example.org como dominio '
    'personalizado de Cloudflare Pages, con un CNAME proxied y certificado activo.'
)


def _no_lanes(*_, **__):
    raise AssertionError('deterministic content leaks must not call LLM judge lanes')


def test_cross_user_media_request_history_is_detected():
    assert dj._is_cross_user_request_history_disclosure(REQUEST_HISTORY_LEAK)


def test_name_before_request_verb_is_also_detected():
    assert dj._is_cross_user_request_history_disclosure(
        'Robin solicitó esta película ayer en Chill.'
    )


@pytest.mark.parametrize(
    'text',
    [
        'Tu película fue solicitada por Robin.',
        'La película la pidió robin ayer.',
        'La película la pidió P. Jesse ayer.',
        'La película fue solicitada por María José.',
        'The movie was requested by John.',
        # Name + clitic + verb. Before 2026-09-27 these were only "caught"
        # when a leading "Nadie pidió" false-positive happened to fire too.
        'Robin la pidió ayer en Chill, esa película.',
        'Nadie pidió esa película, pero Robin la pidió ayer.',
        'Nadie pidió esa película; Oscar la solicitó ayer.',
    ],
)
def test_requester_disclosure_voice_case_and_multiword_variants(text):
    assert dj._is_cross_user_request_history_disclosure(text)


@pytest.mark.parametrize(
    'text',
    [
        'La película fue solicitada por ti.',
        'The movie was requested by you.',
        'No puedo compartir quién solicitó esa serie.',
        # Negative-existential requester (incident 2026-09-27, P. Jesse DM):
        # saying NOBODY asked for a title names no third party.
        'Ya lo comprobé, Jessi: _Unabomber_ no la pidió nadie. Chill la agregó '
        'solo el 26 de septiembre en la mañana, junto con otras películas de '
        'estreno y populares. Por eso no aparece a nombre de ninguna persona.',
        'Esa película no fue solicitada por nadie; Chill la agregó sola.',
        'Esa serie no la solicitó ningún usuario.',
        'Nadie pidió esa película; entró sola a Chill.',
        'Nobody requested that movie; Chill added it automatically.',
        'The movie was requested by no one.',
        'No one requested that show.',
    ],
)
def test_requester_disclosure_does_not_block_own_history_or_refusal(text):
    assert not dj._is_cross_user_request_history_disclosure(text)


def test_cross_user_request_history_is_held_before_llm(monkeypatch):
    monkeypatch.setattr(dj, 'ask_lanes', _no_lanes)
    v = dj.judge_delivery(
        REQUEST_HISTORY_LEAK, 'P. Jesse', request=REQUEST_HISTORY_ASK
    )
    assert v.hold
    assert v.reason == 'cross_user_request_history'


def test_private_request_history_refusal_is_deliverable():
    text = 'No puedo compartir quién solicitó esa serie; cada cuenta conserva su privacidad.'
    assert not dj._is_cross_user_request_history_disclosure(text)


def test_ordinary_social_third_party_request_is_not_media_history():
    text = 'Robin pidió que le avise cuando llegues a la comida.'
    assert not dj._is_cross_user_request_history_disclosure(text)


def test_explicit_internal_ports_and_service_version_are_detected():
    assert dj._is_backend_substance_disclosure(BACKEND_LEAK_PORTS)


def test_content_sourcing_mechanics_are_detected():
    assert dj._is_backend_substance_disclosure(BACKEND_LEAK_SOURCING)


def test_named_backend_attached_to_chill_is_detected_in_any_word_order():
    assert dj._is_backend_substance_disclosure('Chill usa Jellyfin como servidor.')


def test_dns_pages_certificate_mechanics_are_backend_substance():
    assert dj._is_backend_substance_disclosure(BACKEND_LEAK_GROUP_DNS)


def test_outcome_only_website_reply_is_deliverable():
    assert not dj._is_backend_substance_disclosure('cv.example.org ya está disponible.')


def test_backend_disclosure_is_held_before_llm(monkeypatch):
    monkeypatch.setattr(dj, 'ask_lanes', _no_lanes)
    v = dj.judge_delivery(
        BACKEND_LEAK_PORTS,
        'Alex',
        request='Hal, ¿por qué no me puedo conectar?',
    )
    assert v.hold
    assert v.reason == 'backend_substance_disclosure'


def test_outcome_only_chill_reply_is_deliverable():
    text = 'Ya quedó disponible en Chill; prueba otra vez y dime si abre bien.'
    assert not dj._is_backend_substance_disclosure(text)


def test_general_knowledge_about_jellyfin_is_not_deployment_disclosure():
    text = 'Jellyfin es un proyecto de software libre para organizar una biblioteca multimedia.'
    assert not dj._is_backend_substance_disclosure(text)


def test_shared_destinations_require_privacy_guard_even_for_owner_sender():
    from gateway.run_turn import _destination_requires_privacy_guard

    assert _destination_requires_privacy_guard(skip_owner_profile=False, chat_type='group')
    assert _destination_requires_privacy_guard(skip_owner_profile=True, chat_type='dm')
    assert not _destination_requires_privacy_guard(skip_owner_profile=False, chat_type='dm')


def test_shared_destination_guard_is_wired_to_deterministic_content_gate():
    """Structural backstop for the shared-destination branch (behaviour is pinned end to
    end in test_owner_register_gate_e2e.py: group leak held + rerouted, owner-trusted
    group exempt). Scoped to the enclosing privacy block instead of a fixed character
    window, which broke every time the block grew."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[2].joinpath('gateway', 'run_turn.py').read_text()
    start = src.index('_shared_destination =')
    end = src.index('elif _verdict.reason in (', start)
    block = src[start:end]
    assert 'elif _tail_reaches_recipient and _shared_destination and response:' in block
    assert 'deterministic_privacy_leak_reason(response)' in block
    assert '_reroute_leaked_reply_to_owner(source, response)' in block


# --- own-task-progress narration ("Significa que estoy en el proceso…") ---
# A contact received a narration of internal task machinery — which stage the
# task is in, which fetch step is running, and that an internal error was
# overcome — instead of an outcome. Ground truth: this exact message reached
# a contact. It is grammatically addressed TO them, so
# the register judge cannot catch it; the leaked CONTENT is the problem.
TASK_PROGRESS_LEAK_CONTACT = (
    'Significa que estoy en el proceso de buscar y añadir el audio en español '
    'latino para el episodio 1 de Elementary. Ya superamos un error inicial, y '
    'ahora estoy en la etapa de obtenerlo. Te avisaré en cuanto esté listo.'
)


def test_own_task_progress_narration_is_detected():
    assert dj._is_own_task_progress_narration(TASK_PROGRESS_LEAK_CONTACT)


def test_overcame_internal_error_framing_alone_is_a_leak():
    assert dj._is_own_task_progress_narration('Ya solucionamos el error, sigo bajando el archivo.')
    assert dj._is_own_task_progress_narration('Hubo un error inicial pero ya lo pasamos.')


def test_stage_framing_with_fetch_verb_is_a_leak():
    assert dj._is_own_task_progress_narration('Ando en proceso de descargar el capítulo.')
    assert dj._is_own_task_progress_narration('Estoy en la etapa de añadir la pista de audio.')


def test_outcome_only_status_is_not_task_progress_narration():
    # A contact's own requested task, reported as an OUTCOME, must deliver.
    assert not dj._is_own_task_progress_narration('¡Listo! Ya quedó tu película, pruébala cuando quieras.')
    assert not dj._is_own_task_progress_narration('Ya te lo dejé listo en Chill, ábrelo y dale play.')
    assert not dj._is_own_task_progress_narration('Ya casi está, en cuanto termine te aviso 👍')


def test_offband_process_and_fetch_words_do_not_misfire():
    # "en el parque" is not a stage; "proceso de mudanza" has no fetch verb;
    # "descárgalo tú" is an imperative to the contact, not self-narration.
    assert not dj._is_own_task_progress_narration('Estoy en el parque, te marco al rato.')
    assert not dj._is_own_task_progress_narration('Voy en proceso de mudanza, ando ocupado.')
    assert not dj._is_own_task_progress_narration('Sí sirve, descárgalo tú directo de la página.')
    assert not dj._is_own_task_progress_narration('Oye, ¿ya buscaste en tu correo el número de serie?')


def test_task_progress_narration_is_held_before_llm(monkeypatch):
    monkeypatch.setattr(dj, 'ask_lanes', _no_lanes)
    v = dj.judge_delivery(
        TASK_PROGRESS_LEAK_CONTACT,
        'Robin',
        request='Tengo otra laptop, ¿se pueden conectar entre sí para respaldar la información?',
    )
    assert v.hold
    assert v.reason == 'own_task_progress_narration'


# Incident 2026-09-27 (Kevin DM): a 6.5k-char technical audit mentioned
# "audio oficial" in one paragraph and "los diffs que pidió Polar" in another.
# The media context and the named requester were evaluated over the WHOLE text,
# so two unrelated paragraphs combined into a "cross-user request history" hold.
def test_requester_and_media_context_in_unrelated_paragraphs_do_not_combine():
    text = (
        "El sonido oficial va aparte: exportar el vídeo sin audio.\n\n"
        "No escribí los diffs que pidió Polar. Hacerlos a ciegas sería inventar código."
    )
    assert not dj._is_cross_user_request_history_disclosure(text)


def test_requester_disclosure_across_sentences_in_one_paragraph_still_detected():
    assert dj._is_cross_user_request_history_disclosure(
        "Te cuento de esa película. La pidió Robin ayer."
    )


def test_audit_excerpt_with_audio_rules_and_colleague_request_is_not_held():
    # Distilled from the incident body (no third-party content committed).
    text = (
        "*2. Sonido oficial.* L15 pide lo contrario: no le pongas el audio "
        "oficial encima, va el original.\n\n"
        "*Lo que no hice.* No escribí los diffs que pidió Polar: sin el repo "
        "a la vista serían inventados."
    )
    assert not dj._is_cross_user_request_history_disclosure(text)
