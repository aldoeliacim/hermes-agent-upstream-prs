"""Regression tests for the non-owner outbound backstop.

Design note (2026-06-16): the gateway deliberately does NOT cosmetically
rewrite the owner's *name* out of replies on non-owner silos. A blanket
owner-name -> "the owner" substitution reads as robotic (arguably more
conspicuous to a contact than a name slipping through once) and a
whole-message regex mangles FILE:/MEDIA: delivery paths, silently dropping
attachments. The PRIMARY owner-identity defense is prompt-side:
skip_user_profile gates USER.md out of a non-owner contact's prompt.

What the send-path backstop DOES do on proven non-owner silos is scrub
genuine sensitive info (credentials/tokens/keys) via
_redact_secrets_for_non_owner — the same secret scrub
_sanitize_gateway_final_response applies to Telegram, extended to all
platforms for non-owner destinations. It must NOT touch names, tone, or
delivery paths.
"""

from gateway.run import _redact_secrets_for_non_owner


def test_empty_input_is_passthrough():
    assert _redact_secrets_for_non_owner("") == ""
    assert _redact_secrets_for_non_owner(None) is None


def test_clean_reply_is_unchanged():
    clean = "Listo, ya quedó. Te puse subtítulos a los 8 episodios."
    assert _redact_secrets_for_non_owner(clean) == clean


def test_owner_name_in_prose_is_NOT_rewritten():
    # The owner's name is intentionally left alone — no robotic "the owner".
    inp = "Hola Sam, claro que sí."
    out = _redact_secrets_for_non_owner(inp)
    assert out == inp
    assert "the owner" not in out


def test_meta_preamble_is_NOT_stripped():
    # Tone/preamble shaping was removed; conversational openers survive.
    inp = "Everything's wired up and flowing. Listo, ya quedó."
    assert _redact_secrets_for_non_owner(inp) == inp


def test_file_media_delivery_paths_are_untouched():
    # The backstop must never mangle a delivery directive / path.
    for directive in (
        "Here you go FILE:/home/owner/videos/x.mp4",
        "See MEDIA:/home/owner/clip.mp4",
        'Attached: FILE:"/home/owner/My Files/report.pdf"',
    ):
        assert _redact_secrets_for_non_owner(directive) == directive


def test_secret_tokens_are_redacted():
    # Built at runtime so no credential-shaped literal sits in the source. The live test used a
    # pre-masked literal that the scrub does not match, so its assertion held vacuously.
    anthropic_key = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0" * 4
    github_token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    for secret in (anthropic_key, github_token):
        out = _redact_secrets_for_non_owner(f"Your key is {secret} ok")
        assert secret not in out, f"{secret[:8]}... reached a non-owner verbatim"
        assert out.startswith("Your key is ") and out.endswith(" ok")


def test_non_owner_final_is_scrubbed_before_it_reaches_the_contact(monkeypatch, tmp_path):
    """End to end: a non-owner DM final carrying a credential leaves the gateway scrubbed.

    Two layers do this, deliberately: the upstream chat-surface scrub
    (_sanitize_gateway_final_response -> redact_for_egress, every non-raw platform) and this
    layer's non-owner backstop inside the privacy block. The assertion pins the OUTCOME, so
    it holds if either layer regresses alone; the helper tests above pin the backstop itself.
    """
    import asyncio
    from tests.gateway import test_owner_register_gate_e2e as e2e

    github_token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    final = f"Listo, ya quedó. El token es {github_token}"
    response, _adapter, calls = asyncio.run(
        e2e._turn(monkeypatch, tmp_path, text=final, verdict="DELIVER"))
    assert github_token not in (response or ""), "credential reached the contact's final"
    assert response.startswith("Listo, ya quedó.")
    assert calls and all(github_token not in c for c in calls), (
        "the judge lanes (a third-party model) must never be sent the raw credential"
    )


def test_backstop_covers_raw_text_surfaces_the_chat_scrub_skips(monkeypatch, tmp_path):
    """Where the backstop is the ONLY scrub: webhook/api_server are raw-text surfaces
    (_GATEWAY_RAW_TEXT_PLATFORMS), so _sanitize_gateway_final_response passes their text
    through untouched. A non-owner reader on such a surface still gets a credential-free
    final, and the judge lanes (a third-party model) never see the raw credential."""
    import asyncio
    from gateway.config import Platform
    from tests.gateway import test_owner_register_gate_e2e as e2e

    github_token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    final = f"Listo, ya quedó. El token es {github_token}"

    def _raw_surface_source(chat_id, name, _orig=e2e._source):
        src = _orig(chat_id, name)
        src.platform = Platform.WEBHOOK
        return src

    monkeypatch.setattr(e2e, "_source", _raw_surface_source)
    response, _adapter, calls = asyncio.run(
        e2e._turn(monkeypatch, tmp_path, text=final, verdict="DELIVER"))
    assert github_token not in (response or ""), "credential reached a non-owner on a raw surface"
    assert calls and all(github_token not in c for c in calls), "judge lanes saw the raw credential"
