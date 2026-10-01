"""Generic, PII-free unit tests for recipient alias derivation.

The reply-gate third-person leak guard needs candidate first-name aliases for a
recipient whose stored display name is not a clean first name (a pushname, a
handle, an emoji-glued or emoji-wrapped name, an abbreviated "P. Name" token).

``recipient_aliases_for`` derives these *generically* from the display name at
runtime — there is deliberately no hardcoded chat-id -> real-name table in
source (zero PII). These tests exercise only generic, invented inputs.
"""

from gateway.response_filters import (
    mentions_recipient_third_person,
    recipient_aliases_for,
)


class TestRecipientAliasesGeneric:
    def test_emoji_glued_name_yields_letter_token(self):
        assert recipient_aliases_for("id-1", "Rainbow🦉") == ("Rainbow",)

    def test_emoji_wrapped_name_yields_letter_token(self):
        assert recipient_aliases_for("id-2", "✨💫Nova✨") == ("Nova",)

    def test_honorific_abbreviation_dropped_keeps_name(self):
        # "P." is < 3 chars and a bare initial -> dropped; "Sam" kept.
        assert recipient_aliases_for("id-3", "P. Sam") == ("Sam",)

    def test_handle_with_separators_splits_into_tokens(self):
        out = recipient_aliases_for("id-4", "Sunflowerpop 🌸 sunflower_kid")
        assert "Sunflowerpop" in out
        assert "sunflower" in out and "kid" in out

    def test_camelcase_handle_splits(self):
        out = recipient_aliases_for("id-5", "SunFlowerKid")
        assert out == ("Sun", "Flower", "Kid")

    def test_clean_first_name_not_re_emitted(self):
        # Token equal to the trimmed display name is redundant (guard tests it
        # directly), so a clean single-token name yields nothing extra.
        assert recipient_aliases_for("id-6", "Someone") == ()

    def test_honorific_stopword_dropped(self):
        # "Don" is a stopword; only the real name survives.
        assert recipient_aliases_for("id-7", "Don Rio") == ("Rio",)

    def test_empty_or_non_string_returns_empty(self):
        assert recipient_aliases_for("id", "") == ()
        assert recipient_aliases_for("id", "   ") == ()
        assert recipient_aliases_for("id", None) == ()
        assert recipient_aliases_for("id", 12345) == ()

    def test_no_private_identifier_lookup(self):
        # chat_id is accepted for signature stability but must not drive output:
        # two different ids with the same display name derive identical aliases.
        a = recipient_aliases_for("aaaa@id", "Nova🦉")
        b = recipient_aliases_for("bbbb@id", "Nova🦉")
        assert a == b == ("Nova",)

    def test_deduplicates_case_insensitively(self):
        out = recipient_aliases_for("id", "Nova nova NOVA extra")
        # one 'Nova' + 'extra'; case-insensitive dedup.
        lowered = [t.lower() for t in out]
        assert lowered.count("nova") == 1
        assert "extra" in out


class TestGuardRearmedByGenericAliases:
    def test_guard_blind_without_alias_then_catches_with_alias(self):
        display_name = "Rainbow🦉"
        leak = "Ya le avisé a Rainbow que su cuenta quedó lista."
        # Without aliases the stored (emoji-glued) name doesn't match the prose.
        assert mentions_recipient_third_person(leak, display_name) is False
        aliases = recipient_aliases_for("id", display_name)
        assert mentions_recipient_third_person(leak, display_name, aliases) is True

    def test_guard_does_not_overblock_unrelated_text(self):
        display_name = "Nova🦉"
        aliases = recipient_aliases_for("id", display_name)
        benign = "Listo, ya quedó todo configurado."
        assert mentions_recipient_third_person(benign, display_name, aliases) is False
