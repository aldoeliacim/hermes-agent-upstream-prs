"""Regression tests for channel_skill_bindings supplied as a JSON string.

`hermes config set platforms.<p>.extra.channel_skill_bindings '[...]'` stores
the value verbatim as a string rather than parsed YAML. Before this was
handled, resolve_channel_skills() hit `isinstance(bindings, list)` -> False
and returned None *silently*: the CLI printed a success line, the config file
looked right, and the bound skill simply never loaded.

That false green is exactly how the whatsapp-group-participation skill (which
carries the "Default is SILENCE" reply gate) went unloaded across 10 WhatsApp
group sessions / 15,413 messages while nine unrelated skills loaded normally.
"""

from gateway.platforms.base import resolve_channel_skills


class TestChannelSkillBindingsFromJsonString:
    def test_json_string_bindings_resolve(self):
        """A JSON-string binding must behave like a parsed list."""
        extra = {
            "channel_skill_bindings": (
                '[{"id":"120363555000000001@g.us",'
                '"skills":["whatsapp-group-participation"]}]'
            )
        }
        assert resolve_channel_skills(
            extra, "120363555000000001@g.us", None
        ) == ["whatsapp-group-participation"]

    def test_json_string_matches_on_parent_id(self):
        """Parent/bare-id fallback still applies to string-form bindings."""
        extra = {
            "channel_skill_bindings": '[{"id":"120363555000000001","skills":["s"]}]'
        }
        assert resolve_channel_skills(
            extra, "120363555000000001@g.us", "120363555000000001"
        ) == ["s"]

    def test_json_string_single_skill_key(self):
        """The singular `skill` key works through the string path too."""
        extra = {"channel_skill_bindings": '[{"id":"C1","skill":"solo"}]'}
        assert resolve_channel_skills(extra, "C1", None) == ["solo"]

    def test_non_matching_id_returns_none(self):
        extra = {"channel_skill_bindings": '[{"id":"C1","skills":["a"]}]'}
        assert resolve_channel_skills(extra, "C2", None) is None

    def test_malformed_json_string_returns_none_without_raising(self):
        """Bad JSON must degrade to None, never break message ingestion."""
        extra = {"channel_skill_bindings": "not-json-at-all"}
        assert resolve_channel_skills(extra, "C1", None) is None

    def test_parsed_list_still_works(self):
        """The normal YAML-parsed path must be unaffected."""
        extra = {"channel_skill_bindings": [{"id": "C1", "skills": ["a", "b"]}]}
        assert resolve_channel_skills(extra, "C1", None) == ["a", "b"]

    def test_empty_and_missing_bindings(self):
        assert resolve_channel_skills({}, "C1", None) is None
        assert resolve_channel_skills({"channel_skill_bindings": []}, "C1", None) is None
        assert resolve_channel_skills(
            {"channel_skill_bindings": "[]"}, "C1", None
        ) is None
