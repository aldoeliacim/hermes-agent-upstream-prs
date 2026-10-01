"""Passive event observation — reactions/polls never wake the agent, any chat.

The bug class (measured 2026-08-11, group session 20260811_023058): 58
tool-mode turns in one day, 54 of which ended in a stay-silent decision. The
dominant wake sources carry zero conversational content — bare reactions
("[Reaction: 👍 to …]"), poll creations, poll votes.

ARCHITECTURAL INVARIANT (2026-08-26): a bare reaction/poll event CANNOT be a
request — it carries no text to answer — and that holds in EVERY chat type.
So the observe-vs-dispatch decision keys on the EVENT's media type, not the
chat's shape. Reactions/polls are always persisted as attributed transcript
context and never dispatch an agent turn.

This subsumes the reply-gate for the whole reaction class: the agent is never
asked to "decide silence" on a reaction, so it can never leak a stay-silent
narration ("Reaction acknowledged — no reply needed.") to any chat. The
earlier DM exclusion (which forced a dispatch, then relied on response-filter
regexes to catch the leaked reasoning) is gone — the leak is impossible at the
source.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.reply_policy import ReplyPolicy


def _mk_adapter(observe_cfg=True):
    from gateway.config import Platform
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.config = SimpleNamespace(extra={"observe_passive_events": observe_cfg})
    adapter.platform = Platform.WHATSAPP  # backs the base-class `name` property
    return adapter


def _mk_event(reply_policy=ReplyPolicy.FREE_RESPONSE, chat_id="120363555000000001@g.us",
              text="[Reaction: 👍 to ABC123]"):
    return SimpleNamespace(
        source=SimpleNamespace(
            reply_policy=reply_policy,
            chat_id=chat_id,
            user_id="5215551000001@s.whatsapp.net",
            user_name="Contact",
        ),
        text=text,
        message_id="MSG1",
    )


class TestShouldObservePassiveEvent:
    def test_group_reaction_is_observed(self):
        adapter = _mk_adapter()
        data = {"isGroup": True, "mediaType": "reaction"}
        assert adapter._should_observe_passive_event(data, _mk_event()) is True

    def test_poll_creation_and_vote_are_observed(self):
        adapter = _mk_adapter()
        for mt in ("poll", "poll_update"):
            data = {"isGroup": True, "mediaType": mt}
            assert adapter._should_observe_passive_event(data, _mk_event()) is True

    def test_dm_reaction_is_observed_not_dispatched(self):
        # THE FIX: a reaction in a DM is passive signal, not a request. It is
        # recorded as transcript context, never dispatched — so the agent can
        # never emit a stay-silent narration to the DM.
        adapter = _mk_adapter()
        data = {"isGroup": False, "mediaType": "reaction"}
        assert adapter._should_observe_passive_event(
            data, _mk_event(reply_policy=ReplyPolicy.DM, chat_id="5215551000001@s.whatsapp.net")
        ) is True

    def test_mention_gated_group_reaction_is_observed(self):
        # A reaction can't @-mention the bot; it's passive in every policy.
        adapter = _mk_adapter()
        data = {"isGroup": True, "mediaType": "reaction"}
        assert adapter._should_observe_passive_event(
            data, _mk_event(reply_policy=ReplyPolicy.MENTION_GATED)
        ) is True

    def test_text_messages_never_observed(self):
        # Real conversational content always reaches the agent, any chat type.
        adapter = _mk_adapter()
        for is_group in (True, False):
            data = {"isGroup": is_group, "mediaType": ""}
            assert adapter._should_observe_passive_event(data, _mk_event()) is False

    def test_config_kill_switch_restores_dispatch(self):
        adapter = _mk_adapter(observe_cfg=False)
        data = {"isGroup": True, "mediaType": "reaction"}
        assert adapter._should_observe_passive_event(data, _mk_event()) is False


class TestObservePassiveEvent:
    def test_persists_observed_row_without_dispatch(self):
        adapter = _mk_adapter()
        store = MagicMock()
        store.get_or_create_session.return_value = SimpleNamespace(session_id="S1")
        adapter._session_store = store

        adapter._observe_passive_event(_mk_event())

        store.append_to_transcript.assert_called_once()
        session_id, entry = store.append_to_transcript.call_args[0]
        assert session_id == "S1"
        assert entry["role"] == "user"
        assert entry["observed"] is True
        assert "Contact" in entry["content"]
        assert "[Reaction: 👍" in entry["content"]

    def test_missing_store_is_silent_noop(self):
        adapter = _mk_adapter()
        adapter._session_store = None
        # Must not raise — the poll loop lives above this call.
        adapter._observe_passive_event(_mk_event())

    def test_store_error_never_propagates(self):
        adapter = _mk_adapter()
        store = MagicMock()
        store.get_or_create_session.side_effect = RuntimeError("db locked")
        adapter._session_store = store
        adapter._observe_passive_event(_mk_event())  # no raise
