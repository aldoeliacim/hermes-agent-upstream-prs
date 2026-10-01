"""Poll votes that answer a clarify must WAKE the agent, not be observed away.

Bug (2026-09-01): a clarify rendered as a native WhatsApp poll registers a
pending question and blocks the agent. The vote came back correctly — the
bridge logged `poll_update_decode` with the right option for four consecutive
polls — but `_should_observe_passive_event` classifies mediaType
"poll_update" as passive signal, so the event was appended to the transcript
and never dispatched. The agent waited out the full clarify timeout and
reported "user did not respond" while the user was looking at their own vote.

Two correct features collided: passive-event observation (a bare reaction must
not burn a paid turn) and poll-rendered clarify (a vote IS the answer). The fix
is not to weaken either one, but to distinguish them: a poll vote is passive
UNLESS the session has a clarify waiting on it.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


class _Source:
    def __init__(self, chat_id="123@s.whatsapp.net", user_name="Owner"):
        self.chat_id = chat_id
        self.user_name = user_name
        self.user_id = "123"
        self.platform = "whatsapp"
        self.chat_type = "dm"


class _Event:
    def __init__(self, text="Option A"):
        self.text = text
        self.source = _Source()
        self.message_id = "poll:update:1"
        self.metadata = {}


class _Store:
    """Minimal session store exposing the key derivation the adapter uses."""

    def __init__(self, key="whatsapp:123"):
        self._key = key

    def _generate_session_key(self, source):
        return self._key


class _Adapter:
    """Bind the real methods under test onto a minimal host object.

    Importing the full WhatsApp adapter drags in the plugin platform stack; the
    logic under test only needs `_session_store`, `config.extra`, `name`, and
    the two methods themselves.
    """

    name = "whatsapp"

    def __init__(self, store=None, observe=True):
        self._session_store = store
        self.config = types.SimpleNamespace(extra={"observe_passive_events": observe})

    # Bound from the real class below.


@pytest.fixture
def adapter_cls():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    return WhatsAppAdapter


@pytest.fixture
def adapter(adapter_cls):
    def _make(store=None, observe=True):
        host = _Adapter(store=store, observe=observe)
        # The methods under test read the class-level media-type set.
        host._PASSIVE_MEDIA_TYPES = adapter_cls._PASSIVE_MEDIA_TYPES
        host._should_observe_passive_event = (
            adapter_cls._should_observe_passive_event.__get__(host)
        )
        host._resolves_pending_clarify = (
            adapter_cls._resolves_pending_clarify.__get__(host)
        )
        return host
    return _make


@pytest.fixture
def clarify_registry(monkeypatch):
    """Control what get_pending_for_session returns, without a live gateway."""
    state = {"pending": None}

    def _fake(session_key, *, include_choice_prompts=False):
        return state["pending"]

    import tools.clarify_gateway as cg
    monkeypatch.setattr(cg, "get_pending_for_session", _fake)
    return state


# --------------------------------------------------------------------------
# The regression
# --------------------------------------------------------------------------

def test_poll_vote_answering_a_clarify_is_dispatched(adapter, clarify_registry):
    """THE BUG: with a clarify pending, the vote must NOT be treated as passive."""
    clarify_registry["pending"] = object()  # a clarify is waiting
    host = adapter(store=_Store())
    data = {"mediaType": "poll_update"}
    assert host._should_observe_passive_event(data, _Event()) is False, (
        "a poll vote answering a pending clarify must wake the agent"
    )


def test_poll_vote_without_pending_clarify_stays_passive(adapter, clarify_registry):
    """A human poll in a group must still not burn a turn on every vote."""
    clarify_registry["pending"] = None
    host = adapter(store=_Store())
    data = {"mediaType": "poll_update"}
    assert host._should_observe_passive_event(data, _Event()) is True


def test_reactions_are_always_passive(adapter, clarify_registry):
    """The exception is scoped to poll votes; reactions never wake the agent,
    even while a clarify is open."""
    clarify_registry["pending"] = object()
    host = adapter(store=_Store())
    assert host._should_observe_passive_event({"mediaType": "reaction"}, _Event()) is True


def test_poll_creation_is_always_passive(adapter, clarify_registry):
    clarify_registry["pending"] = object()
    host = adapter(store=_Store())
    assert host._should_observe_passive_event({"mediaType": "poll"}, _Event()) is True


def test_ordinary_text_is_never_passive(adapter, clarify_registry):
    clarify_registry["pending"] = None
    host = adapter(store=_Store())
    assert host._should_observe_passive_event({"mediaType": ""}, _Event()) is False


# --------------------------------------------------------------------------
# Fail-open: the check must never break message intake
# --------------------------------------------------------------------------

def test_missing_session_store_falls_back_to_passive(adapter, clarify_registry):
    clarify_registry["pending"] = object()
    host = adapter(store=None)
    assert host._should_observe_passive_event({"mediaType": "poll_update"}, _Event()) is True


def test_clarify_lookup_error_falls_back_to_passive(adapter, monkeypatch):
    """A raising registry must degrade to prior behaviour, not crash intake."""
    import tools.clarify_gateway as cg

    def _boom(*a, **kw):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(cg, "get_pending_for_session", _boom)
    host = adapter(store=_Store())
    assert host._should_observe_passive_event({"mediaType": "poll_update"}, _Event()) is True


def test_key_derivation_error_falls_back_to_passive(adapter, clarify_registry):
    class _BadStore:
        def _generate_session_key(self, source):
            raise RuntimeError("no key")

    clarify_registry["pending"] = object()
    host = adapter(store=_BadStore())
    assert host._should_observe_passive_event({"mediaType": "poll_update"}, _Event()) is True


def test_kill_switch_still_disables_observation(adapter, clarify_registry):
    """observe_passive_events: false restores full dispatch for every type."""
    clarify_registry["pending"] = None
    host = adapter(store=_Store(), observe=False)
    assert host._should_observe_passive_event({"mediaType": "reaction"}, _Event()) is False
