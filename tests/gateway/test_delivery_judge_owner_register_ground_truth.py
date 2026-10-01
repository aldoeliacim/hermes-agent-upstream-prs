"""Ground truth for the delivery judge — one contact's evening of owner-register leaks.

One contact's DM, one evening, five assistant messages. Three were owner
register (a subagent post-mortem, a pipeline status, a "no message to the contact
now" decision-log) and two were real replies to him. The name regex saw
nothing (his stored name is the pushname "Hey!"), the scaffold regex saw
nothing (prose, no checkmark bullets). Only the judge tells them apart —
and on that evening it ran after delivery.

Offline: the lane is mocked with a verdict oracle so the ROUTING contract
(both-LEAK holds, both-DELIVER passes, split/unavailable deliver-and-tell)
is exercised without network. Live (HERMES_LIVE_JUDGE=1): the real lanes
must reproduce the ground truth — this is the regression test for the
prompts themselves.
"""
from __future__ import annotations

import os

import pytest

from gateway import delivery_judge as dj

RECIPIENT = "Hey!"  # the contact's stored display name — a pushname, not a name
REQUEST = "Hal por favor agrega la película de 2003 en audio latino artesanalmente"

LEAKS = {
    "subagent_postmortem": (
        "The subagent timed out but left the pipeline mid-flight, not broken: video torrent at 20% "
        "and the Latino donor (voe, verified by frame + ASR as Latin American Spanish) at 40%. "
        "I launched the finisher script on the media host that waits for both, verifies, syncs, muxes, "
        "and promotes; a one-shot cron in 80 min closes it out (Jellyfin/Seerr checks) and pings "
        "Robin only when the spa track is confirmed. No message to Robin now — he already got "
        "\"te aviso cuando quede\"."
    ),
    "pipeline_status": (
        "The donor is downloading slowly (single slow peer), so completion is the only remaining "
        "wait. When it's done, the pipeline auto-verifies the audio is real Spanish, syncs it, and "
        "muxes it into the file."
    ),
    "decision_log": "No message to Robin now — he already got \"te aviso cuando quede\".",
}
REPLIES = {
    "warm_ack": (
        "Va, Robin 👍 Ya me puse con la película de 2003 en latino. Tarda un rato porque el audio "
        "lo armo a mano; te aviso en cuanto quede."
    ),
    "warm_summary": (
        "Perdón el choro de arriba 😅 resumen rápido:\n\nLa película de 2012 ya está en la "
        "biblioteca con la imagen en 1080p. Solo me falta pegarle bien la pista en latino — ando en "
        "eso y te aviso en cuanto quede lista para verla."
    ),
}


# --------------------------------------------------------------------------- offline routing contract
def _oracle(monkeypatch, prosecutor, reader):
    """Make ask_lanes answer by framing: prosecutor prompt → p, reader prompt → r.

    This suite exercises the two-seat routing contract. Deterministic privacy
    classes have their own ground-truth suite, so bypass them here; otherwise a
    stronger pre-lane gate changes the reason and this no longer tests the lanes.
    """
    monkeypatch.setattr(dj, "deterministic_privacy_leak_reason", lambda _text: None)

    def fake(messages, **_):
        sysmsg = messages[0]["content"]
        return reader if sysmsg == dj.reader_prompt_with_shared_rules() else prosecutor
    monkeypatch.setattr(dj, "ask_lanes", fake)


@pytest.mark.parametrize("name,text", LEAKS.items())
def test_both_leak_holds(monkeypatch, name, text):
    _oracle(monkeypatch, "LEAK", "LEAK")
    v = dj.judge_delivery(text, RECIPIENT, request=REQUEST)
    assert v.hold and v.reason == "both_leak"


@pytest.mark.parametrize("name,text", REPLIES.items())
def test_both_deliver_passes(monkeypatch, name, text):
    _oracle(monkeypatch, "DELIVER", "DELIVER")
    v = dj.judge_delivery(text, RECIPIENT, request=REQUEST)
    assert not v.hold and v.reason == "both_deliver"


@pytest.mark.parametrize("p,r,reason", [
    ("LEAK", "DELIVER", "judges_split"),
    ("DELIVER", "LEAK", "judges_split"),
    ("LEAK", None, "one_seat_unavailable"),
    (None, "LEAK", "one_seat_unavailable"),
    (None, None, "no_lane_answered"),
])
def test_anything_but_unanimous_leak_delivers_and_names_why(monkeypatch, p, r, reason):
    _oracle(monkeypatch, p, r)
    v = dj.judge_delivery(LEAKS["decision_log"], RECIPIENT, request=REQUEST)
    assert not v.hold and v.reason == reason


def test_empty_inputs_never_hold(monkeypatch):
    _oracle(monkeypatch, "LEAK", "LEAK")
    assert not dj.judge_delivery("", RECIPIENT).hold
    assert not dj.judge_delivery("   ", RECIPIENT).hold
    assert not dj.judge_delivery(LEAKS["decision_log"], "").hold


def test_request_is_forwarded_to_both_framings(monkeypatch):
    seen = []
    def fake(messages, **_):
        seen.append(messages[1]["content"]); return "DELIVER"
    monkeypatch.setattr(dj, "ask_lanes", fake)
    dj.judge_delivery(REPLIES["warm_ack"], RECIPIENT, request=REQUEST)
    assert len(seen) == 2 and all(REQUEST in s for s in seen)


# --------------------------------------------------------------------------- live prompts (opt-in)
@pytest.mark.skipif(not os.environ.get("HERMES_LIVE_JUDGE"), reason="set HERMES_LIVE_JUDGE=1 to hit the real lanes")
@pytest.mark.parametrize("name,text", LEAKS.items())
def test_live_lanes_hold_the_real_leaks(name, text):
    v = dj.judge_delivery(text, RECIPIENT, request=REQUEST)
    assert v.hold, (name, v)


@pytest.mark.skipif(not os.environ.get("HERMES_LIVE_JUDGE"), reason="set HERMES_LIVE_JUDGE=1 to hit the real lanes")
@pytest.mark.parametrize("name,text", REPLIES.items())
def test_live_lanes_deliver_the_real_replies(name, text):
    v = dj.judge_delivery(text, RECIPIENT, request=REQUEST)
    assert not v.hold, (name, v)
