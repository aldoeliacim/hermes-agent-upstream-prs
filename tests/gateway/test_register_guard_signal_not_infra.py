"""Register-guard: an owner alert must fire on SIGNAL, never on the guard's own
infra outage, and NEVER on a bare judges_split with no deterministic corroboration.

Root cause history:
  1. The old code fired the same ground-truth banner (with the full contact reply
     pasted in) for no_lane_answered — i.e. neither judge ran at all. Pure infra
     failure, zero content signal, spamming the owner during any lane flap.
  2. Then it still treated a bare ``judges_split`` (regex=none) as "signal" — one
     noisy LLM judge crying LEAK on clean content with zero corroboration. That
     echoed the contact's clean reply to the owner: itself the false positive.

The decision now lives in ONE pure function, ``register_overrule_action``, imported
by both the run path and this test, so the table cannot silently drift from the
code again (make-the-bug-class-impossible).
"""
from gateway.run_turn import register_overrule_action

import pytest


@pytest.mark.parametrize(
    "reason,regex,expect_action",
    [
        # Deterministic corroboration -> ground-truth banner (only real signal).
        ("judges_split", "ephemeral_infra_teardown", "banner"),
        ("no_lane_answered", "backend_substance_disclosure", "banner"),
        ("one_seat_unavailable", "some_detector", "banner"),
        # Lanes could not answer + regex blind -> throttled body-free infra alert.
        ("no_lane_answered", "none", "infra"),
        ("one_seat_unavailable", "none", "infra"),
        # Bare split, zero corroboration -> silent. THIS is the 2026-09-22 fix:
        # one noisy LLM judge crying LEAK on a clean reply is not signal.
        ("judges_split", "none", "silent"),
    ],
)
def test_overrule_action_table(reason, regex, expect_action):
    assert register_overrule_action(reason, regex) == expect_action


def test_bare_split_is_silent_not_banner():
    """The 2026-09-22 false positive: prosecutor=DELIVER, reader=LEAK, regex=none
    on a completely clean teaching answer. A bare split must NOT banner (which
    would paste the contact's clean reply to the owner) — it must deliver
    silently."""
    assert register_overrule_action("judges_split", "none") == "silent"


def test_lanes_down_incident_is_infra_not_banner():
    """Lanes down + regex clean = infra advisory, never the ground-truth banner."""
    assert register_overrule_action("no_lane_answered", "none") == "infra"


def test_infra_alert_is_throttled_and_bodyless():
    """The degraded-guard alert must (a) throttle to one per window and (b) carry
    no message body. Verified structurally against the helper on the mixin."""
    import inspect

    from gateway.run_notifications import GatewayNotificationsMixin  # type: ignore[attr-defined]

    src = inspect.getsource(GatewayNotificationsMixin._alert_register_judges_degraded)
    assert "_register_degraded_alert_ts" in src
    assert "WINDOW" in src.upper()
    assert "{response}" not in src and "response" not in src.split('"""')[-1]


# --- DM-path twin: uncorroborated LLM hold must not silence the contact --------
from gateway.run_turn import dm_hold_is_corroborated


def test_pure_llm_dm_hold_is_uncorroborated():
    # both_leak / judges_split = the LLM lanes decided alone, regex=none. These are
    # the warm clean-DM false-positive class: a clean reply the lanes flagged and
    # that re-judges DELIVER on a re-run. Must NOT be treated as corroborated.
    assert dm_hold_is_corroborated("both_leak") is False
    assert dm_hold_is_corroborated("judges_split") is False
    assert dm_hold_is_corroborated("") is False


def test_detector_backed_dm_hold_is_corroborated():
    # A real deterministic detector firing (infra teardown, backend disclosure, a
    # name/structure leak) IS corroboration -> silencing+rerouting stays justified.
    assert dm_hold_is_corroborated("ephemeral_infra_teardown") is True
    assert dm_hold_is_corroborated("backend_substance_disclosure") is True
    assert dm_hold_is_corroborated("name_leak") is True
