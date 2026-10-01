"""Regression: `action="silent"` must NOT count as a current-chat delivery.

LIVE INCIDENT 2026-08-03, WhatsApp group 5215551000002-1448604752@g.us.
The model behaved perfectly: 20 send_message(action="silent") calls, every
turn logging decided_silent=1. Yet 16 of its private reply-gate rationales
were posted into the group as chat messages:

    "Silent — a sticker with no address to me doesn't warrant a reply."
    "Silent — Riley is captioning old family photos, not addressing me."

MECHANISM: action="silent" returns {"success": true, "decision": "silent"} —
success:true because the *call* succeeded, though nothing was delivered.
_transcript_has_current_chat_send() only checked tool name + success, so it
read that as proof of a send. _delivered_via_tool flipped True, and in the
reply-gate branch order:

    if _delivered_via_tool:      -> suppress tail, outcome="replied"
    elif _decided_silent:        -> suppress tail, outcome="silent"

the FIRST branch won. Both suppress the tail, so the leak came from the tail
being delivered upstream as the turn's response while the gate believed a
tool send had already covered the turn. Net effect: on precisely the turns
where the model chose silence, its reasoning shipped to the group.

The tell in the logs: `decided_silent=1 ... outcome=replied` on every turn,
plus a "counter MISS" warning (counter=0, transcript "proves" a send).
"""

import json


def _scan(runner_cls, messages):
    return runner_cls._transcript_has_current_chat_send(messages)


def _tool_msg(payload, name="send_message"):
    return {"role": "tool", "tool_name": name, "content": json.dumps(payload)}


class TestSilentDecisionIsNotADelivery:
    """The exact shapes the live incident produced."""

    def _scan_fn(self):
        from gateway.run import GatewayRunner

        return GatewayRunner._transcript_has_current_chat_send

    def test_silent_decision_is_not_a_send(self):
        """{"success": true, "decision": "silent"} must read as NO delivery."""
        scan = self._scan_fn()
        msgs = [
            {"role": "user", "content": "[Sticker]"},
            _tool_msg({"success": True, "decision": "silent"}),
        ]
        assert scan(msgs) is False

    def test_real_send_still_counts(self):
        """A genuine delivery must still dedup (no regression on double-send)."""
        scan = self._scan_fn()
        msgs = [
            {"role": "user", "content": "hola"},
            _tool_msg({"success": True, "target": "current"}),
        ]
        assert scan(msgs) is True

    def test_silent_then_real_send_counts(self):
        """A real send after a silent decision still counts as delivered."""
        scan = self._scan_fn()
        msgs = [
            {"role": "user", "content": "hola"},
            _tool_msg({"success": True, "decision": "silent"}),
            _tool_msg({"success": True, "target": "current"}),
        ]
        assert scan(msgs) is True

    def test_many_silent_decisions_never_count(self):
        """The live pattern: repeated silent turns must never fake a send."""
        scan = self._scan_fn()
        msgs = [{"role": "user", "content": "[Sticker]"}]
        msgs += [_tool_msg({"success": True, "decision": "silent"}) for _ in range(20)]
        assert scan(msgs) is False

    def test_failed_send_is_not_a_delivery(self):
        scan = self._scan_fn()
        msgs = [
            {"role": "user", "content": "hola"},
            _tool_msg({"success": False, "error": "boom"}),
        ]
        assert scan(msgs) is False

    def test_only_current_turn_is_scanned(self):
        """Stale sends from earlier turns must not poison this turn."""
        scan = self._scan_fn()
        msgs = [
            {"role": "user", "content": "turno viejo"},
            _tool_msg({"success": True, "target": "current"}),
            {"role": "user", "content": "turno nuevo"},
            _tool_msg({"success": True, "decision": "silent"}),
        ]
        assert scan(msgs) is False
