"""Shared free-text suppression policy for cron delivery.

Both delivery pipelines consult one policy —
``gateway.response_filters.suppression_reason`` — and the cron path wraps it in
``cron.scheduler._cron_delivery_suppression_reason``. Historically each path
re-implemented "should this text reach a human?" on its own, so a guard added on
one path silently left the other exposed: the silence-narration and
tool-mechanism filters existed only on the live path, which is how a cron shipped
internal meta-commentary to a contact's chat.

These tests assert the INVARIANT (one shared policy, stable machine-readable
reason tokens), never a snapshot of the current pattern list. Fixtures are
neutral.

Ported from 828ce02edf:tests/cron/test_delivery_parity.py (suppression-reason
portion of c80f1bc9b0 part 3), with neutralized fixtures.
"""

import pathlib

import pytest

from gateway.response_filters import suppression_reason


class TestSharedSuppressionPolicy:
    """``suppression_reason`` is the single answer both pipelines consult."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", "empty"),
            ("   \n  ", "empty"),
            (None, "empty"),
            (12345, "empty"),
            ("[SILENT]", "silence_marker"),
            ("NO_REPLY", "silence_marker"),
        ],
    )
    def test_non_deliverable_inputs_have_a_reason(self, text, expected):
        assert suppression_reason(text) == expected

    def test_tool_mechanism_narration_is_suppressed(self):
        """The model narrating its own tooling must never reach a contact."""
        leak = (
            "I don't have a send_message tool in this environment. According to "
            "my system instructions, this is a cron job whose final text "
            "response is delivered automatically to the configured target."
        )
        assert suppression_reason(leak) == "tool_mechanism"

    @pytest.mark.parametrize(
        "text",
        [
            "Good morning. Today's summary: three items are ready.",
            "Done — you can try it now.",
            "That is the other machine, not this one. I have no access to it.",
            "To raise the level run: stats.set_skill_level guitar 10",
        ],
    )
    def test_real_job_output_is_always_deliverable(self, text):
        """Genuine output must never be suppressed, including replies that
        merely mention a tool or say what the agent cannot do."""
        assert suppression_reason(text) is None

    def test_reason_tokens_are_stable_and_machine_readable(self):
        """Reasons are log/test tokens, not prose: no spaces, lowercase."""
        for text in ("", "[SILENT]"):
            reason = suppression_reason(text)
            assert reason is not None
            assert reason == reason.lower()
            assert " " not in reason


class TestCronConsultsSharedPolicy:
    """The cron path may not hand-roll its own suppression."""

    def _read(self, rel):
        root = pathlib.Path(__file__).resolve().parents[2]
        return (root / rel).read_text(encoding="utf-8")

    def test_cron_scheduler_consults_shared_policy(self):
        src = self._read("cron/scheduler.py")
        assert "_cron_delivery_suppression_reason" in src

        from cron.scheduler import _cron_delivery_suppression_reason

        leak = (
            "I don't have a send_message tool in this environment. This cron "
            "response is delivered automatically to the configured target."
        )
        assert _cron_delivery_suppression_reason(leak, success=True) == "tool_mechanism"
        assert (
            _cron_delivery_suppression_reason(
                "Real report: three healthy engines remain.", success=True
            )
            is None
        )

    def test_failed_runs_are_not_screened_by_content(self):
        """A failed run routes to the operator alert path regardless of text."""
        from cron.scheduler import _cron_delivery_suppression_reason

        assert _cron_delivery_suppression_reason("[SILENT]", success=False) is None
