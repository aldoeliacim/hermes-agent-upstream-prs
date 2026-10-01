"""Delivery-ledger parity contract for the cron delivery pipeline.

Hermes delivers agent output through two independent paths that both must be
durable and recoverable:

    1. the live gateway turn   (gateway/run.py, gateway/stream_consumer.py)
    2. the cron scheduler      (cron/scheduler.py::_deliver_result)

Historically the durable delivery ledger was used only by the live path, which
is how a generated-but-undelivered cron report could vanish with no retry
(measured drift 2026-07-26). These tests assert the INVARIANT that the cron
delivery path records and finalizes a delivery obligation for every real
target, exactly like the live path -- never a snapshot of the current message
text.

Scope note: the shared free-text *suppression* policy and the reply-decision
stop gate (``gateway.response_filters.suppression_reason``,
``agent/turn_stop_gates.py``, the reply-gate cron marker in
``agent/conversation_loop.py``) are owned by separate branches (gateway
semantic / agent turn finalizer) and are asserted by their own parity tests.
This file pins only the cron delivery-ledger obligation contract.
"""


class TestCronDeliveryUsesTheDurableLedger:
    """Structural parity: the cron delivery path checkpoints the ledger."""

    def _read(self, rel):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        return (root / rel).read_text(encoding="utf-8")

    def test_cron_scheduler_uses_the_durable_ledger(self):
        """The ledger is what makes a generated-but-undelivered report
        recoverable. Cron skipped it entirely until 2026-07-26."""
        src = self._read("cron/scheduler_delivery.py")
        assert "delivery_ledger" in src
        for checkpoint in ("record_obligation", "mark_attempting", "mark_delivered", "mark_failed"):
            assert checkpoint in src, f"cron delivery must checkpoint {checkpoint}"
        # One record per target before its first send; one finalize per target
        # outcome (delivered/failed) plus the target-preparation-failed branch.
        assert src.count("_record_cron_delivery_obligation(") >= 2
        assert src.count("_finish_cron_delivery_obligation(") >= 3

    def test_cron_turn_marker_is_a_contextvar_not_process_global(self):
        """The cron-agent-turn marker must be a ContextVar: the cron ticker
        runs in-process inside the gateway, so an env-backed flag would also
        mark concurrent live conversations and never revert."""
        src = self._read("cron/agent_turn_context.py")
        # Defined as a ContextVar (task-local), not a module global or env var.
        assert "ContextVar(" in src
        # The flag must never be backed by a process-global env slot: a read or
        # write of HERMES_CRON_AGENT_TURN via os.environ would leak the cron
        # disposition into concurrent live turns and never revert. (The module
        # comment mentions os.environ only to explain why it is NOT used, so we
        # pin the specific env-key access rather than the bare substring.)
        for forbidden in (
            'os.environ["HERMES_CRON_AGENT_TURN"]',
            "os.environ['HERMES_CRON_AGENT_TURN']",
            'os.environ.get("HERMES_CRON_AGENT_TURN"',
            "os.environ.get('HERMES_CRON_AGENT_TURN'",
            'os.getenv("HERMES_CRON_AGENT_TURN"',
            "os.getenv('HERMES_CRON_AGENT_TURN'",
        ):
            assert forbidden not in src, (
                "backing the cron-turn flag with a process-global env var "
                "disables the reply gate for concurrent live turns and never "
                "reverts"
            )


def test_cron_delivery_obligation_records_attempt_and_terminal_outcomes(monkeypatch):
    from cron.scheduler_delivery import (
        _finish_cron_delivery_obligation,
        _record_cron_delivery_obligation,
    )
    from gateway import delivery_ledger

    calls = []
    monkeypatch.setattr(delivery_ledger, "ledger_enabled", lambda: True)
    monkeypatch.setattr(delivery_ledger, "compute_obligation_id", lambda *_args: "oid")
    monkeypatch.setattr(
        delivery_ledger,
        "record_obligation",
        lambda **kwargs: calls.append(("record", kwargs)),
    )
    monkeypatch.setattr(
        delivery_ledger,
        "mark_attempting",
        lambda obligation_id: calls.append(("attempt", obligation_id)),
    )
    monkeypatch.setattr(
        delivery_ledger,
        "mark_delivered",
        lambda obligation_id: calls.append(("delivered", obligation_id)),
    )
    monkeypatch.setattr(
        delivery_ledger,
        "mark_failed",
        lambda obligation_id, error: calls.append(("failed", obligation_id, error)),
    )

    job = {"id": "job-1"}
    target = {"platform": "telegram", "chat_id": "42", "thread_id": "7"}
    obligation_id = _record_cron_delivery_obligation(job, target, "owed report")
    _finish_cron_delivery_obligation(job, obligation_id, delivered=True)
    _finish_cron_delivery_obligation(
        job, obligation_id, delivered=False, error="transport refused"
    )

    assert obligation_id == "oid"
    assert calls[0] == (
        "record",
        {
            "obligation_id": "oid",
            "session_key": "cron:job-1",
            "platform": "telegram",
            "chat_id": "42",
            "thread_id": "7",
            "content": "owed report",
        },
    )
    assert calls[1:] == [
        ("attempt", "oid"),
        ("delivered", "oid"),
        ("failed", "oid", "transport refused"),
    ]


def test_cron_delivery_obligation_is_a_noop_when_ledger_disabled(monkeypatch):
    """Ledger off -> no obligation id, no bookkeeping, delivery never blocked."""
    from cron.scheduler_delivery import (
        _finish_cron_delivery_obligation,
        _record_cron_delivery_obligation,
    )
    from gateway import delivery_ledger

    calls = []
    monkeypatch.setattr(delivery_ledger, "ledger_enabled", lambda: False)
    for name in ("record_obligation", "mark_attempting", "mark_delivered", "mark_failed"):
        monkeypatch.setattr(
            delivery_ledger, name,
            lambda *a, **k: calls.append(name),
        )

    job = {"id": "job-2"}
    target = {"platform": "telegram", "chat_id": "9", "thread_id": None}
    obligation_id = _record_cron_delivery_obligation(job, target, "x")
    assert obligation_id is None
    # A None obligation id must finalize as a no-op, never raise.
    _finish_cron_delivery_obligation(job, obligation_id, delivered=True)
    assert calls == []
