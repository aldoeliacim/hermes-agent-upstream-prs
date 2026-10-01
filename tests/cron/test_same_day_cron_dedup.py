"""An at-most-once-per-local-day cron must not fire twice on the same local date.

Timezone-change / DST / catch-up anomaly: a once-per-local-day cron whose stored
next_run_at was computed under a different UTC offset can look due a SECOND time
on the same local calendar date after the offset is reinterpreted (an old UTC
next_run_at fires in the morning; the same expression, re-rendered in local
wall-clock, looks due again later that day). The guard stamps the occurrence the
scheduler is ABOUT to run (``last_scheduled_run_at``, written by
``advance_next_runs`` — the batch function the real tick calls), then a due-scan
check skips any further same-local-day occurrence and advances past it.

Invariants proven here:
  1. classification is croniter-derived: daily/weekly/monthly = once-per-day;
     ``0 */3 * * *`` and ``0 13,19 * * *`` (multi-daily) are left alone.
  2. advance_next_runs stamps last_scheduled_run_at BEFORE advancing.
  3. a second same-local-day occurrence is skipped and next_run_at advances.
  4. a genuinely new local date is NOT skipped.
  5. an explicit manual ``cron run`` is NEVER suppressed (positive control).
  6. the scheduler-only stamp (not last_run_at) drives the guard, so a manual
     run earlier the same day cannot mask the real scheduled occurrence.
  7. END-TO-END, real lifecycle: a manual `cron run` earlier the same local day
     (trigger_job -> advance_next_runs -> mark_job_run) must NOT stamp the
     scheduler-only last_scheduled_run_at, so the genuine scheduled occurrence
     later that day is still fired.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("croniter")

import hermes_time
from cron.jobs import (
    _DueJob,
    _DueScan,
    _cron_runs_at_most_once_per_local_day,
    _ensure_aware,
    _skip_duplicate_same_day_cron,
    advance_next_runs,
    mark_job_run,
    save_jobs,
    load_jobs,
    trigger_job,
)

TORONTO = ZoneInfo("America/Toronto")


@pytest.fixture
def toronto(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Toronto")
    hermes_time.reset_cache()
    yield
    hermes_time.reset_cache()


def _due(job, now):
    """Build a _DueJob the way _evaluate_due_job does."""
    next_run = job["next_run_at"]
    raw = datetime.fromisoformat(next_run)
    scan = _DueScan(raw_jobs=[dict(job)], now=now)
    return _DueJob(job, scan, next_run, raw, _ensure_aware(raw))


# ---- (1) classification -----------------------------------------------------

@pytest.mark.parametrize("expr", ["0 9 * * *", "30 8 * * 1", "0 0 1 * *"])
def test_daily_weekly_monthly_are_once_per_local_day(toronto, expr):
    ref = datetime(2026, 6, 1, 12, 0, tzinfo=TORONTO)
    assert _cron_runs_at_most_once_per_local_day({"kind": "cron", "expr": expr}, ref) is True


@pytest.mark.parametrize("expr", ["0 */3 * * *", "0 13,19 * * *", "*/15 * * * *"])
def test_multi_daily_schedules_are_not_gated(toronto, expr):
    ref = datetime(2026, 6, 1, 12, 0, tzinfo=TORONTO)
    assert _cron_runs_at_most_once_per_local_day({"kind": "cron", "expr": expr}, ref) is False


def test_interval_kind_is_never_gated(toronto):
    ref = datetime(2026, 6, 1, 12, 0, tzinfo=TORONTO)
    assert _cron_runs_at_most_once_per_local_day({"kind": "interval", "minutes": 1440}, ref) is False


# ---- (2) advance_next_runs stamps the occurrence ----------------------------

def test_advance_next_runs_stamps_last_scheduled_run_at(toronto, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import cron.jobs as jobs
    monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    (tmp_path / "cron").mkdir(parents=True, exist_ok=True)

    scheduled = "2026-06-01T09:00:00-04:00"
    job = {
        "id": "j1",
        "name": "morning",
        "enabled": True,
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
        "next_run_at": scheduled,
        "repeat": {"times": None, "completed": 0},
    }
    save_jobs([job])
    advanced = advance_next_runs(["j1"])
    assert advanced == 1
    reloaded = load_jobs()[0]
    assert reloaded["last_scheduled_run_at"] == _ensure_aware(
        datetime.fromisoformat(scheduled)
    ).isoformat(), "the occurrence about to run must be stamped BEFORE advancing"
    assert reloaded["next_run_at"] != scheduled, "next_run_at must advance"


# ---- (3)+(4) skip duplicate / allow new date --------------------------------

def test_second_occurrence_same_local_day_is_skipped(toronto):
    now = datetime(2026, 6, 1, 18, 0, tzinfo=TORONTO)
    job = {
        "id": "j1",
        "name": "morning",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
        "next_run_at": "2026-06-01T09:00:00-04:00",
        # already scheduled earlier THE SAME local date
        "last_scheduled_run_at": "2026-06-01T09:00:00-04:00",
    }
    d = _due(job, now)
    assert _skip_duplicate_same_day_cron(d) is True
    # next_run_at advanced to a LATER local date
    persisted = d.scan.find("j1")
    assert persisted["next_run_at"] is not None
    advanced = datetime.fromisoformat(persisted["next_run_at"]).astimezone(TORONTO)
    assert advanced.date() > datetime(2026, 6, 1).date()


def test_new_local_date_is_not_skipped(toronto):
    now = datetime(2026, 6, 2, 9, 0, tzinfo=TORONTO)
    job = {
        "id": "j1",
        "name": "morning",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
        "next_run_at": "2026-06-02T09:00:00-04:00",
        "last_scheduled_run_at": "2026-06-01T09:00:00-04:00",  # yesterday
    }
    d = _due(job, now)
    assert _skip_duplicate_same_day_cron(d) is False


def test_no_stamp_means_no_skip(toronto):
    now = datetime(2026, 6, 1, 18, 0, tzinfo=TORONTO)
    job = {
        "id": "j1",
        "name": "morning",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
        "next_run_at": "2026-06-01T09:00:00-04:00",
    }
    d = _due(job, now)
    assert _skip_duplicate_same_day_cron(d) is False


def test_multi_daily_second_occurrence_is_allowed(toronto):
    """A schedule that legitimately fires many times a day is never deduped."""
    now = datetime(2026, 6, 1, 19, 0, tzinfo=TORONTO)
    job = {
        "id": "j1",
        "name": "thrice",
        "schedule": {"kind": "cron", "expr": "0 13,19 * * *"},
        "next_run_at": "2026-06-01T19:00:00-04:00",
        "last_scheduled_run_at": "2026-06-01T13:00:00-04:00",  # earlier same day
    }
    d = _due(job, now)
    assert _skip_duplicate_same_day_cron(d) is False


# ---- (5) DST positive control -----------------------------------------------

def test_dst_fallback_day_duplicate_is_skipped(toronto):
    """Fall-back (Nov 1 2026): the 09:00 occurrence, once scheduled, is not re-fired
    later the same local date even though the offset changed mid-day."""
    now = datetime(2026, 11, 1, 15, 0, tzinfo=TORONTO)  # after the 02:00 fall-back
    job = {
        "id": "j1",
        "name": "morning",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
        "next_run_at": "2026-11-01T09:00:00-05:00",  # reinterpreted post-DST offset
        "last_scheduled_run_at": "2026-11-01T09:00:00-04:00",  # fired pre-DST offset, same date
    }
    d = _due(job, now)
    assert _skip_duplicate_same_day_cron(d) is True


# ---- (6) manual-run positive control ----------------------------------------

def test_manual_run_earlier_today_does_not_suppress_scheduled(toronto):
    """The guard reads last_scheduled_run_at, NOT last_run_at: a human `cron run`
    earlier today (which stamps last_run_at) must not mask the scheduled fire."""
    now = datetime(2026, 6, 1, 9, 0, tzinfo=TORONTO)
    job = {
        "id": "j1",
        "name": "morning",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
        "next_run_at": "2026-06-01T09:00:00-04:00",
        "last_run_at": "2026-06-01T07:00:00-04:00",  # manual run earlier today
        # no last_scheduled_run_at -> scheduler has NOT fired this date yet
    }
    d = _due(job, now)
    assert _skip_duplicate_same_day_cron(d) is False


# ---- (7) END-TO-END real-lifecycle regression -------------------------------

def test_manual_run_lifecycle_does_not_suppress_scheduled_occurrence(
    toronto, monkeypatch, tmp_path
):
    """Full real lifecycle: a manual run earlier today must not eat the scheduled fire.

    Reproduces the review blocker: a manual `cron run` goes through the SAME
    ``advance_next_runs`` batch the scheduler uses. If that batch stamps the
    scheduler-only ``last_scheduled_run_at`` for the manual occurrence, the
    genuine scheduled occurrence later the same local day is then wrongly
    skipped as a duplicate.

    Drives the actual functions (trigger_job -> advance_next_runs ->
    mark_job_run), not a hand-set ``last_run_at`` fixture, then asserts the
    real scheduled occurrence still fires.
    """
    import cron.jobs as jobs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    (tmp_path / "cron").mkdir(parents=True, exist_ok=True)

    # Mutable clock so trigger/advance/mark all observe the manual-run instant.
    clock = {"now": datetime(2026, 6, 1, 7, 0, tzinfo=TORONTO)}
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock["now"])

    # Daily 09:00 job, next scheduled occurrence today at 09:00.
    scheduled_today = "2026-06-01T09:00:00-04:00"
    job = {
        "id": "j1",
        "name": "morning",
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
        "next_run_at": scheduled_today,
        "repeat": {"times": None, "completed": 0},
    }
    save_jobs([job])

    # 1) Operator triggers a manual run at 07:00. trigger_job re-anchors
    #    next_run_at = manual_run_at = 07:00.
    triggered = trigger_job("j1")
    assert triggered is not None
    manual_at = load_jobs()[0]["next_run_at"]
    assert load_jobs()[0]["manual_run_at"] == manual_at

    # 2) The tick advances the due set BEFORE running (this is where the batch
    #    would wrongly stamp last_scheduled_run_at for the manual occurrence).
    advance_next_runs(["j1"])

    # 3) The manual run completes; mark_job_run stamps last_run_at and clears
    #    the manual marker.
    mark_job_run("j1", True)

    after_manual = load_jobs()[0]
    # A manual occurrence must NEVER stamp the scheduler-only field.
    assert after_manual.get("last_scheduled_run_at") is None, (
        "manual run stamped the scheduler-only last_scheduled_run_at; the "
        "genuine scheduled occurrence would be wrongly suppressed"
    )
    # mark_job_run re-anchored next_run_at back onto the cron grid: 09:00 today.
    assert (
        datetime.fromisoformat(after_manual["next_run_at"]).astimezone(TORONTO).date()
        == datetime(2026, 6, 1).date()
    )

    # 4) Clock advances to the genuine scheduled occurrence (09:00). The guard
    #    must NOT skip it — the scheduler has not fired this local date yet.
    now_scheduled = datetime(2026, 6, 1, 9, 0, tzinfo=TORONTO)
    d = _due(after_manual, now_scheduled)
    assert _skip_duplicate_same_day_cron(d) is False, (
        "scheduled occurrence suppressed by an earlier manual run"
    )
