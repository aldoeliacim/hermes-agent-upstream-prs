"""Bounded retries must advance through real attempts, never guard checks.

The dispatch replay adapts Tarik Bencheikh's PR #73371 regression, using the
existing cooldown rather than an event-count protocol; PR #63268 makes local
PermissionError retries immediate instead of classifying them as provider auth.
"""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    clock = [5_000_000]
    monkeypatch.setattr(kb.time, "time", lambda: clock[0])
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    return clock


@pytest.mark.parametrize("lane", ["ready", "review"])
@pytest.mark.parametrize("outcome,ended_at", [
    (None, None), ("crashed", None), ("crashed", 4_999_999),
    ("crashed", 4_999_700), ("spawn_failed", 4_999_999),
    ("gave_up", 4_999_999), ("rate_limited", 4_999_999),
    ("rate_limited", 4_999_700),
])
@pytest.mark.parametrize("cooldown", [300, 0, -1])
@pytest.mark.parametrize("duplicate", [None, "recent_success", "active_pr"])
def test_guard_is_read_only_and_probes_preserve_lane_guards(
    board, monkeypatch, lane, outcome, ended_at, cooldown, duplicate,
):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", str(cooldown))
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="auth retry", assignee="alice")
        conn.execute(
            "UPDATE tasks SET status=?, last_failure_error=?, consecutive_failures=1 WHERE id=?",
            (lane, "Permission denied: invalid API key", tid),
        )
        if duplicate == "recent_success":
            conn.execute(
                "INSERT INTO task_runs(task_id,status,outcome,started_at,ended_at) "
                "VALUES (?,'completed','completed',?,?)",
                (tid, board[0] - 400, board[0] - 350),
            )
        if duplicate == "active_pr":
            kb.add_comment(conn, tid, "alice", "https://github.com/example/repo/pull/42")
        if outcome and duplicate is None:
            # An older quota run must not win a same-second tie or supply a
            # misleading timestamp for a newer legacy run with unknown age.
            conn.execute(
                "INSERT INTO task_runs(task_id,status,outcome,started_at,ended_at) "
                "VALUES (?,'rate_limited','rate_limited',?,?)",
                (tid, board[0] - 500, ended_at if ended_at is not None else board[0]),
            )
        if outcome:
            conn.execute(
                "INSERT INTO task_runs(task_id,status,outcome,started_at,ended_at) "
                "VALUES (?,'failed',?,?,?)",
                (tid, outcome, board[0] - 400, ended_at),
            )
        conn.commit()
        before = list(conn.iterdump())
        # Rate-limit's established early return intentionally has precedence;
        # auth probes, unlike quota probes, must fall through duplicate guards.
        # Negative configuration is invalid and retains the existing default;
        # zero disables the cooldown. Do not change the shared config policy.
        effective_cooldown = kb.DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS if cooldown < 0 else cooldown
        recent = ended_at is not None and board[0] - ended_at < effective_cooldown
        # The latest FINISHED run decides (upstream: ``ended_at IS NOT NULL``). A run with no end
        # time is not a finished attempt, so the older quota run inserted above stays the latest.
        quota_run_is_latest = outcome is not None and ended_at is None and duplicate is None
        if quota_run_is_latest:
            expected = "rate_limit_cooldown" if effective_cooldown > 0 else None
        elif outcome == "rate_limited":
            expected = "rate_limit_cooldown" if recent else None
        elif outcome in ("crashed", "spawn_failed", "gave_up"):
            # Upstream #117097 + PR 63268 provenance: a crashed / spawn-failed / gave-up run is
            # classified by its own outcome, never by auth words in its captured text.
            expected = duplicate if lane == "ready" else None
        else:
            expected = duplicate if lane == "ready" else None
        for _ in range(3):
            assert kb.check_respawn_guard(conn, tid, lane=lane) == expected
        assert list(conn.iterdump()) == before


@pytest.mark.parametrize("lane", ["ready", "review"])
@pytest.mark.parametrize("auth_history", [False, True])
def test_dispatch_retry_reaches_breaker_only_after_real_failures(
    board, all_assignees_spawnable, lane, auth_history,
):
    attempts = []

    def failing_spawn(task, workspace):
        attempts.append(task.id)
        raise PermissionError("Permission denied")

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="persistent-permission", assignee="alice")
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (lane, tid))
        if auth_history:
            conn.execute(
                "UPDATE tasks SET consecutive_failures=1, last_failure_error='invalid API key' WHERE id=?",
                (tid,),
            )
            # A worker that exited with an auth failure it reported itself (``blocked``), not a
            # crash: since upstream #117097 a crash's captured output is context, never a
            # blocker_auth diagnosis.
            conn.execute(
                "INSERT INTO task_runs(task_id,status,outcome,started_at,ended_at) "
                "VALUES (?,'failed','blocked',?,?)", (tid, board[0] - 10, board[0]),
            )
        conn.commit()
        first = kb.dispatch_once(conn, spawn_fn=failing_spawn, failure_limit=2)
        if auth_history:
            assert (tid, "blocker_auth") in first.respawn_guarded
            assert attempts == []
            # Checking/deferring repeatedly cannot manufacture attempts.
            kb.dispatch_once(conn, spawn_fn=failing_spawn, failure_limit=2)
            task = kb.get_task(conn, tid)
            assert task is not None
            assert task.consecutive_failures == 1
            assert attempts == []
            board[0] += 300
        else:
            assert attempts == [tid]
            task = kb.get_task(conn, tid)
            assert task is not None
            assert task.consecutive_failures == 1
            assert task.status == lane
        retried = kb.dispatch_once(conn, spawn_fn=failing_spawn, failure_limit=2)
        task = kb.get_task(conn, tid)
        assert task is not None
        assert attempts == [tid] * (1 if auth_history else 2)
        assert tid in retried.auto_blocked
        assert task.status == "blocked"
        assert task.consecutive_failures == 2
        assert task.claim_lock is None
        assert task.current_run_id is None
        assert conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()[0] == "gave_up"
