"""Isolation contract for the cron-agent-turn marker.

Architectural context: the cron ticker runs IN-PROCESS inside the gateway
(gateway/run.py -> InProcessCronScheduler, daemon thread). Anything that marks
"this is a cron run" with process-global state (os.environ) is therefore
visible to every concurrent live conversation, and -- since nothing clears it
-- stays visible for the rest of the process lifetime.

A first attempt at fixing the news-cron incident used
``os.environ["HERMES_CRON_SESSION"]``, which is exactly that mistake: it would
have disabled the reply-decision gate for real user turns too. These tests pin
the isolation properties so the ContextVar cannot silently regress back into a
global.

The marker lives in the cron-owned module ``cron.agent_turn_context`` (not
``gateway.session_context``): the cron package owns its own turn state so the
gateway/session module stays outside the cron delivery/silence boundary.
"""
import concurrent.futures
import contextvars
import threading

from cron.agent_turn_context import (
    is_cron_agent_turn,
    reset_cron_agent_turn,
    set_cron_agent_turn,
)


def test_defaults_to_false():
    """A plain context is never mistaken for a cron turn."""
    assert is_cron_agent_turn() is False


def test_set_then_reset_restores_previous_value():
    token = set_cron_agent_turn(True)
    assert is_cron_agent_turn() is True
    reset_cron_agent_turn(token)
    assert is_cron_agent_turn() is False


def test_flag_does_not_leak_into_a_sibling_thread():
    """THE regression this guards: a cron run must not mark other threads.

    The gateway serves live conversations on other threads while the cron
    ticker runs. If the marker were process-global, this assertion would fail
    and the reply gate would be silently disabled for real users.
    """
    set_cron_agent_turn(True)
    try:
        seen = {}
        done = threading.Event()

        def sibling():
            # A brand-new thread starts from an empty context.
            seen["value"] = is_cron_agent_turn()
            done.set()

        t = threading.Thread(target=sibling)
        t.start()
        done.wait(timeout=5)
        t.join(timeout=5)
        assert seen.get("value") is False
    finally:
        set_cron_agent_turn(False)


def test_flag_propagates_into_a_copied_context():
    """cron/scheduler.py hands the job to a worker thread via
    ``contextvars.copy_context()``; the marker must travel with it, otherwise
    the gate fires inside the very turn we are protecting."""
    set_cron_agent_turn(True)
    try:
        ctx = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(ctx.run, is_cron_agent_turn).result(timeout=5) is True
    finally:
        set_cron_agent_turn(False)


def test_concurrent_contexts_do_not_interfere():
    """Two jobs/turns running side by side each see only their own marker."""
    results = {}

    def marked():
        token = set_cron_agent_turn(True)
        try:
            return is_cron_agent_turn()
        finally:
            reset_cron_agent_turn(token)

    def unmarked():
        return is_cron_agent_turn()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_marked = pool.submit(contextvars.copy_context().run, marked)
        f_plain = pool.submit(contextvars.copy_context().run, unmarked)
        results["marked"] = f_marked.result(timeout=5)
        results["plain"] = f_plain.result(timeout=5)

    assert results["marked"] is True
    assert results["plain"] is False


def test_reset_with_foreign_token_falls_back_to_false():
    """A token created in another context must not leave the flag stuck on.

    Failing open (False) is the safe direction: the reply gate stays active
    for live conversations.
    """
    ctx = contextvars.copy_context()
    foreign_token = ctx.run(set_cron_agent_turn, True)

    set_cron_agent_turn(True)
    reset_cron_agent_turn(foreign_token)
    assert is_cron_agent_turn() is False


def test_marker_is_not_backed_by_os_environ():
    """Explicitly pin that the flag is NOT a process-global env var.

    This is the property the first fix attempt got wrong.
    """
    import os

    token = set_cron_agent_turn(True)
    try:
        assert os.environ.get("HERMES_CRON_AGENT_TURN") is None
    finally:
        reset_cron_agent_turn(token)
