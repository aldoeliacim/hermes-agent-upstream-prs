"""Cron agent-turn context flag (cron-owned).

Homes the "am I a cron job's agent turn right now?" predicate as a task-local
``ContextVar`` inside the cron package. The layer originally landed this on
``gateway/session_context.py``; on this line it lives in cron so the cron
package owns its own turn state and the gateway/session module stays outside
the cron delivery/silence ownership boundary.

Consumers (wired by cron/scheduler.py, which is out of this port's scope):
  * ``set_cron_agent_turn(True)`` around a job's agent turn, reset in ``finally``.
  * the reply-decision gate consults ``is_cron_agent_turn()`` so a cron prompt
    is not ordered to call ``send_message``.
  * ``tools/cronjob_silent_tool.py`` requires ``is_cron_agent_turn()`` to refuse
    outside a cron turn.
"""

from __future__ import annotations

from contextvars import ContextVar

# True only while a cron job's agent turn is executing.  MUST be a ContextVar,
# not an env var: the cron ticker runs IN-PROCESS inside the gateway (the
# scheduler is an in-process daemon thread), so os.environ is shared with every
# live conversation.  A process-global "am I a cron?" flag set by one job would
# be observed as True by concurrent user turns and, once set, never reverts --
# silently changing behaviour for the rest of the process lifetime.
# cron/scheduler.py copies the context into its worker thread
# (contextvars.copy_context()), so the flag propagates exactly as far as the job
# and no further.
#
# Deliberately NOT environment-backed: this is purely the reply-gate's "is a
# cron turn executing right now" predicate, task-local by construction.
_CRON_AGENT_TURN: ContextVar = ContextVar("HERMES_CRON_AGENT_TURN", default=False)


def set_cron_agent_turn(active: bool = True):
    """Mark (or unmark) the current context as a cron job's agent turn.

    Returns the ContextVar token so the caller can ``reset_cron_agent_turn``
    in a ``finally``.  See ``_CRON_AGENT_TURN`` for why this is a ContextVar
    and not an environment variable.
    """
    return _CRON_AGENT_TURN.set(bool(active))


def reset_cron_agent_turn(token) -> None:
    """Restore the previous cron-turn flag value (pairs with ``set_``)."""
    try:
        _CRON_AGENT_TURN.reset(token)
    except Exception:
        # A token from another context is not resettable here; fall back to
        # the safe default rather than leaving the flag stuck on.
        _CRON_AGENT_TURN.set(False)


def is_cron_agent_turn() -> bool:
    """True while a cron job's agent turn is running in THIS context.

    Callers use this to skip machinery that only makes sense for live user
    turns (e.g. reply-tool gating) and to authorise the native cron silence
    tool, which must refuse to run outside a cron turn.
    """
    return bool(_CRON_AGENT_TURN.get())
