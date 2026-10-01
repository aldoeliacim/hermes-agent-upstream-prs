"""Leaf constants shared by ``gateway/run.py`` and its ``run_*`` mixin modules.

Kept import-cycle free (imports nothing from ``gateway.run``) because these values
are used as default-argument sentinels, which must resolve at ``def`` time.
"""

from typing import Optional

# Sentinel for "caller did not pass metadata" vs "caller passed None".
_UNSET = object()


# ---- Interruption / resume recovery notes ---------------------------------
# These carry the resume-pending contract: after a gateway restart or shutdown
# interrupted a turn, the history contains a restart/shutdown command that has ALREADY
# run and tool outputs whose outcomes may be unknown. The note tells the model not to
# re-execute recorded work, not to treat a startup notification as fresh authorization,
# and (on a shared group) not to narrate the interruption to the room.

_INTERRUPTED_WORK_GUIDANCE = (
    "Do NOT re-execute tool calls whose results are already recorded. "
    "Use recorded results as evidence; if an interrupted action's outcome is "
    "unknown, inspect its current state before retrying it. Preserve unfinished "
    "authorized work unless the user explicitly cancels, pauses, or replaces "
    "it. A side question or additional request does not by itself cancel "
    "earlier work. After addressing the new input, continue the remaining "
    "authorized work or report a concrete blocker; do not infer fresh "
    "authorization from old history alone."
)


_INTERRUPTED_WORK_WAIT_GUIDANCE = (
    "Do NOT re-execute old tool calls. Any unfinished work remains pending, "
    "not completed or cancelled. This startup notification is not a new "
    "authorization to resume it; wait for real user direction."
)


def _resume_reason_phrase(reason: Optional[str]) -> str:
    """Map a session ``resume_reason`` marker to its human-readable phrase."""
    if reason == "restart_timeout":
        return "a gateway restart"
    if reason == "shutdown_timeout":
        return "a gateway shutdown"
    return "a gateway interruption"


def _build_interruption_system_note(
    reason: Optional[str],
    has_new_message: bool,
    group_mode: bool = False,
) -> str:
    """Build the ``[System note: ...]`` resume/recovery text.

    Centralizes the two near-duplicate resume-note construction sites (the primary
    ``resume_pending`` branch and the empty-message safety net) so a future wording
    change or group-mode variant only needs one edit point. ``reason`` is the session's
    ``resume_reason`` marker; ``has_new_message`` selects the guidance clause — address
    the user's new message, or (for the synthesized empty auto-resume turn) report
    recovery and ask what's next.

    ``group_mode`` (True when the turn's reply policy is not ``dm``) swaps the recovery
    guidance for owner-routing guidance: the model must NOT narrate the
    restart/interruption into a shared group; any recovery/status report goes to the
    owner's private channel via ``send_message``. Defaults to False so DM output is
    byte-identical to before this variant existed.

    Returns only the bracketed note; the caller appends any real message.
    """
    phrase = _resume_reason_phrase(reason)
    tail_guidance = (
        _INTERRUPTED_WORK_GUIDANCE if has_new_message
        else _INTERRUPTED_WORK_WAIT_GUIDANCE
    )
    if group_mode:
        guidance = (
            "Recovery context only — do NOT narrate this restart/interruption "
            "into the group. If the new message below needs a substantive "
            "reply, answer it normally; send any recovery/status report to the "
            "owner's private channel via send_message if needed."
        )
    elif has_new_message:
        guidance = (
            "Address the user's NEW message below FIRST and focus "
            "on what the user is asking now."
        )
    else:
        guidance = (
            "Report to the user that the session was restored "
            "successfully and ask what they would like to do next."
        )
    return (
        f"[System note: The previous turn was interrupted by "
        f"{phrase}; the gateway is now back online. "
        f"Any restart/shutdown command in the history has already "
        f"run — do NOT re-execute or verify it. {guidance} "
        f"{tail_guidance}]"
    )


def _resolve_resume_note_kind(
    is_resume_pending: bool,
    has_fresh_tool_tail: bool,
    message_is_empty: bool,
) -> str:
    """Classify which auto-continue system note applies to an inbound turn.

    Formalizes the historical ``if/elif/if`` dispatch at the inbound auto-continue site
    into one table-testable predicate:

    - ``"resume"``     — a fresh ``resume_pending`` session: emit the reason-aware
                         resume/recovery note.
    - ``"tool_tail"``  — no fresh resume mark, but the transcript ends in a fresh
                         interrupted tool result.
    - ``"safety_net"`` — neither fresh signal fired yet the turn carries an empty
                         message; the caller still guards this on the raw
                         ``resume_pending`` marker so a legitimately empty non-resume
                         turn (e.g. a captionless image) is untouched.
    - ``"none"``       — ordinary turn; no system note.

    The ``safety_net`` vs ``resume`` split is the "two freshness signals disagree" edge
    case documented inline at the call site.
    """
    if is_resume_pending:
        return "resume"
    if has_fresh_tool_tail:
        return "tool_tail"
    if message_is_empty:
        return "safety_net"
    return "none"


def _build_tool_tail_system_note() -> str:
    """Build the ``[System note: ...]`` text for the interrupted-tool-tail case.

    Distinct wording from ``_build_interruption_system_note``: there is no
    restart/shutdown reason, only recorded tool outputs to reconcile with the user's new
    message without replaying actions or discarding work. Returns only the bracketed
    note.
    """
    return (
        "[System note: A new message has arrived. The conversation "
        "history contains pending tool outputs from an interrupted turn. "
        "Address the user's NEW message below FIRST. "
        f"{_INTERRUPTED_WORK_GUIDANCE}]"
    )
