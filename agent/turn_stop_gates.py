"""Text-response stop gates for the conversation turn loop.

When the model stops with a text answer, three gates may instead append the answer as an
interim row plus a synthetic user-role nudge and continue the turn: verify-on-stop (#65919),
the ``pre_verify`` plugin hook after code edits, and the kanban worker terminal-tool guard.
Each keeps the candidate answer as a budget-exhaustion fallback
(``pending_verification_response``) and clears ``final_response`` so the finalizer can tell
this gate from error exits (#61631). Nothing here imports ``agent.conversation_loop`` at
module level (cycle).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.message_metadata import append_message

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class StopGateVerdict:
    """``continue_turn`` True → a nudge was appended; re-enter the turn loop with
    ``final_response=None`` and the pending-verification fields updated."""

    continue_turn: bool
    final_response: Any
    pending_verification_response: Any
    pending_verification_response_previewed: Any


def _verify_on_stop_nudge(agent) -> Optional[str]:
    try:
        from agent.verification_stop import (
            build_verify_on_stop_nudge, verify_on_stop_enabled
        )

        if verify_on_stop_enabled():
            return build_verify_on_stop_nudge(
                session_id=getattr(agent, "session_id", None),
                changed_paths=getattr(agent, "_turn_file_mutation_paths", set()),
                attempts=getattr(agent, "_verification_stop_nudges", 0),
            )
    except Exception:
        logger.debug("verification stop-loop check failed", exc_info=True)
    return None


def _pre_verify_nudge(agent, final_response, attempt: int) -> Optional[str]:
    """After code edits a registered ``pre_verify`` hook may keep the agent going one
    more turn; no default continuation cost."""
    _edited = sorted(getattr(agent, "_turn_file_mutation_paths", set()) or [])
    try:
        from agent.verify_hooks import max_verify_nudges
        from hermes_cli.lifecycle import has_hook
        from hermes_cli.plugins import get_pre_verify_continue_message

        if _edited and has_hook("pre_verify") and attempt < max_verify_nudges():
            # Posture is fixed for the session — resolve once + cache.
            coding = getattr(agent, "_resolved_is_coding", None)
            if coding is None:
                from agent.coding_context import is_coding_context
                coding = bool(is_coding_context(platform=getattr(agent, "platform", "") or ""))
                agent._resolved_is_coding = coding
            return get_pre_verify_continue_message(
                session_id=getattr(agent, "session_id", None) or "",
                platform=getattr(agent, "platform", "") or "",
                model=getattr(agent, "model", "") or "", coding=coding, attempt=attempt,
                final_response=final_response, changed_paths=_edited,
            )
    except Exception:
        logger.debug("pre_verify hook check failed", exc_info=True)
    return None


def _unfinished_todo_nudge(agent) -> Optional[str]:
    """One advisory reconciliation per turn for todos this turn left unfinished.

    Opt-out rides ``_stall_guards`` with the other continuation guards. The check is
    once per TURN, not per revision or per consecutive final: a refusal or blocker can
    end the next attempt with the plan still unfinished, and re-asking would loop.
    """
    if not bool(getattr(agent, "_stall_guards", True)):
        return None
    if getattr(agent, "_todo_reconciliation_attempted", False):
        return None
    baseline = getattr(agent, "_todo_baseline", None)
    store = getattr(agent, "_todo_store", None)
    if not isinstance(baseline, dict) or store is None:
        return None
    try:
        from agent.agent_runtime_helpers import build_unfinished_todo_nudge

        return build_unfinished_todo_nudge(baseline, store.snapshot())
    except Exception:
        logger.debug("unfinished-todo reconciliation check failed", exc_info=True)
        return None


def _kanban_stop_nudge(agent, messages) -> Optional[str]:
    """Workers must end with a terminal board tool (kanban_complete / kanban_block /
    kanban_request_review / kanban_request_changes); a narrated stop is recorded
    as protocol_violation, so nudge once or twice first."""
    try:
        from agent.kanban_stop import build_kanban_stop_nudge

        return build_kanban_stop_nudge(
            messages=messages, attempts=getattr(agent, "_kanban_stop_nudges", 0)
        )
    except Exception:
        logger.debug("kanban stop-loop check failed", exc_info=True)
        return None


def _reply_decision_nudge(agent) -> Optional[str]:
    """Tool-gated free-response turn that stopped with a text answer but has not yet
    delivered it: force one explicit delivery decision (call send_message, or record
    silence) before the fail-loud fallback silently discards the tail. Inert outside a
    tool-gated turn and once a decision was already made this turn (#921)."""
    try:
        from agent.reply_decision_stop import (
            build_reply_decision_nudge, reply_decision_nudge_enabled
        )
        from cron.agent_turn_context import is_cron_agent_turn
        from tools.approval import get_turn_delivery_state, is_reply_tool_gated_turn

        # Cron owns delivery of its final response and deliberately does not expose
        # send_message, so nudging a cron turn to "call send_message" gives it
        # contradictory orders (it burns turns, or fails the job).
        if is_cron_agent_turn() or not is_reply_tool_gated_turn() or not reply_decision_nudge_enabled():
            return None
        _state = get_turn_delivery_state()
        return build_reply_decision_nudge(
            tool_sends=getattr(_state, "tool_sends", 0),
            decided_silent=getattr(_state, "decided_silent", 0),
            attempts=getattr(agent, "_reply_decision_nudges", 0),
        )
    except Exception:
        logger.debug("reply-decision stop-loop check failed", exc_info=True)
    return None


def _append_interim_answer(agent, final_msg, messages, conversation_history, flush_fail_msg: str) -> None:
    """Real content: persist and emit as interim so the user sees the attempted answer;
    only the nudge is flagged synthetic (#65919)."""
    agent._emit_interim_assistant_message(final_msg)
    append_message(messages, final_msg)
    try:
        agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception:
        logger.debug(flush_fail_msg, exc_info=True)


def apply_stop_gates(
    agent: Any, final_msg: Dict[str, Any], *, final_response: Any, messages: List[Dict[str, Any]],
    conversation_history: Any, pending_verification_response: Any,
    pending_verification_response_previewed: Any,
) -> StopGateVerdict:
    """Run verify-on-stop → pre_verify hook → kanban stop guard, in that order. Nudges
    are user-role rows appended only after the assistant answer row, so role alternation
    holds. Hook lookups are imported lazily from their origin modules (tests patch them
    there)."""

    def _continue(nudge: str, flag: str) -> StopGateVerdict:
        """Append the synthetic nudge row and hand the turn back to the loop."""
        append_message(messages, {"role": "user", "content": nudge, flag: True})
        agent._session_messages = messages
        # Keep the answer only as a budget-exhaustion fallback; clear ``final_response`` so
        # the finalizer can tell this gate from error exits. Mark previewed only if the
        # candidate is reused (#61631).
        return StopGateVerdict(
            continue_turn=True, final_response=None,
            pending_verification_response=final_response,
            pending_verification_response_previewed=agent._interim_content_was_streamed(
                final_response or ""
            ),
        )

    _verify_nudge = _verify_on_stop_nudge(agent)
    if _verify_nudge:
        agent._verification_stop_nudges = getattr(agent, "_verification_stop_nudges", 0) + 1
        final_msg["finish_reason"] = "verification_required"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "verify-on-stop interim flush failed"
        )
        verdict = _continue(_verify_nudge, "_verification_stop_synthetic")
        # Internal nudge: stay silent on the terminal, debug-log only.
        logger.debug("verification stop-loop nudge issued (attempt %d)", agent._verification_stop_nudges)
        return verdict

    _attempt = getattr(agent, "_pre_verify_nudges", 0)
    _verify_nudge2 = _pre_verify_nudge(agent, final_response, _attempt)
    if _verify_nudge2:
        agent._pre_verify_nudges = _attempt + 1
        final_msg["finish_reason"] = "verify_hook_continue"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "pre_verify interim flush failed"
        )
        verdict = _continue(_verify_nudge2, "_pre_verify_synthetic")
        logger.debug("pre_verify nudge issued (attempt %d)", agent._pre_verify_nudges)
        return verdict

    _todo_nudge = _unfinished_todo_nudge(agent)
    if _todo_nudge:
        # Latch BEFORE continuing: one check per turn even if the model revises the plan.
        agent._todo_reconciliation_attempted = True
        # Neither half enters ``messages``. The candidate answer was superseded by this
        # continuation and the advisory is internal, so both ride the REQUEST COPY only
        # (see agent/turn_request_assembly.py) — keeping session history, incremental flushes,
        # compaction and later turns clean, while preserving assistant/user alternation
        # and leaving the cached prompt prefix and tool schemas untouched.
        agent._todo_reconciliation_messages = [
            {"role": "assistant", "content": final_response},
            {"role": "user", "content": _todo_nudge},
        ]
        logger.info("unfinished-todo reconciliation nudge issued")
        # The candidate is preserved as the budget-exhaustion fallback: if the continuation
        # runs out of iterations the user still gets the real reply, never a summary that
        # would imply the work finished.
        return StopGateVerdict(
            continue_turn=True,
            final_response=None,
            pending_verification_response=final_response,
            pending_verification_response_previewed=agent._interim_content_was_streamed(
                final_response or ""
            ),
        )

    _kanban_nudge = _kanban_stop_nudge(agent, messages)
    if _kanban_nudge:
        agent._kanban_stop_nudges = getattr(agent, "_kanban_stop_nudges", 0) + 1
        final_msg["finish_reason"] = "kanban_terminal_required"
        final_msg["_kanban_stop_synthetic"] = True
        append_message(messages, final_msg)
        verdict = _continue(_kanban_nudge, "_kanban_stop_synthetic")
        logger.info(
            "kanban stop-loop nudge issued (attempt %d) task=%s",
            agent._kanban_stop_nudges,
            os.environ.get("HERMES_KANBAN_TASK", ""),
        )
        agent._emit_diagnostic_status(
            "⚠️ Kanban worker tried to exit without a terminal board call "
            "(kanban_complete/kanban_request_review/kanban_block) — nudging to finish"
        )
        return verdict

    _reply_nudge = _reply_decision_nudge(agent)
    if _reply_nudge:
        agent._reply_decision_nudges = getattr(agent, "_reply_decision_nudges", 0) + 1
        final_msg["finish_reason"] = "reply_decision_required"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "reply-decision interim flush failed"
        )
        verdict = _continue(_reply_nudge, "_reply_decision_synthetic")
        logger.debug("reply-decision stop-loop nudge issued (attempt %d)", agent._reply_decision_nudges)
        return verdict
    return StopGateVerdict(
        continue_turn=False, final_response=final_response,
        pending_verification_response=pending_verification_response,
        pending_verification_response_previewed=pending_verification_response_previewed,
    )
