"""No-tool-call (final text) branch of the conversation turn loop: empty/think-only recovery,
intent-ack / stall-guard continuation, length-continuation joining, dropped-tool-call
re-prompt, scaffolding pop, stop gates, then the durable final flush. Extracted from
``run_conversation``; nothing here imports ``agent.conversation_loop`` at module level
(cycle) — loop-internal nudge constants resolve lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, Optional

from agent.message_metadata import append_message
from agent.turn_empty_response import recover_empty_response
from agent.turn_stop_gates import apply_stop_gates

logger = logging.getLogger("agent.conversation_loop")

# Some summarizer/context routes return their internal handoff as the assistant
# answer. A whole XML envelope alone is ambiguous — HTML legitimately uses a
# <summary> element — so require at least two handoff-style Markdown headings,
# including one that denotes state or future work. An explicit user request for
# an HTML summary element always wins.
_INTERNAL_SUMMARY_ENVELOPE = re.compile(
    r"^\s*<summary(?:\s[^>]*)?>[\s\S]*</summary>\s*$", re.IGNORECASE,
)
_INTERNAL_SUMMARY_HEADING = re.compile(
    r"(?im)^\s*(?:#{1,6}\s+|<h[1-6]\b[^>]*>\s*)"
    r"(overview|summary|task|progress|current\s+state|next\s+steps|completed|remaining|open\s+tasks)\b"
    r"(?:\s*</h[1-6]>)?"
)
# A whole fenced XML handoff is still a handoff, not a user-facing code sample.
# Only unwrap when the fence encloses the ENTIRE answer; examples embedded in a
# real answer remain examples and do not trigger the recovery path.
_WHOLE_XML_FENCE = re.compile(
    r"\A\s*```(?:xml|html)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```[ \t]*\s*\Z",
    re.IGNORECASE,
)
_INTERNAL_SUMMARY_STATE_HEADINGS = {
    "progress", "current state", "next steps", "remaining", "open tasks",
}
_EXPLICIT_SUMMARY_ELEMENT_REQUEST = re.compile(
    r"(?is)(?:\b(?:return|respond|output|write|generate|provide|emit|create)\b.{0,120}<\s*summary\b"
    r"|<\s*summary\b.{0,120}\b(?:html|element|tag|markup)\b)"
)
_INTERNAL_SUMMARY_RECOVERY_NUDGE = (
    "Your previous output was an internal context handoff, not an answer to the user. "
    "Do not repeat, quote, or discuss it. Answer the latest real user message directly."
)
_INTERNAL_SUMMARY_SAFE_FAILURE = "I couldn't complete that response correctly. Please try again."
_INTERNAL_SUMMARY_MAX_RETRIES = 2

# The same handoff can arrive as a PREFIX (envelope, then prose, then a tool call written
# as text), which the whole-response match lets through. The handoff is a BLOCK: the
# opening tag starts the response on its own line and the closing tag sits on its own
# line. Inline ``<summary>label</summary> prose`` (HTML discussion) never has that shape,
# and the block must still carry handoff headings (same rule as the whole envelope).
_LEADING_SUMMARY_BLOCK = re.compile(
    r"\A\s*<summary(?:\s[^>]*)?>[ \t]*\r?\n([\s\S]*?)\r?\n[ \t]*</summary>", re.IGNORECASE,
)
# Internal handoffs are structural, not identified by a magic English sentence.
# Providers may preface the block with acknowledgements or append a trailing note;
# both are unsafe if a top-level multiline <summary> contains state headings.
# Limit the preface to prevent matching an XML example deep in a normal answer.
_PREFATORY_SUMMARY_BLOCK = re.compile(
    r"\A(?P<preface>[\s\S]{1,600}?)^[ \t]*<summary(?:\s[^>]*)?>[ \t]*\r?\n"
    r"(?P<body>[\s\S]*?)\r?\n[ \t]*</summary>(?:[ \t]*\r?\n[\s\S]*)?[ \t]*\Z",
    re.IGNORECASE | re.MULTILINE,
)
# A tool call the model wrote as TEXT in a foreign harness dialect instead of the native
# function-calling channel: a line-anchored ``<tag>{json}</tag>`` call
# (``<execute_bash>{"CommandLine": ...}</execute_bash>``) or XML call markup. The call
# never executes, and the markup (with its internal commands and paths) would be shown
# verbatim. Fenced code is excluded, so a requested example still gets through.
_TEXT_TOOL_MARKUP = re.compile(
    r"^[ \t]*<([a-z][a-z0-9_]{2,40})>[ \t]*\r?\n?[ \t]*\{[\s\S]*?\}[ \t]*\r?\n?[ \t]*</\1>"
    r"|^[ \t]*<(?:function_calls|tool_call)>"
    r"|^[ \t]*<invoke\s+name=",
    re.IGNORECASE | re.MULTILINE,
)
_FENCED_CODE = re.compile(r"^[ \t]*(```|~~~)[\s\S]*?^[ \t]*\1[ \t]*$", re.MULTILINE)
_ARTIFACT_RECOVERY_NUDGES = {
    "summary_envelope": _INTERNAL_SUMMARY_RECOVERY_NUDGE,
    "text_tool_markup": (
        "Your previous output wrote a tool call as text markup instead of using the native "
        "function-calling interface, so nothing executed and the markup would have been "
        "shown to the user. Do not repeat or quote that markup. If you need a tool, call it "
        "through the native tool interface; otherwise answer the latest real user message "
        "directly."
    ),
}


def _has_handoff_headings(text: str) -> bool:
    headings = [" ".join(h.lower().split()) for h in _INTERNAL_SUMMARY_HEADING.findall(text)]
    return len(headings) >= 2 and any(h in _INTERNAL_SUMMARY_STATE_HEADINGS for h in headings)


def is_internal_summary_artifact(text: Any, *, user_message: Any = "") -> bool:
    """Whether *text* is a whole internal handoff envelope, not requested HTML."""
    if not isinstance(text, str) or not _INTERNAL_SUMMARY_ENVELOPE.fullmatch(text):
        return False
    request = user_message if isinstance(user_message, str) else ""
    if request and _EXPLICIT_SUMMARY_ELEMENT_REQUEST.search(request):
        return False
    return _has_handoff_headings(text)


def internal_artifact_kind(text: Any, *, user_message: Any = "") -> Optional[str]:
    """Classify *text* as internal scaffolding that must never reach a user.

    ``"summary_envelope"``: a compaction handoff, either the whole response or a leading
    block envelope followed by anything, in both cases carrying handoff headings.
    ``"text_tool_markup"``: a tool call written as text outside fenced code.
    ``None`` otherwise. Structural, not keyword-based: prose about summaries, HTML tags or
    tool syntax, fenced examples, and an explicitly requested ``<summary>`` element stay
    deliverable.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    request = user_message if isinstance(user_message, str) else ""
    explicit_html = bool(request and _EXPLICIT_SUMMARY_ELEMENT_REQUEST.search(request))
    if not explicit_html:
        fenced = _WHOLE_XML_FENCE.fullmatch(text)
        if fenced and is_internal_summary_artifact(fenced.group("body")):
            return "summary_envelope"
        if is_internal_summary_artifact(text):
            return "summary_envelope"
        # Whole fenced handoffs were handled above. Any remaining fenced
        # blocks are examples inside a real response, not agent handoffs.
        unfenced = _FENCED_CODE.sub("", text)
        block = _LEADING_SUMMARY_BLOCK.match(unfenced)
        if block and _has_handoff_headings(block.group(1)):
            return "summary_envelope"
        prefatory = _PREFATORY_SUMMARY_BLOCK.match(unfenced)
        if prefatory and _has_handoff_headings(prefatory.group("body")):
            return "summary_envelope"
    if _TEXT_TOOL_MARKUP.search(_FENCED_CODE.sub("", text)):
        return "text_tool_markup"
    return None


# Ephemeral retry scaffolding rows popped before the final answer becomes durable.
_EPHEMERAL_SCAFFOLDING_FLAGS = (
    "_thinking_prefill", "_empty_recovery_synthetic", "_empty_terminal_sentinel",
    "_dropped_toolcall_nudge", "_internal_summary_recovery",
)


@dataclass
class FinalResponseVerdict:
    """``action``: ``"break"`` (turn ends with ``final_response``), ``"continue"`` (a
    continuation/re-prompt/stop-gate asked for another API call) or ``"return"``
    (``result`` is the turn's result dict). The other fields are the loop locals rebound."""

    action: str
    active_system_prompt: Any
    final_response: Any
    _turn_exit_reason: Any
    _preflight_compression_blocked: Any
    codex_ack_continuations: Any
    truncated_response_parts: Any
    length_continue_retries: Any
    _pending_verification_response: Any
    _pending_verification_response_previewed: Any
    result: Optional[Dict[str, Any]] = None


def finish_text_response(
    agent: Any, *, assistant_message: Any, response: Any, finish_reason: Any, messages: Any,
    api_messages: Any, conversation_history: Any, api_call_count: Any, user_message: Any,
    active_system_prompt: Any, final_response: Any, _turn_exit_reason: Any,
    _preflight_compression_blocked: Any, codex_ack_continuations: Any,
    truncated_response_parts: Any, length_continue_retries: Any,
    _pending_verification_response: Any, _pending_verification_response_previewed: Any,
) -> FinalResponseVerdict:
    """Finish (or defer) a text-only assistant response in the original guard order. Every
    continuation path sets ``final_response = None`` so an acknowledgment never suppresses
    iteration-limit summarization; the final message is appended and flushed only after the
    stop gates accept it."""
    from agent.conversation_loop import (
        _CODEX_ACK_CONTINUATION_NUDGE, _DEGENERATE_FINAL_NUDGE, _DROPPED_TOOLCALL_NUDGE_CONTENT,
        _join_truncated_parts
    )

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> FinalResponseVerdict:
        return FinalResponseVerdict(
            action=action, active_system_prompt=active_system_prompt, final_response=final_response,
            _turn_exit_reason=_turn_exit_reason,
            _preflight_compression_blocked=_preflight_compression_blocked,
            codex_ack_continuations=codex_ack_continuations,
            truncated_response_parts=truncated_response_parts,
            length_continue_retries=length_continue_retries,
            _pending_verification_response=_pending_verification_response,
            _pending_verification_response_previewed=_pending_verification_response_previewed,
            result=result,
        )

    # Reasoning-only clean stop: some reasoning parsers (vLLM nemotron_v3 past ~500K
    # prompt tokens) file the whole answer as reasoning when the model omits the closing
    # delimiter. ``finish_reason == "stop"`` means the provider considers generation
    # complete, so the empty-response ladder would only re-bill the same input to arrive
    # at a truncated preview of this text; promote the reasoning to the visible answer
    # BEFORE the ladder. ``length`` (cut off mid-thought) stays on the continuation path.
    # The promoted text is RETURNED as the answer but never written into the assistant
    # row's ``content``: chain-of-thought stored as ordinary content is indistinguishable
    # from a real reply on every history surface (#111761). The row keeps ``content``
    # empty with the text in its reasoning fields and carries the promoted text as the
    # ``api_content`` sidecar, so the next turn still replays it byte-identically.
    _content = assistant_message.content
    _promoted = None
    if (
        finish_reason == "stop"
        and not assistant_message.tool_calls
        and (_content is None or (isinstance(_content, str) and not _content.strip()))
    ):
        _promoted = agent._extract_reasoning(assistant_message) or None
        if _promoted:
            # WARNING, not INFO: a model that keeps ending turns this way is stalled
            # (planning monologue, zero tool calls) while the turn reports "complete".
            logger.warning(
                "Reasoning-only clean stop (%d chars) — returning the reasoning as the final "
                "response (model=%s provider=%s api_calls=%d tool_turns=%d)",
                len(_promoted), agent.model, agent.provider, api_call_count,
                sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")),
            )
    final_response = _promoted or assistant_message.content or ""
    # Unmute: _mute_post_response from a housekeeping tool turn must not silence
    # empty-response warnings on the final response path.
    agent._mute_post_response = False

    # Think-block-only / empty content: recovery path.
    if not agent._has_content_after_think_block(final_response):
        _ev = recover_empty_response(
            agent, assistant_message, response, finish_reason, final_response=final_response,
            messages=messages, api_messages=api_messages, conversation_history=conversation_history,
            active_system_prompt=active_system_prompt, api_call_count=api_call_count,
            turn_exit_reason=_turn_exit_reason,
            preflight_compression_blocked=_preflight_compression_blocked,
        )
        final_response = _ev.final_response
        _turn_exit_reason = _ev.turn_exit_reason
        active_system_prompt = _ev.active_system_prompt
        _preflight_compression_blocked = _ev.preflight_compression_blocked
        if _ev.action == "return":
            return _verdict("return", _ev.result)
        if _ev.action == "break":
            return _verdict("break")
        return _verdict("continue")

    # Reject an internal artifact (compaction handoff or a tool call written as text)
    # before it can become durable or user-visible. Preserve strict alternation with an
    # ephemeral assistant/user retry pair; a kind-specific nudge tells the model what to fix.
    _artifact_kind = internal_artifact_kind(final_response, user_message=user_message)
    if _artifact_kind:
        retries = int(getattr(agent, "_internal_summary_retries", 0) or 0)
        logger.error(
            "Internal artifact (%s) rejected before persistence "
            "(retry=%d/%d model=%s provider=%s session=%s)",
            _artifact_kind,
            retries + 1,
            _INTERNAL_SUMMARY_MAX_RETRIES,
            getattr(agent, "model", "") or "unknown",
            getattr(agent, "provider", "") or "unknown",
            getattr(agent, "session_id", "") or "none",
        )
        if retries < _INTERNAL_SUMMARY_MAX_RETRIES:
            agent._internal_summary_retries = retries + 1
            artifact_msg = agent._build_assistant_message(assistant_message, "incomplete")
            if _promoted:
                artifact_msg["api_content"] = final_response
            artifact_msg["_internal_summary_recovery"] = True
            append_message(messages, artifact_msg)
            append_message(messages, {
                "role": "user",
                "content": _ARTIFACT_RECOVERY_NUDGES[_artifact_kind],
                "_internal_summary_recovery": True,
            })
            agent._session_messages = messages
            final_response = None
            return _verdict("continue")
        agent._internal_summary_retries = 0
        assistant_message.content = _INTERNAL_SUMMARY_SAFE_FAILURE
        final_response = _INTERNAL_SUMMARY_SAFE_FAILURE
    else:
        agent._internal_summary_retries = 0

    agent._empty_content_retries = 0
    agent._thinking_prefill_retries = 0
    # Surface the one-shot fallback switch notice before dropping the retry buffer so a
    # provider/model switch stays visible on success.
    agent._emit_pending_fallback_notice()
    agent._clear_status_buffer()

    # Defensive: repair malformed role-alternation before API call. Catches cases where the history got
    # wedged into a ``tool → user`` or ``user → user`` tail (e.g. after empty- response scaffolding was
    # stripped and a new user message landed after an orphan tool result). Most providers return empty
    # content on malformed sequences, which would otherwise retrigger the empty-retry loop indefinitely.
    # repair_message_sequence_with_cursor also recomputes the SessionDB flush cursor (_last_flushed_db_idx)
    # when repair compacts the list, so the turn-end flush doesn't skip the assistant/tool chain (#44837).
    # One-time repeated-heal escalation notice (#96870): if the sanitizer above just crossed the per-session
    # heal threshold, deliver the queued notice through the status/warning callback — the normal out-of-band
    # delivery channel (gateway status message / CLI print). NEVER appended to messages/api_messages:
    # conversation context and the cached prompt prefix stay byte-identical.
    from agent.agent_runtime_helpers import (
        intent_ack_continuation_mode, looks_like_degenerate_final, promoted_reasoning_announces_action,
        tool_results_this_turn, trailing_continue_intent,
    )

    _ack_mode = intent_ack_continuation_mode(agent)
    # Said-continue-but-stopped guard: no tool calls but the short reply TAILS with an
    # announced next action. Reuses the SAME bounded continuation counter (max 2 per turn).
    # Promoted reasoning gets the broader first-person-plan tail detector: with tools offered
    # and zero tool calls, chain-of-thought ending on "Let me batch the terminal calls..." is a
    # stalled model, and returning it as the answer aborts the tool loop while reporting
    # "complete" (#111761). Same cap, so a model that never acts still ends after 2 nudges.
    _stall_text = agent._strip_think_blocks(final_response or "")
    _stall_continue_intent = (
        bool(getattr(agent, "_stall_guards", True))
        and agent.valid_tool_names
        and codex_ack_continuations < 2
        and (
            trailing_continue_intent(_stall_text)
            or (bool(_promoted) and promoted_reasoning_announces_action(_stall_text))
        )
    )
    # Degenerate-final guard (#103483): the turn did real tool work and then stopped on a
    # fragment. Same scope knob and the SAME bounded counter as the ack continuation; the nudge
    # row itself closes the tool-work window, so a second fragment ends the turn as the answer.
    _tool_rows = tool_results_this_turn(messages)
    _degenerate_final = (
        bool(getattr(agent, "_stall_guards", True))
        and _ack_mode != "off"
        and codex_ack_continuations < 2
        and _tool_rows > 0
        and looks_like_degenerate_final(_stall_text, user_message=user_message)
    )
    # Precedence: an announced next action outranks the fragment shape; the codex ack is last.
    if _stall_continue_intent:
        _continuation_kind = "stall"
    elif _degenerate_final:
        _continuation_kind = "degenerate"
    elif (
        _ack_mode != "off"
        and agent.valid_tool_names
        and codex_ack_continuations < 2
        and agent._looks_like_codex_intermediate_ack(
            user_message=user_message, assistant_content=final_response, messages=messages,
            require_workspace=(_ack_mode == "codex_only"),
        )
    ):
        _continuation_kind = "ack"
    else:
        _continuation_kind = None
    if _continuation_kind:
        if _continuation_kind == "stall":
            logger.info(
                "Stall guard: turn ending on trailing continue-"
                "intent with no tool calls — re-prompting to act "
                "(%d/2)", codex_ack_continuations + 1,
            )
        elif _continuation_kind == "degenerate":
            logger.warning(
                "Degenerate final: %d-char fragment %r ended the turn after %d tool result(s) — "
                "re-prompting (%d/2)", len(_stall_text), _stall_text[:40], _tool_rows,
                codex_ack_continuations + 1,
            )
        codex_ack_continuations += 1
        interim_msg = agent._build_assistant_message(assistant_message, "incomplete")
        if _promoted:
            # Same sidecar as the final row: the wire copy must carry the promoted text, not only
            # ``reasoning_content``, or the continuation replays an empty assistant turn.
            interim_msg["api_content"] = final_response
        append_message(messages, interim_msg)
        agent._emit_interim_assistant_message(interim_msg)
        append_message(messages, {
            "role": "user",
            "content": (
                _DEGENERATE_FINAL_NUDGE if _continuation_kind == "degenerate"
                else _CODEX_ACK_CONTINUATION_NUDGE
            ),
        })
        agent._session_messages = messages
        # An acknowledgment is non-final: its text must not suppress iteration-limit
        # summarization if the continuation exhausts budget.
        final_response = None
        return _verdict("continue")

    codex_ack_continuations = 0

    if truncated_response_parts:
        final_response = _join_truncated_parts([*truncated_response_parts, final_response])
        truncated_response_parts = []
        length_continue_retries = 0
        # The continuation recovered, so the fragments stay in the transcript.
        for _frag in messages:
            if isinstance(_frag, dict):
                _frag.pop("_length_continuation_fragment", None)
                _frag.pop("_length_continuation_nudge", None)

    final_response = agent._strip_think_blocks(final_response).strip()

    final_msg = agent._build_assistant_message(assistant_message, finish_reason)
    if _promoted:
        # Replay sidecar only: ``content`` stays empty so the row is never mistaken for a
        # real reply; ``build_api_messages`` substitutes ``api_content`` on the wire.
        final_msg["api_content"] = final_response

    # Dropped tool-call recovery (copilot/Claude): finish_reason="tool_calls" with empty
    # tool_calls would end the turn unstarted; re-prompt (max 3 CONSECUTIVE stalls).
    if (
        finish_reason == "tool_calls"
        and not assistant_message.tool_calls
        and getattr(agent, "_dropped_toolcall_retries", 0) < 3
    ):
        agent._dropped_toolcall_retries = getattr(agent, "_dropped_toolcall_retries", 0) + 1
        logger.warning(
            "finish_reason=tool_calls with empty tool_calls array "
            "(narration only) — re-prompting to emit the call "
            "(retry %d/3, model=%s provider=%s)",
            agent._dropped_toolcall_retries, agent.model, agent.provider,
        )
        agent._emit_diagnostic_status(
            "↻ Model signaled a tool call but sent none — "
            f"re-prompting ({agent._dropped_toolcall_retries}/3)"
        )
        # Both halves of the re-prompt pair are ephemeral scaffolding: never persisted,
        # and the finalization pop strips an unanswered tail pair.
        final_msg["_dropped_toolcall_nudge"] = True
        append_message(messages, final_msg)
        append_message(messages, {
            "role": "user",
            "content": _DROPPED_TOOLCALL_NUDGE_CONTENT,
            "_dropped_toolcall_nudge": True,
        })
        agent._session_messages = messages
        final_response = None
        return _verdict("continue")

    # Genuine turn end (no dropped-tool-call mismatch): clear stall budget.
    agent._dropped_toolcall_retries = 0

    # Pop prefill / empty-retry scaffolding before the final response or
    # verification follow-up; it must not become durable transcript.
    while (
        messages
        and isinstance(messages[-1], dict)
        and any(messages[-1].get(flag) for flag in _EPHEMERAL_SCAFFOLDING_FLAGS)
    ):
        messages.pop()

    _sg = apply_stop_gates(
        agent, final_msg, final_response=final_response, messages=messages,
        conversation_history=conversation_history,
        pending_verification_response=_pending_verification_response,
        pending_verification_response_previewed=_pending_verification_response_previewed,
    )
    _pending_verification_response = _sg.pending_verification_response
    _pending_verification_response_previewed = _sg.pending_verification_response_previewed
    if _sg.continue_turn:
        final_response = None
        return _verdict("continue")

    # Plugins rewrite the reply BEFORE it is appended and flushed: SQLite treats a non-blank
    # assistant row as settled, so a transform after this write would reach the user but never
    # the stored/replayed transcript (#44239). finalize_turn reads the recorded outcome; like
    # there, an interrupted turn keeps the raw text.
    from agent.turn_finalizer import apply_llm_output_transform
    _transformed = False
    if not getattr(agent, "_interrupt_requested", False):
        final_response, _transformed, _ = apply_llm_output_transform(
            agent, final_response, turn_id=getattr(agent, "_current_turn_id", "") or "", logger=logger,
        )
    if _transformed:
        if _promoted:
            final_msg["api_content"] = final_response
        else:
            final_msg["content"] = final_response

    append_message(messages, final_msg)
    # Make the answer durable before leaving the loop (_DB_PERSISTED_MARKER keeps
    # _persist_session idempotent). Failure must NOT abort the turn: finalize retries.
    try:
        agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception:
        logger.warning(
            "final text-turn flush failed (session=%s) — reply is "
            "not yet durable; relying on finalize_turn retry",
            getattr(agent, "session_id", None) or "none",
            exc_info=True,
        )

    _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
    if not agent.quiet_mode:
        agent._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
    return _verdict("break")
