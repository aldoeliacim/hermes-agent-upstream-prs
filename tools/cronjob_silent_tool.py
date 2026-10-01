#!/usr/bin/env python3
"""Explicit cron-silence tool.

Gives a cron-spawned agent a STRUCTURAL way to decide "nothing to report" --
a tool call the scheduler checks by attribute, never a free-text response the
scheduler has to pattern-match after the fact.

Why this exists (2026-08-12): before this tool, cron's silence contract was
purely LEXICAL -- the scheduler injected an instruction telling the model to
make its entire response equal one of a small marker-string set ([SILENT] /
SILENT / NO_REPLY / NO REPLY), and a regex-based matcher
(``gateway.response_filters.is_autonomous_silence_response``) tried to
recognize that string after the fact. A job author is free to author a
DIFFERENT silence instruction in their own prompt (e.g. "responde
únicamente la palabra SILENCIO"), and the model dutifully complies -- but
that word was never registered anywhere as a silence marker, so the exact
same regex-based matcher that correctly suppresses "[SILENT]" ships
"SILENCIO" straight to the recipient's chat every tick. Confirmed live:
cron_4fa958ec7c45 (a delivery reminder for a contact) leaked "SILENCIO" to
a real WhatsApp DM on three consecutive 20-minute ticks before this was
caught and fixed.

No regex can close that class of bug -- string-matching a free-text
response is fundamentally a guess at what the model meant, and a job author
can always author a synonym the guess doesn't cover. The only enforceable
fix is a decision the SCHEDULER controls the vocabulary of: a tool the
agent must actually CALL to register silence, whose availability and
semantics are defined by hermes-agent's own code, not by whatever string a
per-job prompt happens to author.

Delivery semantics: ``cron/scheduler.py`` reads
``tools.approval.get_cron_silence_state()`` after ``run_conversation()``
returns. If ``state.decided`` is True, delivery is suppressed -- full stop,
regardless of what text (if any) the response also contains. This REPLACES
(does not supplement) the old string-marker check for jobs that call this
tool; a job whose prompt was never updated to call it still falls back to
the legacy marker-string matcher (see ``cron/scheduler.py::
_is_cron_silence_response`` and ``gateway.response_filters.
is_autonomous_silence_response``) for backward compatibility with existing
job prompts, but the fix for THIS class of bug is authoring new/edited job
prompts to call ``cronjob_silent`` instead of relying on a magic word.

Availability is gated by toolset membership alone, deliberately with NO
``check_fn``. ``cron/scheduler.py::_resolve_cron_enabled_toolsets`` always
injects "cronjob_silent" into every cron job's resolved toolset list, so
the tool schema only ever reaches a cron-spawned agent. This is a
DELIBERATE deviation from the check_fn pattern several sibling tools use
(``clarify``, ``whatsapp_action_admin``): check_fn results are cached
PROCESS-WIDE with a 30s TTL (``tools/registry.py::_check_fn_cached``),
keyed only by ``(fn, profile_scope)`` -- not by which turn/thread is
asking. A single non-cron caller evaluating ``get_tool_definitions()``
anywhere in the process (a dashboard health probe, a live chat turn,
another concurrent cron tick that raced the cache) caches
``is_cron_agent_turn() == False`` for up to 30 real seconds, and every
OTHER concurrent or subsequent call -- including a genuine cron turn --
reads that same stale False and never sees the tool at all. Confirmed live
2026-08-12: the tool was correctly registered, correctly present in
``_HERMES_CORE_TOOLS`` (so ``tool_search`` never deferred it), and
``is_cron_agent_turn()`` reliably True at the moment ``run_conversation()``
executed -- yet a real cron agent still searched an empty catalog and
reported "No such tool exists". Root-caused only by reproducing the exact
process-wide TTL cache collision in isolation, not by re-checking any
single piece of the wiring alone (do not attempt to re-add
``check_fn=is_cron_agent_turn`` here; it was tried and removed for exactly
this reason). The handler's own ``is_cron_agent_turn()`` +
``CronSilenceState`` checks below are the real, per-call, never-cached
authorization boundary; toolset membership is the only thing that needs to
control schema visibility, and membership resolution has no caching layer
at all.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from tools.registry import registry, tool_error

CRONJOB_SILENT_SCHEMA = {
    "name": "cronjob_silent",
    "description": (
        "Call this when running as a scheduled cron job and there is "
        "genuinely nothing new to report this tick. This is the ONLY "
        "reliable way to suppress delivery -- do NOT try to signal "
        "silence by returning a specific word or phrase as your final "
        "response text (e.g. \"SILENT\", \"SILENCIO\", \"no change\"); "
        "that is NOT recognized and WILL be delivered verbatim to the "
        "recipient. Call this tool instead, then end your turn with no "
        "further message content. Only valid inside a cron job run -- "
        "calling it outside one is a no-op error."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": (
                    "Optional short internal note on why nothing was "
                    "delivered (for logs/audit only -- never shown to the "
                    "recipient)."
                ),
            },
        },
        "required": [],
    },
}


def cronjob_silent_tool(args: Dict[str, Any], **_kw) -> str:
    from cron.agent_turn_context import is_cron_agent_turn
    from tools.approval import note_cron_silence_decided

    if not is_cron_agent_turn():
        return tool_error(
            "cronjob_silent is only valid inside a running cron job -- no "
            "cron turn is active in this context."
        )

    reason = str(args.get("reason") or "").strip()
    ok = note_cron_silence_decided(reason)
    if not ok:
        # is_cron_agent_turn() said True but no CronSilenceState was bound --
        # a wiring bug in cron/scheduler.py, not a caller mistake. Surface it
        # loudly rather than pretending the decision took effect.
        return tool_error(
            "cronjob_silent could not record the decision: no "
            "CronSilenceState is bound for this run (internal wiring gap, "
            "please report)."
        )
    return json.dumps({"success": True, "decision": "silent"})


registry.register(
    name="cronjob_silent",
    toolset="cronjob_silent",
    schema=CRONJOB_SILENT_SCHEMA,
    handler=cronjob_silent_tool,
    emoji="\U0001f507",
)
