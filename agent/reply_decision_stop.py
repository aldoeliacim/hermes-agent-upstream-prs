"""Turn-end forced-decision guard for the tool-gated reply gate.

This module is intentionally policy-only. It never delivers a message itself;
it turns the passive "undecided" outcome (see ``gateway/run.py``'s reply-gate
telemetry) into a bounded follow-up when the model tries to finish a
tool-gated group/DM turn without ever calling ``send_message(target="current",
...)`` to reply or ``send_message(target="current", action="silent")`` to
explicitly decide silence.

Why this exists (CONFIRMED real casualty, 2026-07-11, in a free-response
group): the reply-gate redesign shipped 2026-07-09 correctly stopped
auto-delivering a substantive-looking free-text tail when the model never
decided — but "correctly suppressed" and "actually delivered a real answer"
are different outcomes, and the redesign had no mechanism to push the model
toward the latter. A user asked a real technical question; the agent did 11
tool calls of real investigation and wrote a complete, correct, 1260-char
answer ending in a direct question back to the user — then simply never
called ``send_message``. The turn ended, logged ``outcome=undecided``, and
the answer was silently discarded. The user explicitly said afterward: "I
actually asked for you to FORCE an answer[,] a decision." This module is that
force: mirrors ``agent/verification_stop.py``'s proven pattern (a synthetic
user-role nudge that holds the tool-calling loop open for one more round when
a real condition is unmet), applied to "did the model decide?" instead of
"did the model verify?".

Bounded and fails open: capped attempts (default 2, config
``agent.max_reply_decision_nudges``), and if the model still doesn't decide
after nudging, the ORIGINAL gateway suppression (log ``outcome=undecided``,
no delivery) is the final fallback — this module cannot regress to "always
deliver" behavior, it can only reduce how often "undecided" actually fires by
giving the model more chances to make the explicit call before the turn ends.
"""

from __future__ import annotations

import os
from typing import Any, Optional


DEFAULT_MAX_REPLY_DECISION_NUDGES = 2

# The nudge text. Deliberately explicit about BOTH valid actions (reply or
# explicit silence) so the model isn't pushed toward always replying just to
# satisfy the gate — the Reply Gate skill's silence criteria (bare reactions,
# emoji-echo loops, pure banter) remain in force. This only forces a DECISION,
# not a particular decision.
_NUDGE_TEXT = (
    "[System: This is a tool-gated turn — your response text above is NOT "
    "delivered on its own. You must now call send_message to actually decide "
    "what happens: "
    'send_message(target="current", message="...") to deliver a reply, or '
    'send_message(target="current", action="silent") to explicitly record '
    "that you're choosing not to reply (per the Reply Gate skill's silence "
    "criteria — bare reactions, emoji-echo loops, pure banter between other "
    "people). If you already wrote a substantive answer above, the most "
    "common mistake is writing it and then NOT calling send_message to "
    "deliver it — call send_message now with that same content rather than "
    "writing it again. Ending this turn without calling send_message means "
    "your response, even a good one, is silently discarded.]"
)


def max_reply_decision_nudges(config: Optional[dict[str, Any]] = None) -> int:
    """Bound on consecutive forced-decision nudges per tool-gated turn (>= 0)."""
    agent_cfg = _agent_cfg(config)
    raw = agent_cfg.get("max_reply_decision_nudges")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_REPLY_DECISION_NUDGES


def _agent_cfg(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
    agent_cfg = (config or {}).get("agent") if isinstance(config, dict) else None
    return agent_cfg if isinstance(agent_cfg, dict) else {}


def reply_decision_nudge_enabled(config: Optional[dict[str, Any]] = None) -> bool:
    """Return whether the forced-decision nudge is active.

    Precedence: an explicit ``HERMES_REPLY_DECISION_NUDGE`` env var wins, then
    an explicit ``agent.reply_decision_nudge`` config bool. Defaults True —
    unlike ``verify_on_stop``'s surface-aware default, this nudge is scoped
    entirely by the CALLER already checking ``tool_gated`` (see
    ``build_reply_decision_nudge``'s docstring); it is meaningless to enable
    outside a tool-gated turn, so there is no separate messaging-surface
    default to reason about here.
    """
    env = os.environ.get("HERMES_REPLY_DECISION_NUDGE")
    if env is not None:
        return env.strip().lower() not in {"0", "false", "no", "off"}
    cfg_val = _agent_cfg(config).get("reply_decision_nudge")
    if isinstance(cfg_val, bool):
        return cfg_val
    if isinstance(cfg_val, str):
        token = cfg_val.strip().lower()
        if token in {"0", "false", "no", "off"}:
            return False
    return True


def build_reply_decision_nudge(
    *,
    tool_sends: int,
    decided_silent: int,
    attempts: int,
    max_attempts: Optional[int] = None,
) -> Optional[str]:
    """Return a synthetic follow-up when a tool-gated turn hasn't decided yet.

    Returns None (no nudge) when: a decision was already made this turn
    (``tool_sends`` or ``decided_silent`` > 0 — nothing to force), or the
    attempt cap is reached (fail open to the existing ``undecided`` gateway
    suppression rather than looping forever).
    """
    if tool_sends > 0 or decided_silent > 0:
        return None
    cap = max_reply_decision_nudges() if max_attempts is None else max_attempts
    if attempts >= cap:
        return None
    return _NUDGE_TEXT


__all__ = [
    "DEFAULT_MAX_REPLY_DECISION_NUDGES",
    "build_reply_decision_nudge",
    "max_reply_decision_nudges",
    "reply_decision_nudge_enabled",
]
