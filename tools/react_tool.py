"""``react`` — attach an emoji reaction to a message in the CURRENT conversation.

Why this tool exists (2026-09-11)
---------------------------------
Three reaction systems grew up in this codebase without knowing about each
other: the adapters' reflexive 👀/✅/❌ lifecycle hooks, the gateway's
slow-turn ack timer, and three model-facing side doors (``send_message
react``, ``whatsapp_action react_message``, desktop ``react_to_message``).
None of them shared a primitive — every caller did ``getattr(adapter,
"add_reaction")`` roulette across eight adapters with eight different method
names and argument orders, and silently no-op'd on the ones it missed.

The reflexes were removed as noise (a reaction that fires identically on
every message carries zero information; ``display.tool_progress: plain`` now
carries the "still alive" signal as *state*).  What was missing was the
legitimate act they were crudely approximating: reacting to a message the way
a person does — occasionally, when felt, as a conversational act.

Design
------
* **No ``target`` parameter, by construction.**  The chat is the live turn's
  bound source (``tools.approval.get_current_message_source``).  The hazard
  that got ``send_message react`` removed upstream (#47856) was a model
  *choosing* a cross-platform target from any turn; a tool with no target
  cannot be pointed anywhere.  Cross-chat reactions stay where they are
  (``send_message action=react``, refused inside live turns).
* **Addresses messages by row, never by platform id.**  The model never sees
  platform message ids; it sees the conversation.  ``messages_back`` counts
  the user's messages backwards from the triggering one (0 = the message
  being answered).  Resolution goes through the session transcript's
  ``platform_message_id`` column, which the gateway attaches to every
  persisted inbound user row.
* **One adapter contract.**  Everything below this tool is
  ``BasePlatformAdapter.react(chat_id, message_id, emoji)`` /
  ``unreact(...)``.  Adapters absorb their own quirks (Discord partial
  messages, Signal's ``author:timestamp`` composite id, Slack short-names,
  Feishu emoji types, Matrix reaction-event redaction) in their own override.
* **Service-gated (footprint ladder rung 2).**  ``check_fn`` exposes the tool
  only when the live turn's adapter declares ``supports_reactions``; on CLI,
  cron, and reaction-less platforms it costs zero schema tokens.  It is a
  ``_HERMES_CORE_TOOLS`` member so ``tool_search`` never demotes it into the
  catalog where a model can fail to find it.
* **Never a delivery decision.**  Reacting does NOT bump ``TurnDeliveryState``:
  a reaction is not a reply, so a tool-gated group turn that only reacts is
  still ``undecided`` and must still call ``send_message`` (or ``silent``).
  This keeps the reply gate's invariant — one explicit decision per inbound
  message — intact.  Conversely the model is told a reaction can BE the
  answer socially; the gate then records deliberate silence, which is the
  truthful outcome.
* **Owner-gated for autonomy, not for existence.**  In the owner's own chats
  the tool is unrestricted.  In any other chat it is only exposed when the
  turn is already under the tool-gated free-response policy — i.e. reacting
  is never *more* permissive than speaking.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from tools.approval import get_current_message_source, is_owner_source, is_reply_tool_gated_turn
from tools.registry import no_cache_check_fn, registry, tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolution helpers (pure; unit-tested without a gateway)
# ---------------------------------------------------------------------------

def _live_adapter(source: Any) -> Any:
    """The running gateway's adapter for ``source``'s platform, or None."""
    if source is None:
        return None
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except Exception:
        return None
    if runner is None:
        return None
    try:
        return runner._adapter_for_source(source)
    except Exception:
        return None


def _adapter_supports_reactions(adapter: Any) -> bool:
    return bool(adapter is not None and getattr(adapter, "supports_reactions", False))


def resolve_target_message_id(
    source: Any,
    messages_back: int,
    *,
    session_db: Any = None,
    session_id: str = "",
) -> Optional[str]:
    """Map ``messages_back`` to a platform message id in the current chat.

    ``0`` is the triggering message and needs no lookup — the gateway binds
    its id on the source.  Anything earlier walks the persisted transcript:
    the user rows of this session, newest first, taking the ones that carry
    a ``platform_message_id`` (rows without one — internal notifications,
    cron injections — are skipped rather than counted, so "two back" means
    two real messages the person sent).
    """
    if messages_back <= 0:
        return str(getattr(source, "message_id", "") or "") or None
    if session_db is None or not session_id:
        return None
    try:
        rows = session_db.get_recent_user_platform_message_ids(session_id, limit=messages_back + 1)
    except Exception as exc:
        logger.debug("react: transcript lookup failed: %s", exc)
        return None
    # rows are newest-first and exclude None ids; index 0 is the triggering
    # message (already persisted by the time a tool runs) when it had an id.
    trigger = str(getattr(source, "message_id", "") or "")
    if rows and trigger and rows[0] != trigger:
        # The trigger wasn't persisted with an id (or not yet) — treat the
        # list as strictly "earlier than the trigger".
        rows = [trigger] + rows
    return rows[messages_back] if len(rows) > messages_back else None


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------

async def react_tool(emoji: str = "", messages_back: int = 0) -> str:
    emoji = (emoji or "").strip()
    try:
        messages_back = max(0, int(messages_back or 0))
    except (TypeError, ValueError):
        return tool_error("messages_back must be a small non-negative integer.")

    source = get_current_message_source()
    if source is None:
        return tool_error("react only works inside a live conversation turn.")

    adapter = _live_adapter(source)
    if not _adapter_supports_reactions(adapter):
        return tool_error("This conversation's platform does not support reactions.")

    # Autonomy boundary: unrestricted in the owner's chats; elsewhere only
    # under the same policy that governs whether a reply is allowed at all.
    if not is_owner_source(source) and not is_reply_tool_gated_turn():
        return tool_error("Reactions are not enabled in this conversation.")

    chat_id = str(getattr(source, "chat_id", "") or "")
    if not chat_id:
        return tool_error("Could not determine the current chat.")

    session_db = None
    session_id = ""
    if messages_back > 0:
        # The session id lives in the turn's session env (the same place the
        # desktop react tool reads it) — SessionSource deliberately carries no
        # session id, it describes the CHAT, not the transcript.
        try:
            from gateway.session_context import get_session_env
            from hermes_state import SessionDB
            session_id = str(get_session_env("HERMES_SESSION_ID", "") or "")
            session_db = SessionDB() if session_id else None
        except Exception:
            session_db = None
    try:
        message_id = resolve_target_message_id(
            source, messages_back, session_db=session_db, session_id=session_id
        )
    finally:
        if session_db is not None:
            try:
                session_db.close()
            except Exception:
                pass
    if not message_id:
        which = "that message" if messages_back else "the current message"
        return tool_error(f"Could not find {which} to react to.")

    try:
        if emoji:
            ok = await adapter.react(chat_id, message_id, emoji)
        else:
            ok = await adapter.unreact(chat_id, message_id)
    except Exception as exc:
        logger.debug("react: adapter call failed: %s", exc)
        ok = False
    if not ok:
        return tool_error("The reaction could not be applied.")
    return json.dumps({"success": True, "emoji": emoji or None, "messages_back": messages_back})


def _react_handler(args: dict, **_kw) -> Any:
    return react_tool(
        emoji=args.get("emoji", ""),
        messages_back=args.get("messages_back", 0) or 0,
    )


# ---------------------------------------------------------------------------
# Availability gate — evaluated per schema build, cached briefly by registry
# ---------------------------------------------------------------------------

@no_cache_check_fn
def check_react_available() -> bool:
    """Expose ``react`` only where it can actually fire.

    True iff a live gateway turn is bound AND its adapter declares
    ``supports_reactions`` AND the turn is either the owner's or already
    tool-gated.  Everything else (CLI, cron, reaction-less platforms,
    prompt-mode non-owner chats) sees no schema entry at all.

    Uncached on purpose: the registry's default check_fn cache is a 30s
    process-wide TTL that is NOT turn-scoped, so a verdict computed for the
    owner's DM would leak into a stranger's group turn inside the window.
    The handler re-checks identity at call time regardless — schema
    presence is a footprint optimisation, never the authorization boundary.
    """
    source = get_current_message_source()
    if source is None:
        return False
    if not _adapter_supports_reactions(_live_adapter(source)):
        return False
    return is_owner_source(source) or is_reply_tool_gated_turn()


REACT_SCHEMA = {
    "name": "react",
    "description": (
        "React to a message in this conversation with a single emoji, the way a "
        "person taps a reaction. Use it when a reaction is what a person would do: "
        "something funny gets 😂, warmth gets ❤️, a plan you're on board with gets 👍, "
        "then carry on with whatever the message actually needs. Occasionally and "
        "when felt — not on every message, and NEVER as a status or 'working on it' "
        "signal. NEVER narrate a reaction ('reacting with…'); the emoji on the "
        "bubble is the whole point. Defaults to the message you're answering. One "
        "reaction per message: a different emoji replaces yours; empty string "
        "retracts it. A reaction is not a reply — if the message needs an answer, "
        "still answer it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "emoji": {
                "type": "string",
                "description": "The emoji (e.g. '❤️', '😂', '👍'). Empty string removes your reaction.",
            },
            "messages_back": {
                "type": "integer",
                "description": (
                    "Optional. 0 (default) = the message you're answering; 1 = the person's "
                    "message before that; 2 = two before. For when something lands late."
                ),
                "minimum": 0,
            },
        },
        "required": ["emoji"],
    },
}


registry.register(
    name="react",
    toolset="reactions",
    schema=REACT_SCHEMA,
    handler=_react_handler,
    check_fn=check_react_available,
    is_async=True,
    emoji="💬",
)
