"""Approval context: who is asking, from where, under which policy.

Session identity and observability contextvars, the interactive/gateway/cron/
unattended predicates, and the ``approvals.*`` config readers used by every
gate in :mod:`tools.approval`.
"""

import contextvars
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional
from hermes_cli.config import cfg_get
from utils import env_var_enabled, is_truthy_value

logger = logging.getLogger("tools.approval")


def _ctx(name: str, default: "str | None" = "") -> contextvars.ContextVar:
    return contextvars.ContextVar(name, default=default)


# Per-thread/per-task gateway session identity: gateway runs agent turns concurrently in executor threads, so a
# process-global env var is racy (the env fallback stays for legacy single-threaded callers).
_approval_session_key: contextvars.ContextVar[str] = _ctx("approval_session_key")
_approval_turn_id: contextvars.ContextVar[str] = _ctx("approval_turn_id")
_approval_tool_call_id: contextvars.ContextVar[str] = _ctx("approval_tool_call_id")
# Hermes session id (observability identity, distinct from the gateway routing session_key), forwarded to approval
# hooks so observer plugins attach marks to the REAL session scope — otherwise they fall back to a synthetic "default"
# session whose scope never closes, so close-time exporters never ship them.
_approval_session_id: contextvars.ContextVar[str] = _ctx("approval_session_id")
# Interactive-CLI flag. Concurrent ACP sessions share a ThreadPoolExecutor, so mutating
# os.environ["HERMES_INTERACTIVE"] races: one session's `finally` restore can clobber another's set mid-run, dropping
# it onto the non-interactive auto-approve path so a dangerous command runs without the approval callback firing
# (GHSA-96vc-wcxf-jjff). None = unset → env fallback.
_hermes_interactive_ctx: contextvars.ContextVar[str | None] = _ctx("hermes_interactive", None)


def set_hermes_interactive_context(interactive: bool) -> contextvars.Token:
    """Bind interactive mode for the current context instead of mutating os.environ."""
    return _hermes_interactive_ctx.set("1" if interactive else "")


def reset_hermes_interactive_context(token: contextvars.Token) -> None:
    """Restore the prior value from :func:`set_hermes_interactive_context`."""
    _hermes_interactive_ctx.reset(token)


def _is_interactive_cli() -> bool:
    """True for an interactive CLI/ACP session (contextvar first, env fallback)."""
    ctx_val = _hermes_interactive_ctx.get()
    return is_truthy_value(ctx_val) if ctx_val is not None else env_var_enabled("HERMES_INTERACTIVE")


def _fire_approval_hook(hook_name: str, **kwargs) -> None:
    """Invoke a plugin lifecycle hook (pre_approval_request / post_approval_response).

    Lazy-imports the plugin manager (approval.py is imported long before plugins
    are discovered). Never raises: approval flow is safety-critical, plugin
    observability is not.
    """
    try:
        from hermes_cli.lifecycle import invoke_hook
    except Exception:
        return  # plugin system unavailable (bare tool-only imports, minimal tests)
    try:
        kwargs.setdefault("turn_id", _approval_turn_id.get())
        kwargs.setdefault("tool_call_id", _approval_tool_call_id.get())
        if _approval_session_id.get():
            kwargs.setdefault("session_id", _approval_session_id.get())
        invoke_hook(hook_name, **kwargs)
    except Exception as exc:
        # invoke_hook() swallows per-callback errors; this is the dispatch layer itself failing.
        logger.debug("Approval hook %s dispatch failed: %s", hook_name, exc)


def set_current_session_key(session_key: str) -> contextvars.Token[str]:
    """Bind the active approval session key to the current context."""
    return _approval_session_key.set(session_key or "")


def reset_current_session_key(token: contextvars.Token[str]) -> None:
    """Restore the prior approval session key context."""
    _approval_session_key.reset(token)


_Tokens = tuple[contextvars.Token[str], contextvars.Token[str], contextvars.Token[str]]


def set_current_observability_context(*, turn_id: str = "", tool_call_id: str = "", session_id: str = "") -> _Tokens:
    """Bind active tool correlation IDs to approval hooks."""
    return (_approval_turn_id.set(turn_id or ""), _approval_tool_call_id.set(tool_call_id or ""),
            _approval_session_id.set(session_id or ""))


def reset_current_observability_context(tokens: _Tokens) -> None:
    """Restore prior approval hook correlation IDs."""
    turn_token, tool_token, session_token = tokens
    _approval_session_id.reset(session_token)
    _approval_tool_call_id.reset(tool_token)
    _approval_turn_id.reset(turn_token)


def get_current_session_key(default: str = "default") -> str:
    """Return the active session key: approval contextvar → session_context → os.environ."""
    if session_key := _approval_session_key.get():
        return session_key
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_KEY", default)


def _session_env(name: str) -> str:
    """Session-scoped env value, contextvar-first so one cron/-q job cannot taint
    unrelated gateway/API/TUI turns in the same process; process env is the
    fallback for CLI tests and older entrypoints."""
    try:
        from gateway.session_context import get_session_env
        return get_session_env(name, "") or ""
    except Exception:
        return os.getenv(name, "") or ""


def _get_session_platform() -> str:
    """Return the current gateway platform from contextvars/env fallback."""
    return _session_env("HERMES_SESSION_PLATFORM")


def _is_cron_approval_context() -> bool:
    """True when the current approval decision is running inside cron."""
    return is_truthy_value(_session_env("HERMES_CRON_SESSION"))


# Programmatic/unattended platforms: no human can answer a prompt and the adapter has no ``send_exec_approval`` /
# ``/approve`` surface. Governed by ``approvals.unattended_mode`` (default deny), mirroring ``cron_mode`` — never an
# interactive round-trip that blocks for the full timeout with nobody to answer.
_UNATTENDED_APPROVAL_PLATFORMS = frozenset({"webhook", "msgraph_webhook", "api_server"})


def _is_unattended_platform_approval_context() -> bool:
    """True when the session platform is a programmatic/unattended surface.

    Webhook, msgraph_webhook, and api_server sessions bind ``HERMES_SESSION_PLATFORM`` like chat gateways
    do, but there is no human who can resolve a pending approval. Treating them as gateway approval contexts
    blocks the session for the full approval timeout (60-300s) and then fails closed anyway — the deadlock
    in #37284/#87509.
    """
    return _get_session_platform() in _UNATTENDED_APPROVAL_PLATFORMS


def _is_single_query_approval_context() -> bool:
    """True for a single-query (-q) session: ``hermes chat -q`` exports
    ``HERMES_INTERACTIVE=1`` (so sudo password prompts work) but nobody is waiting
    to answer approvals; without this marker the gate would wait the full timeout,
    fail closed and push the agent toward workarounds (e.g. execute_code).
    ``approvals.single_query_mode`` makes the path deterministic."""
    return is_truthy_value(_session_env("HERMES_SINGLE_QUERY_SESSION"))


def _is_gateway_approval_context() -> bool:
    """True inside a gateway/API session that can answer an approval.

    Legacy integrations set HERMES_GATEWAY_SESSION; concurrent paths bind
    HERMES_SESSION_PLATFORM via contextvars. Cron is NEVER a gateway approval
    context even when it originated from a platform (cron binds the platform for
    delivery routing): falling through would submit a pending approval with no
    listener and block the job indefinitely; unattended platforms likewise.

    Unattended programmatic platforms (webhook, msgraph_webhook, api_server) are excluded for the same
    reason: those adapters have no ``send_exec_approval`` and no way to receive ``/approve`` replies.
    Submitting a pending approval there blocks the session for the full approval timeout (60-300 s) with no
    human who can resolve it (#37284, 87509). Their dangerous-command handling is governed by
    ``approvals.unattended_mode`` config (default deny), mirroring cron.
    """
    if _is_cron_approval_context() or _is_unattended_platform_approval_context():
        return False
    return env_var_enabled("HERMES_GATEWAY_SESSION") or bool(_get_session_platform())


def _resolve_cli_approval_callback(approval_callback=None):
    """Explicit callback, else the per-thread one from ``terminal_tool.set_approval_callback``."""
    if approval_callback is not None:
        return approval_callback
    try:
        from tools.terminal_tool import _get_approval_callback
        return _get_approval_callback()
    except Exception:
        return None


def _should_fall_through_to_cli_approval(*, is_cli: bool, approval_callback, notify_cb) -> bool:
    """Prefer the CLI Dangerous Command panel over a silent pending approval:
    ``HERMES_EXEC_ASK`` (or a platform marker) can leak into an interactive CLI
    process (historically via ``import gateway.run``), and without a gateway notify
    listener the ask branch used to return ``pending_approval`` immediately and
    skip the panel the user can actually answer."""
    return bool(is_cli and approval_callback is not None and notify_cb is None)


_VALID_MODES = ("manual", "smart", "off")


def _normalize_approval_mode(mode) -> str:
    """Normalize approval mode values loaded from YAML/config. YAML 1.1 parses a
    bare ``off`` as False, so ``mode: off`` arrives as a bool; treat it as the
    intended string mode. Unknown strings (e.g. 'auto') warn and fall back to
    'manual' instead of silently failing every mode check."""
    if isinstance(mode, bool):
        return "off" if mode is False else "manual"
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if normalized in _VALID_MODES:
            return normalized
        if normalized:
            logger.warning("Unknown approvals.mode %r — defaulting to 'manual'. "
                           "Valid values: %s", mode, ", ".join(_VALID_MODES))
    return "manual"


def _get_approval_config() -> dict:
    """Read the approvals config block: the LIVE config-cache sub-dict
    (load_config_readonly contract) — callers must not mutate it or any nested structure."""
    try:
        from hermes_cli.config import load_config_readonly
        return load_config_readonly().get("approvals", {}) or {}
    except Exception as e:
        logger.warning("Failed to load approval config: %s", e)
        return {}


def _get_approval_mode() -> str:
    """Return 'manual', 'smart', or 'off' (a hosted-room policy overrides config)."""
    try:
        from gateway.hosted_room_execution_policy import current_room_execution_policy
        if (room_policy := current_room_execution_policy()) is not None:
            return room_policy.approval_mode
    except Exception:
        pass
    return _normalize_approval_mode(_get_approval_config().get("mode", "manual"))


def _get_approval_timeout() -> int:
    """Read ``approvals.timeout`` (default 300s: gateway push notifications may
    not be seen for minutes; 60s failed closed before Telegram taps landed).
    Clamped to ``agent.deadline.MAX_SAFE_TIMEOUT_S`` (~1 year): a larger value
    overflows ``time_t`` inside ``Thread.join`` / ``Lock.acquire`` on macOS and
    crashed every parallel tool batch; clamping at the single config-read site
    keeps every consumer platform-safe at once."""
    try:
        raw = int(_get_approval_config().get("timeout", 300))
    except (ValueError, TypeError):
        return 300
    try:
        from agent.deadline import MAX_SAFE_TIMEOUT_S
        safe_cap = int(MAX_SAFE_TIMEOUT_S)
    except Exception:
        safe_cap = 365 * 24 * 3600  # fail CLOSED: the raw value would re-open the overflow
    if raw > safe_cap:
        logger.warning("approvals.timeout=%s exceeds the platform-safe maximum; clamping to %ss", raw, safe_cap)
    return min(raw, safe_cap)


def format_approval_window(seconds: int) -> str:
    """The ONE human wording for an approval timeout window, shared by the CLI timeout notice,
    the tool result's ``user_summary`` and the gateway card copy so every surface agrees:
    300 → "5 minutes", 90 → "90 seconds", 7200 → "2 hours"."""
    seconds = max(int(seconds or 0), 0)
    if seconds and seconds % 3600 == 0:
        count, unit = seconds // 3600, "hour"
    elif seconds and seconds % 60 == 0:
        count, unit = seconds // 60, "minute"
    else:
        count, unit = seconds, "second"
    return f"{count} {unit}" if count == 1 else f"{count} {unit}s"


def approval_timeout_notice_kwargs() -> dict:
    """``{waited, suggested}`` for the ``approval.timeout`` copy: how long we waited (``5 minutes`` /
    ``90 seconds``) and a tripled ``approvals.timeout`` value the user can paste into ``hermes config set``."""
    seconds = _get_approval_timeout()
    return {"waited": format_approval_window(seconds), "suggested": seconds * 3}


def _binary_approval_mode(key: str) -> str:
    """Read ``approvals.<key>`` as 'approve' or 'deny' (default deny)."""
    try:
        from hermes_cli.config import load_config_readonly
        mode = str(cfg_get(load_config_readonly(), "approvals", key, default="deny")).lower().strip()
        return "approve" if mode in {"approve", "off", "allow", "yes"} else "deny"
    except Exception:
        return "deny"


def _get_cron_approval_mode() -> str:
    """Read the cron approval mode from config. Returns 'deny' or 'approve'."""
    return _binary_approval_mode("cron_mode")


def _get_single_query_approval_mode() -> str:
    """Read the single-query (-q) approval mode from config. Returns 'deny' or 'approve'."""
    return _binary_approval_mode("single_query_mode")


def _get_unattended_approval_mode() -> str:
    """Approval mode for webhook / msgraph_webhook / api_server sessions; default
    deny — an unattended session never silently runs a flagged action unless the
    operator explicitly trusts it."""
    return _binary_approval_mode("unattended_mode")


def _tirith_fail_open() -> bool:
    """``security.tirith_fail_open`` (default True; True when config is unreadable).
    False means the operator opted into fail-closed: an un-importable scanner
    must not silently grant access."""
    try:
        from hermes_cli.config import load_config_readonly
        _sec = (load_config_readonly() or {}).get("security", {}) or {}
        return bool(_sec.get("tirith_fail_open", True)) if _sec.get("tirith_enabled", True) else True
    except Exception:
        return True


def _get_approval_transport_config() -> tuple[str, str | None]:
    """Return explicitly selected transport and fail-closed fallback mode."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = ((load_config_readonly() or {}).get("security") or {}).get("approval") or {}
        selected = str(cfg.get("transport") or "builtin").strip().lower()
        fallback = str(cfg.get("transport_fallback") or "").strip().lower()
    except Exception:
        # An unreadable/malformed selection must not silently materialize a
        # prompt on a built-in surface the operator may not be watching.
        return "config-error", None
    return selected or "builtin", "builtin" if fallback == "builtin" else None


# ──────────────────────────────────────────────────────────────────────────
# Reply-gate delivery state: message source, per-turn delivery ledger, tool-gate
# ──────────────────────────────────────────────────────────────────────────

_current_message_source: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "current_message_source",
    default=None,
)


def set_current_message_source(source: Any) -> contextvars.Token:
    """Bind the current gateway turn's inbound message source to this context."""
    return _current_message_source.set(source)


def get_current_message_source() -> Optional[Any]:
    """Return the current gateway turn's inbound message source, or None.

    None outside a live gateway turn (CLI, cron, subagent) — callers that need
    a message source (e.g. send_message(target="current")) must surface a clear
    error rather than guess.
    """
    return _current_message_source.get()


def reset_current_message_source(token: contextvars.Token) -> None:
    """Restore the prior message-source context."""
    _current_message_source.reset(token)


@dataclass
class TurnDeliveryState:
    """Per-inbound-message reply-gate decision counters.

    tool_sends: successful send_message(target="current") deliveries.
    decided_silent: explicit send_message(target="current", action="silent").
    Both accumulate across the entire re-entrant follow-up chain for one
    inbound message, because the object is created once at the outer per-turn
    scope and shared by identity down every queued follow-up.
    """

    tool_sends: int = 0
    decided_silent: int = 0

    @property
    def delivery_decided(self) -> bool:
        """True once this turn has explicitly decided its delivery outcome.

        The canonical reply-gate predicate. A turn is decided the moment it
        either delivered to the current chat (a successful current-target
        send_message) or recorded explicit silence (action="silent"). It is a
        property on the identity-stable turn object rather than a process-global
        flag: the in-process cron ticker and concurrent live conversations share
        one process, so any global would leak one turn's decision into another.
        The post-turn delivery block and the stream consumer both read this same
        predicate, so streamed text and post-turn free text cannot disagree.
        """
        return self.tool_sends > 0 or self.decided_silent > 0


_current_turn_delivery_state: contextvars.ContextVar[Optional["TurnDeliveryState"]] = (
    contextvars.ContextVar("current_turn_delivery_state", default=None)
)

_current_turn_reply_tool_gated: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "current_turn_reply_tool_gated",
    default=False,
)




def is_owner_source(source: Any) -> bool:
    """True when ``source`` (a live turn's inbound message source) resolves,
    for its own platform, to this deployment's configured home channel --
    i.e. the message came from the owner's own identity on that platform.
    False for any other sender and for an unresolvable/unconfigured source.

    Platform-agnostic generalization of
    ``tools.whatsapp_action_tool._is_owner_whatsapp_source`` -- reuse that
    WhatsApp-specific helper directly when you already know the source is
    WhatsApp (it also runs LID/phone-JID alias expansion this function
    doesn't need to duplicate for other platforms). This function is for
    disclosure/action gates that must work across ANY platform the gateway
    is deployed on (e.g. send_message(action="list") exposing every
    connected chat across every platform, not just WhatsApp).

    Mirrors ``gateway/run.py::_should_skip_user_profile_for_source``'s core
    identity comparison (the guard that keeps USER.md out of non-owner
    chats) without needing a live ``GatewayRunner`` instance -- this only
    needs the module-level home-channel config, so it can run from a plain
    tool handler with no runner reference. Deliberately a call-time check
    (read fresh via ``get_current_message_source()``), not a ``check_fn``
    registration gate: ``check_fn`` results are cached up to 30s and are
    NOT turn-scoped (see ``tools/registry.py::_check_fn_cached``), so
    gating identity there would let one turn's verdict leak into a
    different sender's turn within the same cache window.
    """
    if source is None:
        return False
    platform = getattr(source, "platform", None)
    if platform is None:
        return False

    try:
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        home = config.get_home_channel(platform) if config else None
    except Exception:
        home = None
    if not home or not home.chat_id:
        return False

    home_ids = {str(home.chat_id)}
    source_values = {
        str(getattr(source, "chat_id", "") or ""),
        str(getattr(source, "user_id", "") or ""),
        str(getattr(source, "chat_id_alt", "") or ""),
        str(getattr(source, "user_id_alt", "") or ""),
    }
    source_values.discard("")

    try:
        from gateway.config import Platform
        if platform == Platform.WHATSAPP:
            from gateway.whatsapp_identity import expand_whatsapp_aliases
            for value in list(home_ids):
                home_ids |= expand_whatsapp_aliases(value)
            for value in list(source_values):
                source_values |= expand_whatsapp_aliases(value)
    except Exception:
        pass

    return bool(home_ids & source_values)


def set_reply_tool_gated_turn(active: bool) -> contextvars.Token:
    """Mark whether the current turn is under tool-mode reply gating.

    Bound once per inbound message from the turn source's resolved reply
    decision; read by the send-message tool and post-turn block to know whether
    the free-text tail is subject to the delivery ledger for this turn.
    """
    return _current_turn_reply_tool_gated.set(bool(active))


def is_reply_tool_gated_turn() -> bool:
    """True when the current turn is under tool-mode reply gating."""
    return bool(_current_turn_reply_tool_gated.get())


def reset_reply_tool_gated_turn(token: contextvars.Token) -> None:
    """Restore the prior reply-tool-gated context."""
    _current_turn_reply_tool_gated.reset(token)


def set_turn_delivery_state(state: "TurnDeliveryState") -> contextvars.Token:
    """Bind the per-turn delivery ledger shared across the follow-up chain."""
    return _current_turn_delivery_state.set(state)


def get_turn_delivery_state() -> Optional["TurnDeliveryState"]:
    """Return the current turn's delivery ledger, or None outside a turn."""
    return _current_turn_delivery_state.get()


def reset_turn_delivery_state(token: contextvars.Token) -> None:
    """Restore the prior turn delivery-state context."""
    _current_turn_delivery_state.reset(token)


def note_current_message_delivered() -> None:
    """Record that the current turn delivered to its originating chat via a tool.

    Increments the turn-scoped TurnDeliveryState (shared by identity across the
    whole re-entrant follow-up chain) so the post-turn block and stream consumer
    can dedup: a tool-driven send to the current chat means the free-text tail
    must never also auto-deliver. Also bumps the legacy source-object counter
    when a source is bound, for readers still consulting reply_gate_tool_sends.
    No-op outside a gateway turn.
    """
    state = _current_turn_delivery_state.get()
    if state is not None:
        try:
            state.tool_sends += 1
        except Exception:
            pass
    src = _current_message_source.get()
    if src is not None:
        try:
            src.reply_gate_tool_sends = int(getattr(src, "reply_gate_tool_sends", 0) or 0) + 1
        except Exception:
            pass


def note_current_message_silent() -> None:
    """Record that the current turn explicitly decided to stay silent.

    Companion to :func:`note_current_message_delivered`: the two make "reply"
    and "silent" symmetric first-class decisions rather than "silent" being the
    mere absence of a send. An explicit silent decision closes the gate but is
    NOT a delivery. Mirrors to the legacy source counter for back-compat.
    No-op outside a gateway turn.
    """
    state = _current_turn_delivery_state.get()
    if state is not None:
        try:
            state.decided_silent += 1
        except Exception:
            pass
    src = _current_message_source.get()
    if src is not None:
        try:
            src.reply_gate_decided_silent = int(getattr(src, "reply_gate_decided_silent", 0) or 0) + 1
        except Exception:
            pass


@dataclass
class CronSilenceState:
    """Whether the currently-running cron agent explicitly decided silence.

    Set by ``tools.cronjob_silent_tool.cronjob_silent_tool`` via
    ``note_cron_silence_decided()``. Read by ``cron/scheduler.py`` after
    ``agent.run_conversation()`` returns, in place of (never in addition
    to) regex-matching the response text.
    """

    decided: bool = False
    reason: str = ""


_current_cron_silence_state: contextvars.ContextVar[Optional["CronSilenceState"]] = (
    contextvars.ContextVar("current_cron_silence_state", default=None)
)


def set_cron_silence_state(state: "CronSilenceState") -> contextvars.Token:
    """Bind a fresh CronSilenceState for the cron tick about to run.

    Call once per ``run_conversation()`` invocation, before the agent
    starts, mirroring ``set_turn_delivery_state``'s lifecycle.
    """
    return _current_cron_silence_state.set(state)


def get_cron_silence_state() -> Optional["CronSilenceState"]:
    """Return the current cron tick's CronSilenceState, or None outside one."""
    return _current_cron_silence_state.get()


def reset_cron_silence_state(token: contextvars.Token) -> None:
    """Restore the prior cron-silence-state context."""
    _current_cron_silence_state.reset(token)


def note_cron_silence_decided(reason: str = "") -> bool:
    """Record that the running cron agent explicitly chose to stay silent.

    Returns False (and records nothing) when called outside a bound
    CronSilenceState -- i.e. not actually inside a cron tick. Callers
    (the ``cronjob_silent`` tool) surface that as a clear tool error rather
    than silently no-op'ing, so a misuse is visible immediately instead of
    behaving like a successful decision that never took effect.
    """
    state = _current_cron_silence_state.get()
    if state is None:
        return False
    state.decided = True
    state.reason = reason
    return True
