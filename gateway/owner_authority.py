"""Generic owner-authority predicate for gateway security gates.

This module answers exactly one question, fail-closed:

    "Is this message source the gateway's proven primary owner?"

It exists so approval / session-trust gates (``/approve``, dangerous-command
prompts) can refuse to grant authority to a routed contact whose DM the agent
happens to be working in. It is intentionally NARROW and shares NO code with the
semantic USER.md profile-suppression layer:

* It derives owner identity from configured homes, but treats a shared home
  chat_id as a delivery address rather than proof of who spoke. An explicit
  home user_id can prove a group participant's identity.
* It NEVER walks WhatsApp LID-alias bridge files or applies any other
  identity-resolution semantics — that logic belongs to the identity layer. For
  opaque identifiers with no proving signal the predicate FAILS CLOSED (treats
  the source as a non-owner), which is the safe direction for a security gate.
* Any failure to obtain proof (missing config, exploding accessor, absent
  source) resolves to non-owner.

* Any exact or digit-normalized match between a source participant identifier
  and a private home chat_id or explicit home user_id proves ownership. A
  shared-room chat_id never proves ownership.
* A session-key tail can only match one of those actor-safe home identifiers.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Set

from gateway.config import Platform


def _digits(value: Any) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _home_ids_for(config: Any, platform: Platform, platform_name: str,
                  user_config: Optional[Dict[str, Any]]) -> Set[str]:
    """Collect every configured home identifier for this platform. Fail-closed:
    an accessor that raises contributes nothing rather than propagating."""
    home_ids: Set[str] = set()

    try:
        get_home = getattr(config, "get_home_channel", None)
        home = get_home(platform) if callable(get_home) else None
        if home is not None and getattr(home, "chat_id", None):
            home_ids.add(str(home.chat_id))
            if getattr(home, "user_id", None):
                home_ids.add(str(home.user_id))
    except Exception:
        pass

    try:
        cfg = user_config or {}
        pdata = (cfg.get("platforms") or {}).get(platform_name) or {}
        hdata = pdata.get("home_channel") or {}
        if isinstance(hdata, dict):
            if hdata.get("chat_id"):
                home_ids.add(str(hdata["chat_id"]))
            if hdata.get("user_id"):
                home_ids.add(str(hdata["user_id"]))
    except Exception:
        pass

    return home_ids


def source_is_owner(
    config: Any,
    source: Any,
    *,
    session_key: Optional[str] = None,
    user_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True only when ``source`` is PROVEN to be the primary owner.

    Fail-closed: any uncertainty returns False.
    """
    if source is None:
        return False

    platform = getattr(source, "platform", None)
    if platform == Platform.LOCAL:
        return True
    if platform is None:
        return False

    platform_name = platform.value if hasattr(platform, "value") else str(platform)
    home_ids = _home_ids_for(config, platform, platform_name, user_config)
    if not home_ids:
        return False

    # A shared chat is a delivery destination, not an actor identity. In a group,
    # the chat id is identical for every participant; never let it prove owner
    # authority (including through the session-key tail). An explicit home
    # user_id or a private home chat can still match the participant below.
    chat_id = str(getattr(source, "chat_id", "") or "")
    chat_type = str(getattr(source, "chat_type", "") or "").lower()
    shared = chat_type in {"group", "supergroup", "channel", "thread", "forum"} or (
        platform == Platform.WHATSAPP and chat_id.endswith(("@g.us", "@broadcast", "@newsletter"))
    )
    if shared:
        # Also exclude alternate encodings of the same room: exact subtraction
        # alone would leave a digit-normalized match to a home room alias.
        room_ids = {chat_id, str(getattr(source, "chat_id_alt", "") or "")}
        room_digits = {_digits(v) for v in room_ids if v}
        home_ids = {h for h in home_ids if h not in room_ids and _digits(h) not in room_digits}
        if not home_ids:
            return False

    source_values = {
        str(getattr(source, "user_id", "") or ""),
        str(getattr(source, "user_id_alt", "") or ""),
    }
    if not shared:
        source_values.update({chat_id, str(getattr(source, "chat_id_alt", "") or "")})
    source_values.discard("")

    # 1) Exact match.
    if home_ids & source_values:
        return True

    # 2) Digit-normalized match (phone homes: '+52…' vs bare '52…').
    home_digits = {d for d in (_digits(v) for v in home_ids) if d}
    source_digits = {d for d in (_digits(v) for v in source_values) if d}
    if home_digits & source_digits:
        return True

    # 3) Session-key tail digits (resolved participant even when the raw id is opaque).
    key_parts = [p for p in str(session_key or "").split(":") if p]
    key_tail_digits = _digits(key_parts[-1]) if key_parts else ""
    if not shared and home_digits and key_tail_digits and key_tail_digits in home_digits:
        return True

    return False


def source_is_non_owner(
    config: Any,
    source: Any,
    *,
    session_key: Optional[str] = None,
    user_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Fail-closed inverse of :func:`source_is_owner` for deny-side gates.

    ``True`` means "not a proven owner" — the value a security gate should use to
    withhold authority. Because ownership must be PROVEN, an unknown source is a
    non-owner.
    """
    return not source_is_owner(
        config, source, session_key=session_key, user_config=user_config
    )


def home_is_configured_for(
    config: Any,
    source: Any,
    *,
    user_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True when a home channel is configured for ``source``'s platform.

    A security *suppression* gate (e.g. hiding an approval prompt from a
    non-owner) must only fire when there is an actual owner to protect. In a
    single-user deployment with NO home configured there is no distinct owner,
    so callers combine this with :func:`source_is_non_owner` to suppress ONLY
    when a home exists AND the source does not match it. LOCAL and an absent
    source return False (nothing to protect / nothing to check)."""
    if source is None:
        return False
    platform = getattr(source, "platform", None)
    if platform is None or platform == Platform.LOCAL:
        return False
    platform_name = platform.value if hasattr(platform, "value") else str(platform)
    return bool(_home_ids_for(config, platform, platform_name, user_config))
