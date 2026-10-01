"""Create-time cross-identity delivery guard for cron jobs.

Bug class (2026-09-11, job 6f43f6241fbe): an agent serving a NON-owner
contact's request created a cron job whose ``deliver`` was the OWNER's home
channel. The job's ``origin`` correctly recorded the requester (auto-captured
from the session), but the model hand-passed a literal owner target, so a
contact's errand reported back to the owner — an owner-report leak in cron
form, latent until the job fired.

This guard closes the class at creation, where both facts are in scope:

* ``origin``  — who asked (the creating session's platform/chat_id).
* ``deliver`` — where the model wants output to go.

The invariant is deliberately NARROW. It is NOT "origin must equal deliver":
a contact may legitimately ask for delivery elsewhere (a family member asking for a
reminder in the family group is correct). The leak is specifically:

    requester is NOT the owner   AND   a deliver element resolves to the owner.

Owner identity is whatever the scheduler already treats as "home" for a
platform (``_get_home_target_chat_id``: env mirror → config.yaml), expanded
through the same WhatsApp phone/LID alias primitive the runtime owner guards
use. No parallel owner registry.

The verdict is only trustworthy when the requester's identity is known
(``origin`` populated). CLI/TUI creates have no origin and are owner-operated
by construction; those pass. A cron-context create is not a person's request
and also passes (its ``origin`` is re-resolved by ``_resolve_cron_context_deliver``).

The blocked-verdict message is written for the model, which is who sends the
offending create: it names the fix (deliver to the requester's origin, or an
explicit non-owner target).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

# Deliver elements that never name a chat and cannot address the owner
# alone. ``origin`` routes to the requester; ``local`` stays on disk.
# ``all`` and bare platform names DO resolve to home channels and are
# handled explicitly below.
_NON_CHAT_ELEMENTS = frozenset({"origin", "local"})


def _resolve_target_chat_id(platform: str, rest: str) -> Optional[str]:
    """Resolve an explicit ``platform:rest`` chat token to its concrete chat id.

    A ``deliver`` element such as ``telegram:Bot Home`` or ``slack:#bot-home`` is
    NOT a literal owner id, so a literal string comparison against the home id
    misses it — yet the scheduler resolves that friendly name/handle to a real
    chat id at fire time via ``resolve_send_target`` and can land it on the
    owner's home channel. To close that bypass we must resolve the token the
    same way the scheduler will, then compare the RESOLVED id against home.

    ``rest`` is the whole substring after the first ``:`` (the scheduler splits
    on the first ``:`` only), so any thread suffix is preserved and handed to
    the resolver exactly as fire-time does. ``pass_unresolved_references=True``
    mirrors ``cron.scheduler_delivery``'s call so the guard sees the same id the
    delivery path will use.

    Read-only by construction: ``resolve_send_target`` consults the in-process
    platform registry and the on-disk channel directory only — no network, no
    external writes. The resolution runs entirely inside a ``try`` and this
    helper NEVER raises: on any resolver exception, resolution error, or a
    malformed / non-string result it returns the sentinel ``""`` (empty string),
    which the caller treats as "could not prove NOT-owner" and fails closed.
    Returns ``None`` only when the resolver cleanly declined to change the token
    (nothing extra to compare beyond the literal already checked).
    """
    try:
        from tools.send_message_tool import resolve_send_target
    except Exception:
        logger.debug("delivery_guard: resolve_send_target import failed", exc_info=True)
        return ""  # fail closed: cannot resolve → cannot prove safe
    try:
        chat_id, _thread_id, error = resolve_send_target(
            platform, rest, pass_unresolved_references=True
        )
    except Exception:
        logger.debug(
            "delivery_guard: resolve_send_target raised for %r:%r", platform, rest,
            exc_info=True,
        )
        return ""  # fail closed on resolver exception
    if error:
        # Resolver could not turn the token into a concrete id. It will not be
        # delivered to the owner's home by name, but a guard that cannot see the
        # resolved id must not conclude the target is safe.
        return ""
    if chat_id is None:
        return None  # cleanly unresolved: nothing beyond the literal to compare
    if not isinstance(chat_id, str) or not chat_id.strip():
        # Malformed resolver output. Fail closed.
        return ""
    return chat_id


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _identity_set(platform: str, chat_id: str) -> Set[str]:
    """All identifiers ``chat_id`` is known to alias on ``platform``.

    WhatsApp expands phone-JID <-> LID through the bridge mapping files; every
    other platform is a single opaque id. Always contains the normalised input.
    """
    base = _norm(chat_id)
    if not base:
        return set()
    ids: Set[str] = {base}
    if platform == "whatsapp":
        try:
            from gateway.whatsapp_identity import expand_whatsapp_aliases

            for alias in expand_whatsapp_aliases(chat_id):
                ids.add(_norm(alias))
        except Exception:
            logger.debug("delivery_guard: whatsapp alias expansion failed", exc_info=True)
    # Strip a WhatsApp/JID suffix so ``5215...@s.whatsapp.net`` and the bare
    # number compare equal; harmless on platforms without ``@``.
    ids |= {i.split("@", 1)[0] for i in list(ids) if "@" in i}
    ids.discard("")
    return ids


def _home_chat_id(platform: str) -> str:
    try:
        from cron.scheduler_delivery import _get_home_target_chat_id

        return _get_home_target_chat_id(platform) or ""
    except Exception:
        logger.debug("delivery_guard: home lookup failed for %r", platform, exc_info=True)
        return ""


def _known_platforms() -> Iterable[str]:
    try:
        from cron.scheduler_delivery import _HOME_TARGET_ENV_VARS

        return list(_HOME_TARGET_ENV_VARS.keys())
    except Exception:
        return ["telegram", "whatsapp", "discord", "slack"]


def _owner_identities() -> Dict[str, Set[str]]:
    """platform -> alias set of that platform's configured home channel."""
    out: Dict[str, Set[str]] = {}
    for platform in _known_platforms():
        home = _home_chat_id(platform)
        if home:
            out[platform] = _identity_set(platform, home)
    return out


def requester_is_owner(origin: Optional[Dict[str, Any]]) -> Optional[bool]:
    """True/False when decidable; None when the requester is unknown.

    ``None`` (no origin, or an origin with no platform/chat_id) means the
    create did not come from a person's chat — CLI/TUI or cron context — and
    the guard must not judge it.
    """
    if not origin:
        return None
    platform = _norm(origin.get("platform"))
    chat_id = str(origin.get("chat_id") or "")
    if not platform or not chat_id:
        return None
    home = _home_chat_id(platform)
    if not home:
        # No home configured on the requester's platform: nothing to compare
        # against, so the requester cannot be proven owner. Treat as
        # non-owner — this is the fail-closed side of the guard.
        return False
    return bool(_identity_set(platform, home) & _identity_set(platform, chat_id))


def owner_targets_in_deliver(deliver: Optional[str]) -> List[str]:
    """Return the deliver elements that resolve to an owner home channel.

    Recognised forms (mirrors the scheduler's parser):
      * ``all``                      -> every configured home channel
      * ``<platform>``               -> that platform's home channel
      * ``<platform>:<chat>[:<thr>]`` -> owner iff <chat> aliases the home id
      * ``bot-chat[:profile]``, ``origin``, ``local`` -> never owner

    ``all`` and a bare platform name address the home channel BY CONSTRUCTION
    — the scheduler resolves them to it at fire time — so they count as owner
    targets even when no home id is resolvable here. Only the explicit
    ``platform:chat`` form needs the id to compare against; without one it is
    treated as owner-addressed (fail-closed): a guard that cannot see the
    home must not conclude the target is safe.
    """
    if not deliver:
        return []
    owners = _owner_identities()
    hits: List[str] = []
    for raw in str(deliver).split(","):
        elem = raw.strip()
        low = elem.lower()
        if not elem or low in _NON_CHAT_ELEMENTS or low.startswith("bot-chat"):
            continue
        if low == "all":
            hits.append(elem)
            continue
        platform, _, rest = low.partition(":")
        if not rest:
            # bare platform name == that platform's home channel
            hits.append(elem)
            continue
        home_ids = owners.get(platform)
        if home_ids is None:
            # No resolvable home on this platform: cannot prove the explicit
            # chat is NOT the owner. Fail closed.
            hits.append(elem)
            continue
        # 1. Literal comparison: the chat token as written (lowercased) may
        #    already be an owner home id or one of its aliases.
        chat = rest.split(":", 1)[0]
        if _identity_set(platform, chat) & home_ids:
            hits.append(elem)
            continue
        # 2. Resolver comparison: a friendly name / handle / raw id (e.g.
        #    ``telegram:Bot Home``, ``slack:#bot-home``) is not literally the
        #    owner id, but the scheduler resolves it to a concrete chat id at
        #    fire time and may land on the owner's home. Resolve it the same
        #    way and compare the resolved id. Use the ORIGINAL-case substring
        #    after the first ``:`` (channel names / raw ids can be
        #    case-sensitive) and preserve any thread suffix, exactly as
        #    fire-time delivery does. ``""`` from the resolver means it could
        #    not prove a concrete id (exception / error / malformed): fail
        #    closed by treating the element as owner-addressed.
        original_rest = elem.split(":", 1)[1] if ":" in elem else ""
        resolved = _resolve_target_chat_id(platform, original_rest)
        if resolved is None:
            # Resolver cleanly declined to change the token: nothing beyond the
            # literal (already checked) to compare — leave it allowed.
            continue
        if not resolved:
            # Fail closed: could not resolve to a concrete id.
            hits.append(elem)
            continue
        resolved_chat = resolved.split(":", 1)[0]
        if _identity_set(platform, resolved_chat) & home_ids:
            hits.append(elem)
    return hits


def check_cross_identity_delivery(
    origin: Optional[Dict[str, Any]],
    deliver: Optional[str],
) -> Optional[str]:
    """Return an error string when a non-owner request would deliver to the owner.

    Returns ``None`` (allowed) when:
      * the requester is unknown (CLI/TUI/cron-context create), or
      * the requester IS the owner, or
      * no deliver element resolves to an owner home channel.
    """
    is_owner = requester_is_owner(origin)
    if is_owner is None or is_owner:
        return None
    hits = owner_targets_in_deliver(deliver)
    if not hits:
        return None
    who = (origin or {}).get("chat_name") or (origin or {}).get("chat_id") or "the requester"
    return (
        f"Cross-identity delivery blocked: this job was requested by a "
        f"non-owner ({who}) but deliver={deliver!r} routes to the owner's "
        f"home channel ({', '.join(hits)}). A contact's job must report to "
        f"the contact, not to the owner. Omit deliver (defaults to the "
        f"requester's origin), pass deliver='origin', or give an explicit "
        f"non-owner target such as 'whatsapp:<their chat id>'."
    )
