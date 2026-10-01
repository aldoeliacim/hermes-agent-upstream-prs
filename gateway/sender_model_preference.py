"""Sender identity for persistent gateway model defaults (not an authorization key)."""

from gateway.whatsapp_identity import expand_whatsapp_aliases, normalize_whatsapp_identifier


def sender_keys(source) -> tuple[str, ...]:
    """Use the author, not chat, scope or profile; no inference for missing authors."""
    if source is None or not source.user_id or not source.platform:
        return ()
    platform = getattr(source.platform, "value", source.platform)
    if platform == "whatsapp":
        return tuple(sorted(expand_whatsapp_aliases(source.user_id)))
    return (str(source.user_id),)


def owner_sender_key(platform: str, sender_id: str) -> str:
    """Normalize CLI input but retain the exact WhatsApp alias supplied by the owner.

    Lookup expands aliases at turn time so a mapping installed later still finds the old row.
    """
    if not platform or not sender_id or not sender_id.strip():
        raise ValueError("platform and sender ID required")
    if platform == "whatsapp":
        key = normalize_whatsapp_identifier(sender_id)
        if not key.isascii() or not key.isdigit():
            raise ValueError("WhatsApp sender must be a phone or LID numeric identifier")
        return key
    return sender_id.strip()


def get_sender_model_preference(db, source):
    keys = sender_keys(source)
    return db.get_sender_model_preference(getattr(source.platform, "value", source.platform), keys) if keys else None
