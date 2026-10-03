"""Fail-closed guard against a config write that silently drops most of the file.

THE bug class this exists for: ``atomic_roundtrip_yaml_save`` merges *by deletion* — a key absent
from the caller's dict is removed from disk ("explicit absence"). ``save_config(config)`` from a
caller that loaded only a *fragment* of the config therefore does not "save what it knows", it
deletes everything else. Reproduced 2026-10-03: a partial ``save_config`` shrank a 6,120-line /
251 KB config to 61 keys / 11.6 KB and silently removed ``gateway.reply_gate_mode``,
``plugins.enabled`` (which gated the whole Mnemosyne memory toolset off) and two platforms.

Every config write already funnels through :func:`hermes_cli.config.atomic_config_write`, and
``scripts/check_config_yaml_writers.py`` proves they all use it — but "centralized" is not
"correct". The funnel itself was the shredder. This guard is the missing semantic check.

Two independent signals, because either one alone has a legitimate case that trips it:

1. **Proportion** — the result holds far fewer *top-level* keys than the on-disk file had.
2. **Mass** — the result is much smaller in bytes than the on-disk file was.

The one legitimate "I really do want to delete most of this" caller is a deliberate restore or
migration, which passes ``allow=...`` with a human-readable reason; that reason lands in the
warning so the deletion stays attributable in the logs (#113301 family).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# A write is refused only when BOTH signals fire, so a legitimate shrink that touches one
# dimension (e.g. trimming a long list, or dropping a few big default blocks while keeping every
# section) still goes through. These are deliberately far from the observed collapse
# (71 keys / 251 KB -> 61 keys / 11.6 KB = 86% and 95% loss); ordinary edits sit well above.
MIN_KEY_RATIO = 0.60   # keep >= 60% of the existing top-level keys
MIN_BYTE_RATIO = 0.40  # keep >= 40% of the existing bytes


def _top_level_keys(doc: Any) -> Sequence[str]:
    return list(doc.keys()) if isinstance(doc, dict) else []


def _lost_leaf_paths(existing: dict, incoming: dict, limit: int = 6) -> list[str]:
    """Dotted paths of *leaves* present before and absent now — e.g. ``gateway.reply_gate_mode``.

    Naming the section alone (``gateway``) is not actionable: the user's damage was one scalar
    inside a large section that legitimately kept other keys. The leaves are what they need to see.
    """
    lost: list[str] = []

    def walk(old: Any, new: Any, prefix: str) -> None:
        if len(lost) >= limit * 4:
            return
        if isinstance(old, dict) and isinstance(new, dict):
            for k, v in old.items():
                walk(v, new.get(k, None), f"{prefix}.{k}" if prefix else str(k))
            return
        if isinstance(old, dict) and not isinstance(new, dict):
            # whole section replaced by a non-mapping: name its leaves, they are all going
            for k, v in old.items():
                walk(v, None, f"{prefix}.{k}" if prefix else str(k))
            return
        lost.append(prefix)

    for key, old_val in existing.items():
        if key not in incoming:
            walk(old_val, None, str(key))
        else:
            walk(old_val, incoming[key], str(key))
    return lost[:limit]


def _read_yaml_mapping(path: Path) -> Optional[dict]:
    """Parse *path* via the canonical owner; None when missing/unparseable (guard skips).

    Delegates to ``hermes_cli.config.read_user_config_raw`` (the sanctioned raw primitive for
    write-back round-trips) rather than opening the YAML itself, so this module needs no entry in
    ``test_config_read_guard``'s allowlist — it inherits the owner's profile-aware pathing instead
    of re-implementing a raw read.
    """
    from hermes_cli.config import read_user_config_raw
    try:
        data = read_user_config_raw(path)
    except Exception:  # noqa: BLE001 - a guard must never be the thing that breaks a write
        return None
    return data if isinstance(data, dict) else None


def guard_against_config_collapse(
    config_path: Path,
    existing: Optional[dict],
    incoming: dict,
    *,
    allow: Optional[str] = None,
) -> Optional[str]:
    """Return a refusal message when *incoming* would gut *existing*; None when the write is fine.

    *allow* is a caller-supplied justification for a deliberately destructive write. It is logged
    at WARNING (never silent) so an intentional restore is auditable after the fact.
    """
    if existing is None or not isinstance(incoming, dict):
        return None
    # A caller explicitly deleting keys is doing so on purpose; only the *unintentional* whole-file
    # shrink is refused, and a write that ADDS keys can never shrink the file.
    old_keys, new_keys = set(_top_level_keys(existing)), set(_top_level_keys(incoming))
    dropped = old_keys - new_keys
    if not dropped:
        return None
    added = new_keys - old_keys

    try:
        old_bytes = config_path.stat().st_size
    except OSError:
        old_bytes = 0

    from utils import atomic_roundtrip_yaml_save  # local import: helper lives beside the writer
    import tempfile

    # Render the candidate exactly as the writer would, so we compare like with like.
    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / config_path.name
        try:
            atomic_roundtrip_yaml_save(probe, incoming)
        except Exception as exc:  # noqa: BLE001 - the real write will raise; don't pre-empt it here
            logger.debug("collapse guard could not render candidate for %s: %s", config_path, exc)
            return None
        try:
            new_bytes = probe.stat().st_size
        except OSError:
            new_bytes = 0

    key_ratio = len(new_keys) / len(old_keys) if old_keys else 1.0
    byte_ratio = new_bytes / old_bytes if old_bytes else 1.0
    if key_ratio >= MIN_KEY_RATIO or byte_ratio >= MIN_BYTE_RATIO:
        return None  # one healthy dimension — not a collapse

    # Both dimensions collapsed. That is the bug class, not an edit.
    leaves = _lost_leaf_paths(existing, incoming)
    msg = (
        f"Refusing to write {config_path}: this change would keep only {len(new_keys)}/{len(old_keys)} "
        f"top-level sections and {new_bytes}/{old_bytes} bytes, dropping {len(dropped)} "
        f"({', '.join(sorted(dropped)[:8])}{'...' if len(dropped) > 8 else ''}). "
        f"Settings that would be lost: {', '.join(leaves)}{'...' if len(leaves) >= 6 else ''}. "
        f"The config writer deletes keys the caller omitted, so a caller that loaded only part of "
        f"the config would erase the rest. Pass merge_existing=True (load the raw config and merge), "
        f"or call save_config with allow_collapse=<reason> if a full replacement is intended."
    )
    if allow:
        logger.warning(
            "Config write dropped %d sections (%.0f%% of keys, %.0f%% of bytes) but was allowed: %s",
            len(dropped), key_ratio * 100, byte_ratio * 100, allow,
        )
        return None
    logger.warning("Blocked config collapse on %s: %s", config_path, msg)
    return msg
