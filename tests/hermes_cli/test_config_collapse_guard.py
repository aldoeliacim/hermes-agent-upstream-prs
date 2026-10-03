"""The config writer must refuse to silently delete most of the user's config.

THE bug class (2026-10-03): ``atomic_roundtrip_yaml_save`` merges by deletion — a key absent from
the caller's dict is removed from disk. ``save_config(config)`` from a caller holding only a
*fragment* of the config therefore erased everything it didn't know about: 71 keys / 251 KB became
61 keys / 11.6 KB, taking ``gateway.reply_gate_mode`` (the WhatsApp silence gate), ``plugins.enabled``
(which gated the whole Mnemosyne memory toolset off) and two platforms with it.

The whole class is gated here through the REAL entry point, not by asserting on a helper: every
production path reaches ``atomic_config_write``, so these tests drive that. Each destructive case
must ALSO assert the file on disk is untouched — a guard that raises but has already written is
not a guard.
"""

import copy
from pathlib import Path

import pytest

from hermes_cli.config import atomic_config_write, read_raw_config, save_config
from hermes_cli.config_collapse_guard import MIN_BYTE_RATIO, MIN_KEY_RATIO

# A config with the same shape as the casualty: many top-level sections, one of which
# (``gateway.reply_gate_mode``) is a single scalar whose loss is invisible until a user complains.
FULL_CONFIG = {
    "model": {"default": "some/model", "provider": "someprovider"},
    "gateway": {
        "reply_gate_mode": "tool",
        "reply_gate_tool_fallback": True,
        "platforms": {"whatsapp": {"enabled": True, "extra": {"allow_from": ["a@g.us", "b@g.us"]}}},
    },
    "plugins": {"enabled": ["mnemosyne", "mnemosyne-dashboard", "disk-cleanup", "ponytail"]},
    "memory": {"mnemosyne": {"tools": [f"tool_{i}" for i in range(38)]}},
    "platforms": {"telegram": {"enabled": True}, "whatsapp": {"enabled": True}},
    "security": {"redact_secrets": True},
    "terminal": {"cwd": "/home/user"},
    "compression": {"enabled": True},
    "display": {"message_reactions": False},
    "logging": {"level": "info"},
    "cron": {"enabled": True},
    "skills": {"enabled": True},
    "delegation": {"enabled": True},
    "kanban": {"enabled": True},
    "browser": {"enabled": True},
}


def _seed(tmp_path: Path, data: dict) -> Path:
    """Write *data* as the user's config.yaml through the real writer (as a create, so the guard
    is inactive: an empty prior file has nothing to protect)."""
    cfg = tmp_path / "config.yaml"
    atomic_config_write(cfg, copy.deepcopy(data))
    return cfg


def test_seed_is_intact(tmp_path):
    """Guard the fixture itself: if the seed doesn't produce a full config the rest proves nothing."""
    import yaml
    cfg = _seed(tmp_path, FULL_CONFIG)
    on_disk = yaml.safe_load(cfg.read_text())
    assert on_disk["gateway"]["reply_gate_mode"] == "tool"
    assert len(on_disk) == len(FULL_CONFIG)


# ---------------------------------------------------------------- the destructive case

def test_partial_write_is_refused(tmp_path):
    """The exact reproduction: a caller holding 2 of 14 sections tries to save what it knows."""
    cfg = _seed(tmp_path, FULL_CONFIG)
    before = cfg.read_text()

    partial = {"model": {"default": "new/model"}}
    with pytest.raises(RuntimeError, match="Refusing to write"):
        atomic_config_write(cfg, partial)

    assert cfg.read_text() == before, "guard raised but the file was already modified"


def test_partial_write_names_what_it_would_drop(tmp_path):
    """The refusal must be actionable: name the sections, not just say 'no'."""
    cfg = _seed(tmp_path, FULL_CONFIG)
    with pytest.raises(RuntimeError) as exc:
        atomic_config_write(cfg, {"model": {"default": "new/model"}})
    msg = str(exc.value)
    assert "reply_gate_mode" in msg, "must name the high-value key that would be lost"
    assert "merge_existing=True" in msg, "must tell the caller the safe pattern"


# ---------------------------------------------------------------- legitimate writes pass

def test_normal_edit_is_allowed(tmp_path):
    """A one-field edit inside an existing section must never be refused — that is the common case."""
    cfg = _seed(tmp_path, FULL_CONFIG)
    edited = copy.deepcopy(FULL_CONFIG)
    edited["model"]["default"] = "another/model"
    atomic_config_write(cfg, edited)
    assert "another/model" in cfg.read_text()


def test_adding_a_section_is_allowed(tmp_path):
    cfg = _seed(tmp_path, FULL_CONFIG)
    grown = copy.deepcopy(FULL_CONFIG)
    grown["a_new_section"] = {"enabled": True}
    atomic_config_write(cfg, grown)
    assert "a_new_section" in cfg.read_text()


def test_single_deliberate_key_removal_is_allowed(tmp_path):
    """Users DO delete keys; the guard targets mass loss, not intent."""
    cfg = _seed(tmp_path, FULL_CONFIG)
    trimmed = copy.deepcopy(FULL_CONFIG)
    del trimmed["browser"]
    atomic_config_write(cfg, trimmed)
    assert "browser" not in cfg.read_text()


def test_creating_a_fresh_config_is_allowed(tmp_path):
    """No prior file -> nothing to protect. `hermes setup` on a new install must work."""
    cfg = tmp_path / "brand-new.yaml"
    atomic_config_write(cfg, copy.deepcopy(FULL_CONFIG))
    assert cfg.is_file() and "reply_gate_mode" in cfg.read_text()


# ---------------------------------------------------------------- the escape hatch

def test_allow_collapse_permits_deletion_with_a_reason(tmp_path, caplog):
    """A deliberate full replacement must be possible, and must be attributable in the logs."""
    cfg = _seed(tmp_path, FULL_CONFIG)
    with caplog.at_level("WARNING"):
        atomic_config_write(cfg, {"model": {"default": "only-this"}}, allow_collapse="deliberate restore")
    assert "only-this" in cfg.read_text()
    assert "deliberate restore" in caplog.text, "an allowed collapse must still be logged"


def test_allow_collapse_never_silences_itself(tmp_path, caplog):
    """Passing the flag must not turn the log off — the whole point is auditability."""
    cfg = _seed(tmp_path, FULL_CONFIG)
    with caplog.at_level("WARNING"):
        atomic_config_write(cfg, {"model": {}}, allow_collapse="restore from backup")
    assert "restore from backup" in caplog.text


# ---------------------------------------------------------------- the real entry point

def test_save_config_partial_is_refused(tmp_path, monkeypatch):
    """``save_config`` is what the 146 unsafe call sites use — it must refuse, not just warn."""
    from hermes_cli import config as cfg_mod
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cfg_mod, "get_config_path", lambda: tmp_path / "config.yaml")
    _seed(tmp_path, FULL_CONFIG)
    before = (tmp_path / "config.yaml").read_text()

    with pytest.raises(RuntimeError, match="Refusing to write"):
        save_config({"model": {"default": "tiny"}})

    assert (tmp_path / "config.yaml").read_text() == before


# ---------------------------------------------------------------- pre-write backup

def test_write_snapshots_the_previous_state(tmp_path):
    """Fix #3: the pre-write copy must hold the PREVIOUS (intact) config, not the casualty.

    ``config.yaml.good.*`` is snapshotted at load time and therefore only ever captures a file
    that already parsed — after a destructive write the whole trail is 1.9 KB of the corpse.
    """
    cfg = _seed(tmp_path, FULL_CONFIG)
    atomic_config_write(cfg, {**copy.deepcopy(FULL_CONFIG), "display": {"message_reactions": True}})

    backups = list((tmp_path / "backups" / "config").glob("config.yaml.pre-write.*"))
    assert backups, "no pre-write backup was taken"
    assert "reply_gate_mode" in backups[0].read_text(), "backup must predate the write"


def test_refused_write_leaves_no_debris(tmp_path):
    """A refused write must not litter the backups dir either."""
    cfg = _seed(tmp_path, FULL_CONFIG)
    with pytest.raises(RuntimeError):
        atomic_config_write(cfg, {"model": {}})
    pre = list((tmp_path / "backups" / "config").glob("config.yaml.pre-write.*"))
    # One from the seed's own write is fine; a refused write must not add a second.
    assert len(pre) <= 1


# ---------------------------------------------------------------- thresholds are sane

def test_thresholds_leave_headroom():
    """The observed collapse was 86% key loss / 95% byte loss. A guard that trips on ordinary
    edits would get disabled, so the thresholds must stay well clear of real editing."""
    assert MIN_KEY_RATIO <= 0.75
    assert MIN_BYTE_RATIO <= 0.5
