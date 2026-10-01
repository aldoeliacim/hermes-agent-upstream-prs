"""A cron provider snapshot must ROUND-TRIP, or it bills the wrong account.

Live incident 2026-09-20: job ce245fe471ab failed 23 consecutive runs with
"HTTP 400: antigravity/gemini-3.7-flash-high is not a valid model ID", every
15 minutes. The pin looked right (cron.provider: omniroute) and the model was
genuinely served by the local router, so the failure read like a bad model id.

It was a SPEND bug. ``resolve_runtime_provider`` returns provider="custom" as an
INTERNAL label for any custom-provider entry (omniroute among them), while
"custom" is ALSO a literal config key pointing at api.openrouter.ai. The
creation snapshot stored the label; the scheduler fed it straight back as
``requested="custom"``; that resolved to the PAID endpoint, which then rejected
the local router's model name. Zero of 2259 OmniRoute call logs contained the
400 — the request never reached the local router at all.

So the very mechanism meant to prevent silent spend changes was causing one.
Two defenses, because the write-side fix alone leaves poisoned jobs on disk:

  1. write side  — snapshot ``requested_provider`` (round-trippable), not the label.
  2. read side   — refuse to honor a legacy "custom" provider snapshot; it cannot
                   express intent, so fall back to the live global default.

v2026.9.21 note: this build snapshots the (provider, model) pair for a
``pinned`` job through ``cron.jobs._main_model_pin`` and reads it back for
dispatch through ``cron.scheduler._resolve_job_runtime``. The behaviors below
are exercised against those actual seams (the old tree's ``_compute_provider_
model_snapshots`` / ``_snapshot_pin`` do not exist here); the invariants are
identical.
"""

import logging

import pytest

from cron.jobs import _main_model_pin
from cron.scheduler import _sanitize_provider_pin


def test_snapshot_records_the_round_trippable_provider_name(monkeypatch, tmp_path):
    """omniroute must be stored as "omniroute", never as the "custom" label."""
    import cron.jobs as jobs
    import hermes_cli.runtime_provider as rp

    # A config with model.default set, so a pinned job actually snapshots a pair.
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("model:\n  default: some-model\n")
    monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path)

    monkeypatch.setattr(
        rp,
        "resolve_runtime_provider",
        lambda **kw: {
            "provider": "custom",               # internal billing label
            "requested_provider": "omniroute",  # what actually round-trips
            "base_url": "http://127.0.0.1:20128/v1",
            "source": "custom_provider:omniroute",
        },
    )
    provider, model = _main_model_pin()
    assert model == "some-model"
    assert provider == "omniroute", (
        "storing the resolved label 'custom' makes the snapshot resolve to the paid "
        "[custom] provider on the next scheduler tick"
    )


def test_snapshot_falls_back_to_label_when_no_requested_name(monkeypatch, tmp_path):
    """A real first-class provider (no custom indirection) still snapshots normally."""
    import cron.jobs as jobs
    import hermes_cli.runtime_provider as rp

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("model:\n  default: claude-x\n")
    monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path)

    monkeypatch.setattr(
        rp, "resolve_runtime_provider",
        lambda **kw: {"provider": "anthropic", "base_url": "https://api.anthropic.com"},
    )
    provider, model = _main_model_pin()
    assert model == "claude-x"
    assert provider == "anthropic"


def test_legacy_custom_snapshot_is_refused(caplog):
    """The poisoned value already on disk must NOT be honored.

    Returning "" makes the caller fall through to the live global default
    (omniroute), instead of resolving "custom" -> openrouter.ai.
    """
    with caplog.at_level(logging.WARNING):
        assert _sanitize_provider_pin("custom", "omniroute", "ce245fe471ab") == ""
    assert "paid" in caplog.text.lower()


def test_legitimate_provider_snapshot_is_still_honored():
    """Only the ambiguous "custom" value is refused — real pins keep working.

    A job created under anthropic must keep running on anthropic even after the
    global default moves; that is the entire point of the snapshot.
    """
    assert _sanitize_provider_pin("anthropic", "omniroute", "job1") == "anthropic"


@pytest.mark.parametrize("value", ["", None])
def test_missing_snapshot_still_follows_the_global_default(value):
    assert _sanitize_provider_pin(value, "omniroute", "j") == ""


def test_resolve_job_runtime_ignores_poisoned_custom_pin(monkeypatch):
    """End-to-end read side: a job carrying provider='custom' must NOT be fed back
    as requested='custom' (paid openrouter). It falls through to the live default."""
    import cron.scheduler as scheduler

    captured = {}

    def fake_resolve(**kwargs):
        captured["requested"] = kwargs.get("requested")
        return {"provider": kwargs.get("requested") or "omniroute", "base_url": "x"}

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
    )
    jc = scheduler._CronJobConfig(
        cfg={}, model="some-model", model_cfg={}, cron_default_provider=""
    )
    job = {"provider": "custom", "model": "some-model"}
    scheduler._resolve_job_runtime(job, "ce245fe471ab", jc)
    assert captured["requested"] is None, (
        "poisoned 'custom' provider pin must not be forwarded as requested='custom'"
    )


def test_resolve_job_runtime_preserves_real_provider_pin(monkeypatch):
    """A legitimate provider pin (anthropic) is still forwarded as requested."""
    import cron.scheduler as scheduler

    captured = {}

    def fake_resolve(**kwargs):
        captured["requested"] = kwargs.get("requested")
        return {"provider": kwargs.get("requested"), "base_url": "x"}

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
    )
    jc = scheduler._CronJobConfig(
        cfg={}, model="some-model", model_cfg={}, cron_default_provider=""
    )
    job = {"provider": "anthropic", "model": "some-model"}
    scheduler._resolve_job_runtime(job, "job1", jc)
    assert captured["requested"] == "anthropic"
