"""Credential-pool last-resort selection and Anthropic terminal-403 rotation.

Covers two resilience behaviors on ``agent/credential_pool.py`` that the
v2026.9.21 baseline was missing (ported forward from the live tree):

  (a) Last-resort selection: an exhausted-but-token-bearing pool never hands
      the caller ``None`` (which builds a keyless ``Bearer None`` request that
      providers answer with a misleading error, re-exhausting the pool in a
      death-loop). It returns the real token of the entry whose cooldown
      elapses soonest, but still reaches ``None`` for genuinely unusable pools
      (DEAD / tokenless / billing / 402) and never re-hands the credential that
      just failed when another option exists.

  (c) Anthropic terminal 403: an org-policy / entitlement 403 whose body reads
      "OAuth authentication is currently not allowed for this organization" is
      permanent — it must mark the entry DEAD (so it does not re-enter rotation
      every hour) yet still rotate to a healthy sibling.
"""

from __future__ import annotations

import json
import time


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _stub_anthropic_singletons(monkeypatch) -> None:
    # HERMES_HOME does not redirect $HOME, so the anthropic singleton seeder
    # still reads the host's ~/.claude/.credentials.json. On a dev box logged
    # into Claude Code that seeds a real healthy entry and "all exhausted" is
    # never true. Stub the singleton readers so the pool holds only what the
    # test writes.
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)
    monkeypatch.setattr("agent.anthropic_credentials.read_hermes_oauth_credentials", lambda: None)


# ---- (a) last-resort selection -------------------------------------------


def test_select_last_resort_returns_real_token_when_all_exhausted(tmp_path, monkeypatch):
    """All entries exhausted but carrying tokens -> select() returns the real
    credential, never None. Returning None makes the caller send a keyless
    'Bearer None' request that providers answer with a misleading error,
    re-exhausting the pool in a death-loop (the 2026-06-16 desktop outage)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    _stub_anthropic_singletons(monkeypatch)
    now = time.time()
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-late",
                        "label": "resets-later",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "tok-late",
                        "last_status": "exhausted",
                        "last_status_at": now,
                        "last_error_code": 429,
                        "last_error_reset_at": now + 3600,
                    },
                    {
                        "id": "cred-soon",
                        "label": "resets-soonest",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "tok-soon",
                        "last_status": "exhausted",
                        "last_status_at": now,
                        "last_error_code": 429,
                        "last_error_reset_at": now + 60,
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    # has_available() stays the honest rotation signal: nothing is genuinely
    # available. Last resort is a selection floor, not restored capacity.
    assert pool.has_available() is False
    selected = pool.select()
    assert selected is not None, "must not return None -> avoids keyless Bearer None"
    # Prefers the entry whose cooldown elapses soonest (most likely live).
    assert selected.id == "cred-soon"
    assert selected.runtime_api_key == "tok-soon"


def test_select_last_resort_skips_dead_and_tokenless_entries(tmp_path, monkeypatch):
    """A DEAD (revoked) entry or one with no token is never a last resort."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    _stub_anthropic_singletons(monkeypatch)
    now = time.time()
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-dead",
                        "label": "revoked",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "tok-dead",
                        "last_status": "dead",
                        "last_status_at": now,
                    },
                    {
                        "id": "cred-empty",
                        "label": "no-token",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "",
                        "last_status": "exhausted",
                        "last_status_at": now,
                        "last_error_reset_at": now + 60,
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    # No usable last resort: DEAD is excluded, tokenless exhausted is excluded.
    assert pool.select() is None


def test_select_last_resort_skips_billing_and_402_entries(tmp_path, monkeypatch):
    """A billing / HTTP-402 exhaustion is terminal for that account, so the
    pool must reach None (surface the real error) rather than re-fire a spent
    credential as a 'last resort'."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    _stub_anthropic_singletons(monkeypatch)
    now = time.time()
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-402",
                        "label": "out-of-credit",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "tok-402",
                        "last_status": "exhausted",
                        "last_status_at": now,
                        "last_error_code": 402,
                        "extra": {"failure_reason": "billing"},
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    assert pool.select() is None


def test_select_last_resort_skips_rejected_key_but_keeps_rate_limited(tmp_path, monkeypatch):
    """A 401 means the provider rejected the key itself; re-sending it cannot
    recover, so it is excluded like a 402 (upstream #111769 expects a sole
    401'd credential to select None). A 429 stays eligible: that is the
    keyless death-loop the last-resort floor exists for."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    _stub_anthropic_singletons(monkeypatch)
    now = time.time()

    def _entry(cid, code):
        return {
            "id": cid, "label": cid, "auth_type": "oauth", "priority": 0,
            "source": "manual", "access_token": f"tok-{cid}",
            "last_status": "exhausted", "last_status_at": now,
            "last_error_code": code, "last_error_reset_at": now + 600,
        }

    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {"anthropic": [_entry("rejected", 401)]}})
    from agent.credential_pool import load_pool

    assert load_pool("anthropic").select() is None

    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {"anthropic": [
        _entry("rejected", 401), _entry("rate-limited", 429),
    ]}})
    picked = load_pool("anthropic").select()
    assert picked is not None and picked.id == "rate-limited"


def test_select_last_resort_skips_usage_limit_quota_wall(tmp_path, monkeypatch):
    """A 429 whose recorded reason is a plan quota wall (``usage_limit_reached``) cannot
    succeed before its reset: the pool reaches None so the caller reports the cooldown and
    reset time (#56810) instead of re-firing a spent key. A plain throttle 429 still
    qualifies as the last resort."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    _stub_anthropic_singletons(monkeypatch)
    now = time.time()

    def _entry(cid, reason):
        return {
            "id": cid, "label": cid, "auth_type": "oauth", "priority": 0,
            "source": "manual", "access_token": f"tok-{cid}",
            "last_status": "exhausted", "last_status_at": now,
            "last_error_code": 429, "last_error_reason": reason,
            "last_error_reset_at": now + 3 * 3600,
        }

    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {"anthropic": [
        _entry("quota-a", "usage_limit_reached"), _entry("quota-b", "usage_limit_reached"),
    ]}})
    from agent.credential_pool import load_pool

    assert load_pool("anthropic").select() is None

    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {"anthropic": [
        _entry("quota-a", "usage_limit_reached"), _entry("throttled", "rate_limit"),
    ]}})
    picked = load_pool("anthropic").select()
    assert picked is not None and picked.id == "throttled"


def test_rotation_last_resort_never_returns_just_failed_key(tmp_path, monkeypatch):
    """When rotation marks the only live key exhausted, the last-resort branch
    must not hand that same key back — retrying a credential we just saw fail
    re-arms a non-convergent rotation. With no other option it reaches None."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    _stub_anthropic_singletons(monkeypatch)
    now = time.time()
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-only",
                        "label": "sole",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "tok-only",
                        "last_status": "ok",
                        "last_status_at": now - 10,
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    assert pool.select().id == "cred-only"
    # A 429 on the sole key: rotation excludes the just-failed key, so the
    # last-resort branch cannot re-hand it. No sibling exists -> surface None.
    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reason": "rate_limit", "message": "slow down"},
    )
    assert next_entry is None


def test_rotation_last_resort_prefers_other_exhausted_over_just_failed(tmp_path, monkeypatch):
    """With two exhausted keys, rotating off one must fall back to the OTHER
    real token (never the just-failed one) rather than returning None."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    _stub_anthropic_singletons(monkeypatch)
    now = time.time()
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-a",
                        "label": "live-a",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "tok-a",
                        "last_status": "ok",
                        "last_status_at": now - 10,
                    },
                    {
                        "id": "cred-b",
                        "label": "benched-b",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "tok-b",
                        "last_status": "exhausted",
                        "last_status_at": now,
                        "last_error_code": 429,
                        "last_error_reset_at": now + 99999,
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    assert pool.select().id == "cred-a"
    # cred-a 429s and is given the SOONEST reset (now+5) — so ordering by
    # cooldown alone would rank the just-failed key FIRST. The last-resort
    # branch must still refuse it (exclude_api_key) and fall to cred-b's own
    # distinct token; otherwise the #97315 guard would collapse rotation to
    # None. This is the discriminating control for the exclude_api_key wiring.
    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reason": "rate_limit", "message": "slow down", "reset_at": now + 5},
    )
    assert next_entry is not None
    assert next_entry.id == "cred-b"
    assert next_entry.runtime_api_key == "tok-b"


# ---- (c) Anthropic terminal 403 ------------------------------------------


def test_anthropic_org_policy_403_marks_dead_and_rotates(tmp_path, monkeypatch):
    """An Anthropic org-policy 403 ("OAuth authentication is currently not
    allowed for this organization") is permanent: mark the entry DEAD so it
    never re-enters rotation on TTL, yet still rotate to a healthy sibling."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _stub_anthropic_singletons(monkeypatch)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-blocked",
                        "label": "org-blocked",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "blocked-at",
                        "refresh_token": "blocked-rt",
                    },
                    {
                        "id": "cred-ok",
                        "label": "healthy",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "healthy-at",
                        "refresh_token": "healthy-rt",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool, STATUS_DEAD

    pool = load_pool("anthropic")
    assert pool.select().id == "cred-blocked"

    next_entry = pool.mark_exhausted_and_rotate(
        status_code=403,
        error_context={
            "message": "OAuth authentication is currently not allowed for this organization",
        },
    )

    # Rotation still hands off to the healthy sibling.
    assert next_entry is not None
    assert next_entry.id == "cred-ok"

    # The blocked credential is permanently DEAD, not merely exhausted.
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    blocked = next(
        e for e in auth_payload["credential_pool"]["anthropic"] if e["id"] == "cred-blocked"
    )
    assert blocked["last_status"] == STATUS_DEAD
    assert blocked["last_error_code"] == 403


def test_anthropic_generic_403_stays_exhausted_not_dead(tmp_path, monkeypatch):
    """Negative control: a run-of-the-mill 403 (no org-policy marker) is
    transient -> exhausted, so it recovers on TTL and is not benched forever."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _stub_anthropic_singletons(monkeypatch)
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-a",
                        "label": "a",
                        "auth_type": "oauth",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "at-a",
                        "refresh_token": "rt-a",
                    },
                    {
                        "id": "cred-b",
                        "label": "b",
                        "auth_type": "oauth",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "at-b",
                        "refresh_token": "rt-b",
                    },
                ]
            },
        },
    )

    from agent.credential_pool import load_pool, STATUS_DEAD

    pool = load_pool("anthropic")
    assert pool.select().id == "cred-a"
    pool.mark_exhausted_and_rotate(
        status_code=403,
        error_context={"message": "Forbidden"},
    )
    auth_payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entry_a = next(
        e for e in auth_payload["credential_pool"]["anthropic"] if e["id"] == "cred-a"
    )
    assert entry_a["last_status"] != STATUS_DEAD


def test_non_anthropic_org_policy_403_not_terminal(tmp_path, monkeypatch):
    """Negative control: the org-policy marker is Anthropic-specific. The same
    body on a non-anthropic provider must NOT be treated as terminal."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)

    from agent.credential_pool import CredentialPool, PooledCredential

    pool = CredentialPool(
        "openai-codex",
        [
            PooledCredential(
                provider="openai-codex",
                id="c1",
                label="c1",
                auth_type="oauth",
                priority=0,
                source="manual",
                access_token="at",
            ),
        ],
    )
    terminal = pool._is_terminal_auth_failure(
        403,
        {"message": "OAuth authentication is currently not allowed for this organization"},
    )
    assert terminal is False
