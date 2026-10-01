"""Gateway config loader expands ``${VAR}`` refs like the CLI loader does.

``hermes_cli.config.load_config`` expands ``${VAR}`` / ``${env:VAR}`` in
every string value.  ``gateway.config.load_gateway_config`` builds its own
dict straight from the YAML and did NOT, so a webhook route declared as
``secret: ${SEERR_WEBHOOK_SECRET}`` reached the adapter as the literal
placeholder and every delivery was rejected as unsigned — silently, since a
wrong secret and an unexpanded one look the same on the wire.  Found while
wiring Seerr → per-user WhatsApp notifications (2026-09-11).
"""

from __future__ import annotations

import textwrap

import pytest

from gateway.config import Platform, load_gateway_config


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path, raising=False)
    return tmp_path


def _write(home, text: str) -> None:
    (home / "config.yaml").write_text(textwrap.dedent(text), encoding="utf-8")


def test_webhook_route_secret_env_ref_is_expanded(home, monkeypatch):
    monkeypatch.setenv("ROUTE_SECRET_FOR_TEST", "s3cr3t-value")
    _write(home, """
        platforms:
          webhook:
            enabled: true
            extra:
              host: 127.0.0.1
              port: 18645
              routes:
                r1:
                  secret: ${ROUTE_SECRET_FOR_TEST}
                  deliver_only: true
                  deliver: telegram
                  prompt: "{text}"
    """)
    cfg = load_gateway_config()
    route = cfg.platforms[Platform.WEBHOOK].extra["routes"]["r1"]
    assert route["secret"] == "s3cr3t-value"


def test_unset_env_ref_stays_verbatim_so_startup_can_detect_it(home, monkeypatch):
    monkeypatch.delenv("ROUTE_SECRET_UNSET_FOR_TEST", raising=False)
    _write(home, """
        platforms:
          webhook:
            enabled: true
            extra:
              routes:
                r1:
                  secret: ${ROUTE_SECRET_UNSET_FOR_TEST}
    """)
    cfg = load_gateway_config()
    assert cfg.platforms[Platform.WEBHOOK].extra["routes"]["r1"]["secret"] == "${ROUTE_SECRET_UNSET_FOR_TEST}"
