"""Per-spawn bridge capability token (loopback hardening).

The WhatsApp bridge exposes state-changing HTTP endpoints on loopback. Any local
process (including the agent's own terminal / execute_code subprocess) could POST
``/send`` with an arbitrary ``chatId`` and speak as the user. The bridge enforces a
per-spawn capability token (``X-Hermes-Bridge-Token``); this suite pins the ADAPTER
(client) half that was dropped on replay:

* ``_load_or_create_bridge_token`` persists a token 0600 in the session dir and
  reuses it across calls / gateway restarts.
* ``_bridge_auth_headers`` / ``_new_bridge_http_session`` carry it as a default
  header so every send authenticates without per-call-site changes.
* ``_bridge_env`` passes ``WHATSAPP_BRIDGE_TOKEN`` to the spawned bridge.
* ``_load_standalone_bridge_token`` (cron path) only READS the file, never creates it.
"""

import os
import stat
from pathlib import Path

import pytest

from gateway.config import PlatformConfig


def _make_adapter(tmp_path: Path):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    # Real __init__ so allowlist/policy resolution is exercised; pin the session dir into the tmp
    # dir so the token file is written/read there and never touches a real profile.
    adapter = WhatsAppAdapter(
        PlatformConfig(enabled=True, extra={"session_path": str(tmp_path), "dm_policy": "open"})
    )
    assert adapter._session_path == tmp_path
    return adapter


def test_load_or_create_persists_0600_and_is_stable(tmp_path):
    adapter = _make_adapter(tmp_path)
    token = adapter._load_or_create_bridge_token()
    assert token and isinstance(token, str)
    token_file = tmp_path / "bridge-token"
    assert token_file.exists()
    # Secret must never be world/group readable.
    mode = stat.S_IMODE(os.stat(token_file).st_mode)
    assert mode == 0o600, oct(mode)
    # Stable across calls (cached) and across a fresh adapter reading the same file.
    assert adapter._load_or_create_bridge_token() == token
    adapter2 = _make_adapter(tmp_path)
    assert adapter2._load_or_create_bridge_token() == token


def test_auth_headers_and_session_carry_token(tmp_path):
    adapter = _make_adapter(tmp_path)
    token = adapter._load_or_create_bridge_token()
    headers = adapter._bridge_auth_headers()
    assert headers.get("X-Hermes-Bridge-Token") == token


def test_bridge_env_sets_token(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path)
    adapter._send_read_receipts = False
    env = adapter._bridge_env()
    assert env.get("WHATSAPP_BRIDGE_TOKEN")
    # And it matches the persisted/cached token.
    assert env["WHATSAPP_BRIDGE_TOKEN"] == adapter._bridge_token


def test_standalone_loader_reads_but_never_creates(tmp_path):
    from plugins.platforms.whatsapp.adapter import _load_standalone_bridge_token

    extra = {"session_path": str(tmp_path)}
    # No file yet -> None, and it must NOT create one (fail-closed, live adapter is sole writer).
    assert _load_standalone_bridge_token(extra) is None
    assert not (tmp_path / "bridge-token").exists()
    # With a file present -> returns its contents.
    (tmp_path / "bridge-token").write_text("secret-abc\n", encoding="utf-8")
    assert _load_standalone_bridge_token(extra) == "secret-abc"


def test_media_upload_timeout_is_size_and_type_aware(tmp_path):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    small = tmp_path / "small.bin"
    small.write_bytes(b"x" * 1024)
    # Non-video/doc floor is 120s; video/doc floor is 300s.
    assert WhatsAppAdapter._media_upload_timeout_ms(str(small), "image") == 120_000
    assert WhatsAppAdapter._media_upload_timeout_ms(str(small), "video") == 300_000

    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (200 * 1024 * 1024))  # 200 MB -> scaled above the floor
    scaled = WhatsAppAdapter._media_upload_timeout_ms(str(big), "video")
    assert scaled > 300_000
    # Capped at 900s regardless of size.
    assert scaled <= 900_000


@pytest.mark.parametrize("missing", ["nope.bin"])
def test_media_upload_timeout_handles_missing_file(tmp_path, missing):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    # Missing file -> size 0, falls back to the type floor without raising.
    assert WhatsAppAdapter._media_upload_timeout_ms(str(tmp_path / missing), "image") == 120_000
