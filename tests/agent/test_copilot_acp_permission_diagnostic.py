"""The ACP permission-denial path must surface a one-time diagnostic.

Hermes denies the ACP agent's native tool-permission requests by design (it
drives tools via injected markers, not the agent's own toolset). But if the
agent is launched in a permission mode that routes EVERY tool through that path
(Claude Agent SDK "dontAsk"/"default", Copilot CLI without bypass), the silent
denial leaves a do-nothing agent with no diagnostic. These tests lock the
contract: the denial still happens (security/design unchanged) AND a single
actionable warning is emitted on the first denial, not one-per-call spam.
"""

import json
import logging

from agent.copilot_acp_client import CopilotACPClient


def _make_client() -> CopilotACPClient:
    return CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        command="/bin/true",
        args=["--acp", "--stdio"],
    )


class _FakeStdin:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def write(self, data):
        self.sent.append(data)

    def flush(self):
        pass


class _FakeProc:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()


def _perm_msg() -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": "session/request_permission", "params": {}}


def test_permission_denied_payload_unchanged():
    """The deny response shape (cancelled outcome) must not regress."""
    client = _make_client()
    proc = _FakeProc()
    handled = client._handle_server_message(
        _perm_msg(), process=proc, cwd="/tmp", text_parts=None, reasoning_parts=None
    )
    assert handled is True
    assert proc.stdin.sent, "a denial response must be written"
    resp = json.loads(proc.stdin.sent[-1])
    assert resp["id"] == 1
    assert resp["result"]["outcome"]["outcome"] == "cancelled"


def test_first_denial_warns_with_actionable_cause(caplog):
    client = _make_client()
    proc = _FakeProc()

    with caplog.at_level(logging.WARNING):
        handled = client._handle_server_message(
            _perm_msg(), process=proc, cwd="/tmp", text_parts=None, reasoning_parts=None
        )

    assert handled is True
    # denial still sent
    assert any("cancelled" in s for s in proc.stdin.sent)
    # exactly one actionable warning, naming the bypass fix
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "bypass" in warnings[0].message.lower()


def test_repeated_denials_warn_only_once(caplog):
    client = _make_client()
    proc = _FakeProc()

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            client._handle_server_message(
                _perm_msg(), process=proc, cwd="/tmp", text_parts=None, reasoning_parts=None
            )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    # every denial still answered
    assert sum("cancelled" in s for s in proc.stdin.sent) == 5
    # counter kept incrementing past the one-time warning
    assert client._permission_denied_count == 5
