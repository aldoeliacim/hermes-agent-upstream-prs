"""H4-lite: the approval-notify path fails CLOSED for a non-owner silo.

``TurnRunner._approval_notify_sync`` delivers the dangerous-command approval
prompt — including native interactive buttons — to the *originating* chat. When
that chat is a routed contact's DM (a non-owner), surfacing the prompt there
would hand a stranger a working "approve" button. This suppression gate denies
the pending request and posts nothing when a home channel is configured for the
source's platform and the source is not that owner. Owner traffic and
single-user (no-home) deployments are unaffected.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.session import SessionSource
from tools import approval as _approval
from tools.approval_gateway_wait import _ApprovalEntry

APPROVAL = {"command": "rm -rf /tmp/x", "description": "recursive delete", "pattern_key": "k"}


class _ButtonAdapter:
    typed_command_prefix = "/"

    def __init__(self) -> None:
        self.sends: List[str] = []
        self.exec_calls: List[dict] = []

    def pause_typing_for_chat(self, chat_id: str) -> None:
        return None

    async def send_exec_approval(self, **k: Any) -> SendResult:
        self.exec_calls.append(k)
        return SendResult(success=True, message_id="card-1")

    async def send(self, chat_id: str, message: str, **k: Any) -> SendResult:
        self.sends.append(message)
        return SendResult(success=True, message_id="m2")

    async def edit_message(self, *a: Any, **k: Any) -> SendResult:
        return SendResult(success=True)


def _config_with_home(chat_id: str = "OWNER") -> GatewayConfig:
    return GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True, token="t",
                home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id=chat_id, name="home"),
            )
        }
    )


def _runner(adapter, source, session_key, config):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._runner = SimpleNamespace(config=config)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id=source.chat_id,
        _status_thread_metadata={"thread_id": "t1"},
        session_key=session_key, source=source,
    )

    class _Fut:
        def __init__(self, result): self._r = result
        def result(self, timeout=None): return self._r

    runner._schedule = lambda coro, _label: _Fut(asyncio.run(coro))
    runner._close_native_stream_boundary = lambda _why: None
    return runner


def _src(chat_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, user_id=chat_id, chat_id=chat_id,
        user_name="u", chat_type="dm",
    )


@pytest.fixture
def queued():
    def _make(session_key: str) -> _ApprovalEntry:
        entry = _ApprovalEntry(dict(APPROVAL))
        with _approval._lock:
            _approval._gateway_queues[session_key] = [entry]
        return entry
    yield _make
    with _approval._lock:
        _approval._gateway_queues.clear()


def test_non_owner_notify_is_suppressed_and_auto_denied(queued):
    """A non-owner source: no buttons, no message; the pending approval is DENIED."""
    adapter = _ButtonAdapter()
    session_key = "agent:main:telegram:dm:STRANGER"
    entry = queued(session_key)
    _runner(adapter, _src("STRANGER"), session_key, _config_with_home("OWNER"))._approval_notify_sync(
        dict(APPROVAL)
    )

    assert adapter.exec_calls == [], "no native approval buttons may reach a non-owner"
    assert adapter.sends == [], "no plain-text prompt may reach a non-owner"
    assert entry.event.is_set() and entry.result == "deny", "pending approval must be auto-denied"


def test_owner_notify_still_shows_native_buttons(queued):
    """The owner (source matching the home) still gets the interactive card."""
    adapter = _ButtonAdapter()
    session_key = "agent:main:telegram:dm:OWNER"
    queued(session_key)
    _runner(adapter, _src("OWNER"), session_key, _config_with_home("OWNER"))._approval_notify_sync(
        dict(APPROVAL)
    )

    assert len(adapter.exec_calls) == 1, "owner must still receive native approval buttons"


def test_no_home_configured_is_not_suppressed(queued):
    """Single-user deploy (no home configured): legacy behavior, prompt shown."""
    adapter = _ButtonAdapter()
    session_key = "agent:main:telegram:dm:SOLO"
    queued(session_key)
    empty = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")})
    _runner(adapter, _src("SOLO"), session_key, empty)._approval_notify_sync(dict(APPROVAL))

    assert len(adapter.exec_calls) == 1, "no home => nothing to protect => prompt proceeds"
