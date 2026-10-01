"""The inbound message source is visible to tools during a live turn, and only then.

send_message(target="current") and action="silent" resolve the chat through
tools.approval.get_current_message_source(). The gateway binds it around the turn in
_handle_message_with_agent and runs the agent through _run_in_executor_with_context
(copy_context), so the agent's worker thread inherits it. Driven through the real
inbound path: the agent's own run reads the source on the worker thread.
"""
from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from tests.gateway import test_owner_register_gate_e2e as e2e


@pytest.mark.asyncio
async def test_worker_thread_sees_the_inbound_source_and_it_is_cleared_after(monkeypatch, tmp_path):
    from tools.approval import get_current_message_source

    adapter = e2e.WireAdapter()
    runner = e2e._runner(monkeypatch, tmp_path, adapter, final=e2e.REPLY)
    e2e._oracle(monkeypatch, "DELIVER")
    seen = {}

    async def _run_agent(*_a, **_kw):
        # Stand in for the agent loop: the tool layer runs on the executor thread.
        def _tool_call():
            src = get_current_message_source()
            seen["chat_id"] = getattr(src, "chat_id", None)
            seen["thread"] = threading.current_thread().name
        await runner._run_in_executor_with_context(_tool_call)
        return {
            "final_response": e2e.REPLY,
            "messages": [{"role": "user", "content": e2e.REQUEST}, {"role": "assistant", "content": e2e.REPLY}],
            "tools": [], "history_offset": 0, "last_prompt_tokens": 0, "api_calls": 1, "failed": False,
        }

    runner._run_agent = AsyncMock(side_effect=_run_agent)
    src = e2e._source(e2e.CONTACT, "Robin")
    await runner._handle_message_with_agent(e2e._event(src), src, f"agent:main:whatsapp:dm:{e2e.CONTACT}", 1)

    assert seen.get("chat_id") == e2e.CONTACT, "the tool thread could not see the inbound chat"
    assert seen.get("thread") != threading.current_thread().name, "the probe must run off the event loop"
    assert get_current_message_source() is None, "the source leaked past the turn"


def test_no_source_outside_a_turn():
    from tools.approval import get_current_message_source

    assert asyncio.run(asyncio.to_thread(get_current_message_source)) is None
