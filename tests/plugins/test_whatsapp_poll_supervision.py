"""Supervised inbound poll task.

The poll task is the adapter's only inbound path. Before this, it was a
fire-and-forget ``create_task``: if it ever ended — an exception escaping the
loop, or the loop returning while the adapter still believed it was running —
nothing noticed. The adapter stayed 'connected' but deaf, silently dropping
every inbound message until a full restart.

``_spawn_poll_task`` attaches ``_on_poll_task_done`` as a supervisor: any
NON-orderly termination while the adapter is still running respawns the task;
an orderly stop (disconnect / cancellation / ``_running`` cleared) is left
alone.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = object.__new__(WhatsAppAdapter)
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter.platform = Platform.WHATSAPP
    adapter._running = True
    adapter._shutting_down = False
    adapter._poll_task = None
    adapter._poll_calls = 0
    return adapter


@pytest.mark.asyncio
async def test_spawn_poll_task_sets_task_and_supervisor():
    adapter = _make_adapter()

    async def _fake_poll():
        adapter._poll_calls += 1
        adapter._running = False  # clean exit so the supervisor doesn't respawn
        await asyncio.sleep(0.01)

    adapter._poll_messages = _fake_poll
    adapter._spawn_poll_task()
    assert adapter._poll_task is not None
    await asyncio.sleep(0.05)
    assert adapter._poll_calls == 1


@pytest.mark.asyncio
async def test_crash_while_running_respawns():
    adapter = _make_adapter()

    async def _crashing_poll():
        adapter._poll_calls += 1
        if adapter._poll_calls == 1:
            raise RuntimeError("poller died")
        await asyncio.sleep(0.05)

    adapter._poll_messages = _crashing_poll
    adapter._spawn_poll_task()
    await asyncio.sleep(0.1)
    # crashed once, supervisor respawned it
    assert adapter._poll_calls >= 2


@pytest.mark.asyncio
async def test_unexpected_return_while_running_respawns():
    adapter = _make_adapter()

    async def _returning_poll():
        adapter._poll_calls += 1
        if adapter._poll_calls >= 3:
            adapter._running = False  # stop the ping-pong
        return  # returns immediately while _running is still True

    adapter._poll_messages = _returning_poll
    adapter._spawn_poll_task()
    await asyncio.sleep(0.1)
    assert adapter._poll_calls >= 2


@pytest.mark.asyncio
async def test_orderly_stop_does_not_respawn():
    adapter = _make_adapter()

    async def _clean_poll():
        adapter._poll_calls += 1
        adapter._running = False  # adapter is shutting down normally

    adapter._poll_messages = _clean_poll
    adapter._spawn_poll_task()
    await asyncio.sleep(0.05)
    assert adapter._poll_calls == 1  # no respawn


@pytest.mark.asyncio
async def test_shutting_down_does_not_respawn():
    adapter = _make_adapter()

    async def _poll_then_flag():
        adapter._poll_calls += 1
        adapter._shutting_down = True
        raise RuntimeError("expected during shutdown")

    adapter._poll_messages = _poll_then_flag
    adapter._spawn_poll_task()
    await asyncio.sleep(0.05)
    assert adapter._poll_calls == 1  # shutdown crash is not respawned


@pytest.mark.asyncio
async def test_cancellation_does_not_respawn():
    adapter = _make_adapter()

    async def _long_poll():
        adapter._poll_calls += 1
        await asyncio.sleep(10)

    adapter._poll_messages = _long_poll
    adapter._spawn_poll_task()
    await asyncio.sleep(0.02)
    adapter._poll_task.cancel()
    await asyncio.sleep(0.05)
    assert adapter._poll_calls == 1  # cancellation is orderly, no respawn
