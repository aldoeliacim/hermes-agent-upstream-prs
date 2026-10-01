"""Real native clients at reviewed startup, reconnect and reaction seams."""
import asyncio
import pytest
from aiohttp import ClientSession
from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.relay.adapter import RelayAdapter
from gateway.relay.auth import make_upgrade_token
from tests.gateway.relay.test_isolated_connector import binding, client, event, server


@pytest.mark.asyncio
async def test_native_topic_react_and_unreact_use_target_ledger():
    b, received, calls = binding(), asyncio.Queue(), []
    async def native(request):
        calls.append(request)
        return SendResult(True)
    async with server([b], native) as (connector, url):
        connector.enqueue(b.gateway_id, 'delivery', event(b))
        async with client(url, b, received.put) as transport:
            descriptor = await transport.handshake()
            await asyncio.wait_for(received.get(), 2)
            adapter = RelayAdapter(PlatformConfig(), descriptor, transport=transport)
            assert await adapter.react(b.chat_id, 'message-1', '👍')
            assert await adapter.unreact(b.chat_id, 'message-1', '👍')
            assert not await adapter.react(b.chat_id, 'foreign-message', '👍')
    assert len(calls) == 2 and calls[0].thread_id == b.thread_id


@pytest.mark.asyncio
async def test_valid_connection_contention_is_retryable_not_auth_revocation():
    b = binding()
    async def native(request):
        return SendResult(True)
    async with server([b], native) as (connector, url):
        async with client(url, b) as original:
            await original.handshake()
            async with ClientSession() as http:
                ws = await http.ws_connect(url, headers={
                    'Authorization': 'Bearer ' + make_upgrade_token(b.gateway_id, b.secret)})
                await ws.receive()
                assert ws.close_code == 1013
                await ws.close()
        async with client(url, b) as replacement:
            assert await replacement.handshake()


@pytest.mark.asyncio
async def test_no_descriptor_or_queue_replay_until_native_ready():
    b, ready = binding(), False
    async def native(request):
        return SendResult(True)
    async with server([b], native, supported_ops=lambda _: ('send',) if ready else ()) as (connector, url):
        connector.enqueue(b.gateway_id, 'queued', event(b))
        async with ClientSession() as http:
            ws = await http.ws_connect(url, headers={
                'Authorization': 'Bearer ' + make_upgrade_token(b.gateway_id, b.secret)})
            await asyncio.wait_for(ws.receive(), 2)
            assert ws.close_code == 1013
            await ws.close()
        ready = True
        received = asyncio.Queue()
        async with client(url, b, received.put) as transport:
            descriptor = await transport.handshake()
            assert descriptor.supported_ops == ('send',)
            assert (await asyncio.wait_for(received.get(), 2)).message_id == 'message-1'


@pytest.mark.asyncio
async def test_native_cron_delivery_metadata_does_not_grant_routing_authority():
    b, calls = binding(), []
    async def native(request):
        calls.append(request)
        return SendResult(True, message_id='cron-result')
    async with server([b], native) as (connector, url):
        async with client(url, b) as transport:
            descriptor = await transport.handshake()
            adapter = RelayAdapter(PlatformConfig(), descriptor, transport=transport)
            result = await adapter.send(b.chat_id, 'cron result', metadata={
                'job_id': 'synthetic-job-1', 'thread_id': b.thread_id})
            assert result.success
            wrong = await adapter.send('foreign', 'must not send', metadata={
                'job_id': 'synthetic-job-1', 'thread_id': b.thread_id})
            assert not wrong.success
    assert len(calls) == 1 and calls[0].chat_id == b.chat_id
