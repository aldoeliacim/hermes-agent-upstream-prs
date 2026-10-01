"""Compatibility and fail-closed gates on the native Relay wire."""
from pathlib import Path
from aiohttp import ClientSession
import pytest
from gateway.platforms.base import SendResult
from gateway.relay.media import RelayMediaClient
from tests.gateway.relay.test_isolated_connector import action, binding, client, server
from tests.gateway.relay.test_isolated_media import scoped, headers


@pytest.mark.asyncio
@pytest.mark.parametrize('filename', ['informe español.pdf', '日本語.txt', '100% listo.txt'])
async def test_native_unicode_filename_roundtrip(tmp_path, filename):
    b = binding()
    async def native(request):
        return SendResult(True)
    artifact = tmp_path / filename
    artifact.write_bytes(b'legitimate bytes')
    async with server([b], native, media_store_path=tmp_path / 'private') as (c, url):
        mc = RelayMediaClient(scoped(url, b), b.gateway_id, b.secret)
        ref = await mc.upload(str(artifact))
        assert ref
        owned = c._media.resolve_url(b.gateway_id, ref, url.replace('ws:', 'http:').removesuffix('/relay'))
        assert owned.filename == filename
        async with ClientSession() as http:
            async with http.get(ref, headers=headers(b)) as response:
                assert response.status == 200
                assert response.content_disposition.filename == filename
                assert await response.read() == artifact.read_bytes()
        localized = await mc.download(ref)
        assert localized and Path(localized).suffix == artifact.suffix
        assert Path(localized).read_bytes() == artifact.read_bytes()


@pytest.mark.asyncio
async def test_descriptor_and_dispatch_have_same_native_operation_gate(tmp_path):
    b, calls = binding(), []
    async def native(request):
        calls.append(request)
        return SendResult(True, message_id='sent')
    async with server([b], native, supported_ops=lambda _: ('send', 'react')) as (c, url):
        async with client(url, b) as t:
            desc = await t.handshake()
            assert not desc.supports_edit and not desc.supports_op('typing')
            assert not desc.supports_op('send_media')
            assert (await t.send_outbound(action(b, content='ok')))['success']
            assert not (await t.send_outbound(action(b, 'edit', message_id='sent', content='edit')))['success']
            assert not (await t.send_outbound(action(b, 'typing')))['success']
    assert len(calls) == 1
