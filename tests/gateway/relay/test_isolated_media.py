"""Owned media over the actual native HTTP and WebSocket wire, no auth mocks."""
import asyncio
from dataclasses import FrozenInstanceError, replace
import hashlib
import io
from pathlib import Path

import pytest
from aiohttp import ClientSession

from gateway.platforms.base import SendResult
from gateway.relay.adapter import RelayAdapter
from gateway.relay.auth import make_upgrade_token
from gateway.relay.media import RelayMediaClient, media_base_url
from tests.gateway.relay.test_isolated_connector import action, binding, client, event, server


def scoped(url, b):
    return media_base_url(url) + '/gateways/' + b.gateway_id


def headers(b):
    return {'Authorization': 'Bearer ' + make_upgrade_token(b.gateway_id, b.secret)}


@pytest.mark.asyncio
async def test_native_upload_localize_and_typed_send(tmp_path):
    a, b, calls = binding(), binding('domain-b', 'chat-b'), []
    async def native(request):
        calls.append(request)
        return SendResult(True, message_id='sent-media')
    file = tmp_path / 'report.pdf'
    file.write_bytes(b'%PDF-test document')
    async with server([a, b], native, media_store_path=tmp_path / 'private') as (c, url):
        base = scoped(url, a)
        mc = RelayMediaClient(base, a.gateway_id, a.secret)
        ref = await mc.upload(str(file), filename='report.pdf', mime='application/pdf')
        assert ref and ref.startswith(base + '/relay/media/')
        async with client(base.replace('http:', 'ws:') + '/relay', a) as t:
            desc = await t.handshake()
            assert desc.supports_op('send_media')
            result = await t.send_outbound(action(a, 'send_media', source_url=ref,
                media_kind='document', content='caption', filename='report.pdf', reply_to=None))
            assert result['success']
            assert (await t.send_outbound(action(a, 'edit', message_id='sent-media', content='edit')))['success']
        m = calls[0].media
        assert m.filename == 'report.pdf' and m.content_type == 'application/pdf'
        assert m.size == file.stat().st_size
        assert m.sha256 == hashlib.sha256(file.read_bytes()).hexdigest()
        with m.open() as f:
            assert f.read() == file.read_bytes()
        with pytest.raises(FrozenInstanceError):
            m.filename = 'changed'
        # Exercise the real native localizer without constructing the agent runtime.
        class Localizer:
            def _get_media_client(self):
                return mc
        e = event(a)
        e.media_urls, e.media_types = [ref], ['application/pdf']
        await RelayAdapter._localize_inbound_media(Localizer(), e)
        assert len(e.media_urls) == 1 and Path(e.media_urls[0]).suffix == '.pdf'
        assert Path(e.media_urls[0]).read_bytes() == file.read_bytes()
        Path(e.media_urls[0]).unlink()
        assert e.media_types == ['application/pdf']
    assert (tmp_path / 'private').stat().st_mode & 0o077 == 0


@pytest.mark.asyncio
async def test_two_domain_auth_and_handle_attacks_never_call_native(tmp_path):
    a, b, calls = binding(), binding('domain-b', 'chat-b'), []
    async def native(request):
        calls.append(request)
        return SendResult(True)
    file = tmp_path / 'doc.txt'
    file.write_bytes(b'private domain A')
    async with server([a, b], native, media_store_path=tmp_path / 'private') as (c, url):
        base_a, base_b = scoped(url, a), scoped(url, b)
        ref = await RelayMediaClient(base_a, a.gateway_id, a.secret).upload(str(file))
        assert ref
        async with ClientSession() as http:
            for auth in ({}, headers(b), headers(replace(a, secret='wrong')),
                         headers(replace(a, gateway_id=b.gateway_id))):
                async with http.post(base_a + '/relay/media', data=b'secret', headers=auth) as r:
                    assert r.status == 401
                async with http.get(ref, headers=auth) as r:
                    assert r.status == 401
            async with http.get(ref.replace(base_a, base_b), headers=headers(b)) as r:
                assert r.status == 404
                assert b'private domain A' not in await r.read()
        async with client(base_b.replace('http:', 'ws:') + '/relay', b) as t:
            await t.handshake()
            for bad in (ref, ref.replace(base_a, base_b), str(file), '../doc.txt',
                        'file:///etc/passwd', 'https://example.invalid/secret',
                        ref.replace('127.0.0.1', 'localhost'), ref + '?x=1',
                        base_b + '/relay/media/../secret'):
                result = await t.send_outbound(action(b, 'send_media', source_url=bad,
                    media_kind='document', content=''))
                assert result['success'] is False
        assert calls == []


@pytest.mark.asyncio
async def test_streams_above_six_mib_caps_and_restart(tmp_path):
    a, calls = binding(), []
    async def native(request):
        calls.append(request)
        return SendResult(True)
    chunk = b'x' * 65536
    size = 112 * len(chunk)
    expected = hashlib.sha256()
    for _ in range(112):
        expected.update(chunk)
    async def body(count=112):
        for _ in range(count):
            yield chunk
    store = tmp_path / 'private'
    async with server([a], native, media_store_path=store, media_max_bytes=size) as (c, url):
        base = scoped(url, a)
        async with ClientSession() as http:
            async with http.post(base + '/relay/media', data=body(), headers={**headers(a),
                    'Content-Type': 'application/octet-stream', 'X-Media-Filename': 'large.bin'}) as r:
                assert r.status == 200
                metadata = await r.json()
                assert metadata['size'] == size and metadata['sha256'] == expected.hexdigest()
            async with http.post(base + '/relay/media', data=body(113), headers=headers(a)) as r:
                assert r.status == 413
            async with http.post(base + '/relay/media', data=b'x', headers={**headers(a),
                    'Content-Length': str(size + 1)}) as r:
                assert r.status == 413
            async with http.post(base + '/relay/media', data=b'x', headers={**headers(a),
                    'X-Media-Filename': '../escape'}) as r:
                assert r.status == 400
    async with server([a], native, media_store_path=store, media_max_bytes=size) as (c, url):
        async with ClientSession() as http:
            async with http.get(scoped(url, a) + '/relay/media/' + metadata['id'], headers=headers(a)) as r:
                assert r.status == 200
                actual, total = hashlib.sha256(), 0
                async for part in r.content.iter_chunked(65536):
                    actual.update(part)
                    total += len(part)
                assert total == size and actual.hexdigest() == expected.hexdigest()
    assert calls == []


@pytest.mark.asyncio
async def test_exported_media_enqueue_uses_owned_objects_not_event_paths(tmp_path):
    from gateway.platforms.base import MessageType
    a, b = binding(), binding('domain-b', 'chat-b')
    async def native(request):
        return SendResult(True)
    async with server([a, b], native, media_store_path=tmp_path / 'private') as (c, url):
        owned = await c.export_media(a.gateway_id, b'inbound file', filename='document.pdf',
                                     content_type='application/pdf')
        e = event(a)
        e.message_type = MessageType.DOCUMENT
        incoming = asyncio.Queue()
        with pytest.raises(ValueError):
            c.enqueue(b.gateway_id, 'wrong-owner', event(b), media=[owned], media_base_url=scoped(url, b))
        with pytest.raises(ValueError):
            c.enqueue(a.gateway_id, 'forged', e, media=[replace(owned, filename='forged.pdf')], media_base_url=scoped(url, a))
        c.enqueue(a.gateway_id, 'media-event', e, media=[owned], media_base_url=scoped(url, a))
        async with client(scoped(url, a).replace('http:', 'ws:') + '/relay', a, incoming.put) as t:
            await t.handshake()
            received = await asyncio.wait_for(incoming.get(), 2)
            assert received.media_urls == [scoped(url, a) + '/relay/media/' + owned.id]
            assert received.media_types == ['application/pdf']
            assert received.message_type == MessageType.DOCUMENT
            assert e.media_urls == []


@pytest.mark.asyncio
async def test_trusted_export_requires_explicit_stream_and_cannot_be_sent(tmp_path):
    a, b, calls = binding(), binding('domain-b', 'chat-b'), []
    async def native(request):
        calls.append(request)
        return SendResult(True)
    class BoundedStream(io.BytesIO):
        def read(self, n=-1):
            assert 0 < n <= 65536
            return super().read(n)
    async with server([a, b], native, media_store_path=tmp_path / 'private') as (c, url):
        exported = await c.export_media(a.gateway_id, BoundedStream(b'inbound document'),
                                        filename='inbound.pdf', content_type='application/pdf')
        ref = scoped(url, a) + '/relay/media/' + exported.id
        localized = await RelayMediaClient(scoped(url, a), a.gateway_id, a.secret).download(ref)
        assert localized and Path(localized).read_bytes() == b'inbound document'
        Path(localized).unlink()
        for bad in ('/etc/passwd', 'https://example.invalid/file', Path('/etc/passwd')):
            with pytest.raises((ValueError, TypeError)):
                await c.export_media(a.gateway_id, bad, filename='file', content_type='text/plain')
        e = event(a)
        e.media_urls = ['/etc/passwd']
        with pytest.raises(ValueError):
            c.enqueue(a.gateway_id, 'bad', e)
        async with client(scoped(url, a).replace('http:', 'ws:') + '/relay', a) as t:
            await t.handshake()
            assert not (await t.send_outbound(action(a, 'send_media', source_url=ref,
                media_kind='document')))['success']
    assert calls == []
