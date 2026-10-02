"""native_action's send_media branch must resolve gateway.native_media at call time.

The import inside IsolatedRuntime.native_action is lazy, so nothing at module-import
time proves the module exists. The 9.21 cycle dropped gateway/native_media.py,
gateway/native_admission.py and scripts/isolated_candidate_gate.py while keeping the
caller in gateway/isolated_routing.py, which left a ModuleNotFoundError waiting on the
first isolated-relay media send. tests/gateway/relay/test_isolated_media.py exercises
send_media through RelayAdapter, not through this branch, so the whole relay suite
stayed green with the module missing.

These tests drive native_action directly with an adapter that advertises send_media.
"""
import asyncio
import hashlib
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from gateway.isolated_routing import dispatch_isolated
from tests.gateway.relay.test_isolated_connector import client, action


class MediaSink(BasePlatformAdapter):
    """Adapter that actually declares the op, so native_action reaches the import."""

    isolated_relay_ops = frozenset({'send', 'edit', 'react', 'typing', 'send_media'})

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.calls = []

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {'id': chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append(('send', chat_id, content))
        return SendResult(True, message_id='sent-1')

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        return SendResult(True, message_id=message_id)

    async def react(self, chat_id, message_id, emoji):
        return True

    async def unreact(self, chat_id, message_id, emoji=''):
        return True

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def send_owned_media(self, chat_id, data, *, filename=None, content_type=None,
                               kind=None, caption='', reply_to=None, metadata=None):
        self.calls.append(('media', chat_id, data, filename, content_type, kind))
        return SendResult(True, message_id='sent-media')


def config(tmp_path):
    return {'gateway': {'isolated_agents': {
        'listen_host': '127.0.0.1', 'listen_port': 0,
        'store_path': str(tmp_path / 'queue.sqlite3'),
        'local_owners': [{'platform': 'telegram', 'chat_id': '900', 'user_id': '900'}],
        'workers': {'guest': {'gateway_id': 'worker-a', 'bot_id': '999',
                              'secret_env': 'TEST_RELAY_SECRET'}},
        'routes': [{'platform': 'telegram', 'chat_id': '-100', 'chat_type': 'group',
                    'thread_id': '7', 'sender_ids': ['100'], 'domain': 'guest'}]}}}


def event(**changes):
    values = dict(platform=Platform.TELEGRAM, chat_id='-100', chat_type='group',
                  thread_id='7', user_id='100')
    values.update(changes)
    return MessageEvent(text='/bg ship it', message_type=MessageType.COMMAND,
                        message_id='1', source=SessionSource(**values))


async def _async_hook_result(*_args, **_kwargs):
    return []


@pytest.fixture
def runner(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    monkeypatch.setenv('TEST_RELAY_SECRET', 'synthetic-only-relay-secret')
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', '100,900')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    raw = config(tmp_path)
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(raw), encoding='utf-8')
    r = GatewayRunner(config=GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)}))
    r.adapters = {Platform.TELEGRAM: MediaSink()}
    r._handle_message_with_agent = AsyncMock(return_value='OK')
    r.hooks = SimpleNamespace(emit=AsyncMock())
    r._private_hook = MagicMock(side_effect=_async_hook_result)
    monkeypatch.setattr('hermes_cli.plugins.ainvoke_hook', r._private_hook)
    return r


def test_native_media_module_imports_on_its_own():
    """The dropped module must be importable, not merely referenced."""
    from gateway import native_media
    assert callable(native_media.read_owned)


def test_native_admission_module_imports_on_its_own():
    """Same for the admission module dropped alongside it in the 9.21 cycle."""
    from gateway import native_admission
    assert callable(native_admission.bind_native_admission)
    assert callable(native_admission.isolated_runner)


def test_candidate_gate_script_present():
    """scripts/isolated_candidate_gate.py is operator tooling; its loss is silent."""
    from pathlib import Path
    import gateway
    root = Path(gateway.__file__).resolve().parent.parent
    assert (root / 'scripts' / 'isolated_candidate_gate.py').is_file()


@pytest.mark.asyncio
async def test_send_media_reaches_the_lazy_import(runner):
    """Drive the real branch: the missing module must not surface as ModuleNotFoundError.

    Any adapter declaring send_media gets a connector media store, so upload an owned
    artifact and send it. Before the restore this raised ModuleNotFoundError for
    gateway.native_media instead of returning a SendResult.
    """
    received = asyncio.Queue()
    e = event()
    before = deepcopy(e)
    assert await runner._handle_message(e) is None
    rt = runner._isolated_runtime
    assert rt is not None
    try:
        assert rt.connector._media is not None, 'send_media adapter must get a media store'
        b = rt.policy.routes[0].binding
        url = f'ws://127.0.0.1:{rt.addresses[0][1]}/relay'
        async with client(url, b, received.put) as t:
            await t.handshake()
            assert await asyncio.wait_for(received.get(), 3)

            # Seed the store with a genuinely uploaded, owned artifact, then address
            # it the way the wire does: an absolute URL in this connector's namespace.
            payload = b'%PDF-native-owned-document'

            async def chunks():
                yield payload

            uploaded = await rt.connector._media.put(
                b.gateway_id, chunks(), filename='report.pdf',
                content_type='application/pdf', uploaded=True)
            origin = f'http://127.0.0.1:{rt.addresses[0][1]}'
            ref = f'{origin}/gateways/{b.gateway_id}/relay/media/{uploaded.id}'

            result = await t.send_outbound(action(b, 'send_media', source_url=ref,
                                                  media_kind='document', content='caption',
                                                  filename='report.pdf', reply_to=None))
            assert result['success'] is True, f'send_media failed: {result}'

        media = [c for c in runner.adapters[Platform.TELEGRAM].calls if c[0] == 'media']
        assert len(media) == 1, 'adapter never received the owned bytes'
        _, chat_id, data, filename, content_type, kind = media[0]
        assert data == payload
        assert hashlib.sha256(data).hexdigest() == hashlib.sha256(payload).hexdigest()
        assert filename == 'report.pdf'
        assert content_type == 'application/pdf'
        assert kind == 'document'
        assert e == before
        runner._handle_message_with_agent.assert_not_awaited()
    finally:
        await rt.close()


@pytest.mark.asyncio
async def test_foreign_media_never_crosses_the_boundary(runner):
    """read_owned re-resolves ownership at the root consumer; a foreign ref must fail."""
    received = asyncio.Queue()
    assert await runner._handle_message(event()) is None
    rt = runner._isolated_runtime
    try:
        b = rt.policy.routes[0].binding
        url = f'ws://127.0.0.1:{rt.addresses[0][1]}/relay'
        async with client(url, b, received.put) as t:
            await t.handshake()
            assert await asyncio.wait_for(received.get(), 3)
            # A media URL from ANOTHER worker's namespace must never be resolved for
            # this binding, and the bytes must never reach the adapter.
            async def chunks():
                yield b'secret'

            await rt.connector._media.put(
                b.gateway_id, chunks(), filename='stolen.pdf',
                content_type='application/pdf', uploaded=True)
            foreign_origin = f'http://127.0.0.1:{rt.addresses[0][1]}'
            foreign = (f'{foreign_origin}/gateways/worker-B/relay/media/'
                       f"{'0' * 32}")
            result = await t.send_outbound(action(b, 'send_media', source_url=foreign,
                                                  media_kind='document', content='',
                                                  filename='stolen.pdf', reply_to=None))
            assert result['success'] is not True
        assert not [c for c in runner.adapters[Platform.TELEGRAM].calls if c[0] == 'media']
    finally:
        await rt.close()
