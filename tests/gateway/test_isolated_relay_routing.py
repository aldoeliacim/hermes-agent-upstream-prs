"""Integration admission regressions: real runner, auth and native Relay wire."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.isolated_routing import load_policy, policy_from_config, dispatch_isolated
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from tests.gateway.relay.test_isolated_connector import client, action


class Sink(BasePlatformAdapter):
    isolated_relay_ops = frozenset({'send', 'edit', 'react', 'typing'})

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
        self.calls.append(('send', chat_id, content, metadata))
        return SendResult(True, message_id='sent-1')

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        self.calls.append(('edit', chat_id, content))
        return SendResult(True, message_id=message_id)

    async def react(self, chat_id, message_id, emoji):
        self.calls.append(('react', chat_id, message_id, emoji))
        return True

    async def unreact(self, chat_id, message_id, emoji=''):
        self.calls.append(('unreact', chat_id, message_id, emoji))
        return True

    async def send_typing(self, chat_id, metadata=None):
        self.calls.append(('typing', chat_id, metadata))


def config(tmp_path):
    return {'gateway': {'isolated_agents': {
        'listen_host': '127.0.0.1', 'listen_port': 0,
        'store_path': str(tmp_path / 'queue.sqlite3'),
        'local_owners': [{'platform': 'telegram', 'chat_id': '900', 'user_id': '900'}],
        'workers': {'guest': {'gateway_id': 'worker-a', 'bot_id': '999', 'secret_env': 'TEST_RELAY_SECRET'}},
        'routes': [{'platform': 'telegram', 'chat_id': '-100', 'chat_type': 'group',
                    'thread_id': '7', 'sender_ids': ['100'], 'domain': 'guest'}]}}}


def event(**changes):
    values = dict(platform=Platform.TELEGRAM, chat_id='-100', chat_type='group',
                  thread_id='7', user_id='100')
    values.update(changes)
    return MessageEvent(text='/bg investigate', message_type=MessageType.COMMAND,
                        message_id='1', source=SessionSource(**values))


async def _async_hook_result(*_args, **_kwargs):
    """Async stand-in for the awaited pre_gateway_dispatch hook."""
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
    r.adapters = {Platform.TELEGRAM: Sink()}
    r._handle_message_with_agent = AsyncMock(return_value='OWNER_PRIVATE_OK')
    r.hooks = SimpleNamespace(emit=AsyncMock())
    # Since #109196 the inbound path awaits pre_gateway_dispatch, so the seam is
    # the ASYNC entry point (hermes_cli.plugins.ainvoke_hook) — patching the old
    # sync symbol silently left _private_hook uncalled and this test went red.
    r._private_hook = MagicMock(side_effect=_async_hook_result)
    monkeypatch.setattr('hermes_cli.plugins.ainvoke_hook', r._private_hook)
    return r


@pytest.mark.asyncio
async def test_owner_dm_retains_real_private_pipeline(runner):
    e = event(chat_id='900', user_id='900', chat_type='dm', thread_id=None)
    e.text, e.message_type = 'hello', MessageType.TEXT
    assert await runner._handle_message(e) == 'OWNER_PRIVATE_OK'
    runner._handle_message_with_agent.assert_awaited_once()
    assert runner._private_hook.called
    assert runner._isolated_runtime is None


@pytest.mark.asyncio
@pytest.mark.parametrize('fields', [
    {'user_id': '901'}, {'user_id': None}, {'thread_id': None}, {'thread_id': '8'},
    {'chat_id': '-200'}, {'scope_id': 'foreign'}, {'user_id': '900'},
    {'chat_id': '900', 'user_id': '100', 'chat_type': 'dm', 'thread_id': None},
])
async def test_wrong_actor_audience_never_private(runner, fields):
    assert await runner._handle_message(event(**fields)) is None
    runner._private_hook.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()
    assert runner._isolated_runtime is None


@pytest.mark.asyncio
async def test_native_forward_busy_observer_and_targeted_callbacks(runner):
    received = asyncio.Queue()
    e = event()
    before = deepcopy(e)
    assert await runner._handle_message(e) is None
    rt = runner._isolated_runtime
    try:
        b = rt.policy.routes[0].binding
        url = f'ws://127.0.0.1:{rt.addresses[0][1]}/relay'
        async with client(url, b, received.put) as t:
            await t.handshake()
            got = await asyncio.wait_for(received.get(), 3)
            assert got.text == '/bg investigate' and got.source.thread_id == '7'
            await runner._handle_gateway_platform_event({'text': 'private observer probe'}, e.source)
            second = event(); second.message_id = '2'
            assert await runner._handle_active_session_busy_message(second, 'unused')
            assert (await asyncio.wait_for(received.get(), 3)).message_id == '2'
            for op, fields in [('send', {'content': 'ok', 'reply_to': None}),
                               ('edit', {'content': 'edit', 'message_id': 'sent-1'}),
                               ('react', {'emoji': '👍', 'message_id': '1'}),
                               ('react', {'remove': True, 'message_id': '1'}), ('typing', {})]:
                assert (await t.send_outbound(action(b, op, **fields)))['success']
            assert not (await t.send_outbound({**action(b, content='wrong'), 'chat_id': '-200'}))['success']
        assert e == before
        runner._private_hook.assert_not_called()
        runner._handle_message_with_agent.assert_not_awaited()
        send = runner.adapters[Platform.TELEGRAM].calls[0]
        assert send[3]['strict_scope'] is True and send[3]['thread_id'] == '7'
    finally:
        await rt.close()


@pytest.mark.asyncio
async def test_stop_before_first_event_cannot_create_listener(runner):
    # An already-running stop task is the first await in stop; admission must
    # close synchronously before waiting for unrelated shutdown work.
    hold = asyncio.Event()
    runner._stop_task = asyncio.create_task(hold.wait())
    stop = asyncio.create_task(runner.stop())
    await asyncio.sleep(0)
    try:
        await runner._handle_message(event())
        assert runner._isolated_runtime is None
    finally:
        hold.set(); await stop
        if runner._isolated_runtime:
            await runner._isolated_runtime.close()


@pytest.mark.asyncio
async def test_unbound_reaction_never_reaches_private_registry(runner):
    await runner._handle_reaction_event({'event_name': 'reaction:added', 'platform': 'telegram',
                                        'chat_id': '-100', 'user_id': '100'})
    runner.hooks.emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiplex_policy_fails_before_profile_mutation(runner):
    runner.config.multiplex_profiles = True
    e = event(chat_id='900', user_id='900', chat_type='dm', thread_id=None)
    handler = runner._make_default_profile_message_handler()
    assert await handler(e) is None
    assert not hasattr(e.source, '_authorization_profile_home')
    runner._private_hook.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_admission_is_observable_without_private_fallback(runner, monkeypatch, caplog):
    from gateway.isolated_routing import IsolatedRelayRuntime
    async def failed(self, e):
        raise OSError('secret synthetic diagnostic must not escape')
    monkeypatch.setattr(IsolatedRelayRuntime, 'enqueue', failed)
    assert await dispatch_isolated(runner, event())
    assert 'Isolated relay admission failed' in caplog.text
    assert 'secret synthetic diagnostic' not in caplog.text
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.parametrize('raw', [None, {}, [], {'workers': {}}, {'unknown': True}])
def test_bad_config_poisoned(tmp_path, raw):
    assert policy_from_config({'gateway': {'isolated_agents': raw}}).invalid


def test_required_missing_duplicate_corrupt_config(tmp_path):
    p = tmp_path / 'config.yaml'
    assert load_policy(p) is None
    (tmp_path / '.isolated-agents-required').touch()
    assert load_policy(p).invalid
    for text in ['', '{}', 'gateway: {}', 'gateway: {}\ngateway: {}', 'gateway: [']:
        p.write_text(text, encoding='utf-8')
        assert load_policy(p).invalid


@pytest.mark.asyncio
async def test_invalid_startup_fails_before_any_listener(runner):
    runner._isolated_policy = policy_from_config({'gateway': {'isolated_agents': {}}})
    runner._start_loop_liveness_guards = MagicMock(side_effect=AssertionError('private startup reached'))
    assert await runner.start() is False
    assert runner._isolated_runtime is None


@pytest.mark.parametrize('update', [
    {'platform': 'signal'}, {'scope_id': 'unsupported-scope'},
    {'thread_id': 'topic-name'}, {'chat_id': 'chat-name'},
    {'chat_type': 'dm', 'thread_id': '7'},
    {'platform': 'whatsapp', 'chat_id': '100@s.whatsapp.net', 'thread_id': '7'},
])
def test_unsupported_native_audience_poisoned(tmp_path, monkeypatch, update):
    monkeypatch.setenv('TEST_RELAY_SECRET', 'synthetic-only-relay-secret')
    cfg = config(tmp_path)
    cfg['gateway']['isolated_agents']['routes'][0].update(update)
    assert policy_from_config(cfg).invalid


@pytest.mark.asyncio
async def test_unqualified_adapter_never_receives_worker_callback(runner):
    from gateway.isolated_routing import IsolatedRelayRuntime
    from gateway.isolated_relay import RelayAction
    native = SimpleNamespace(send=AsyncMock(return_value=SendResult(True)))
    runner.adapters[Platform.TELEGRAM] = native
    rt = IsolatedRelayRuntime(runner._isolated_policy, runner.adapters)
    result = await rt.native_action(RelayAction('worker-a', Platform.TELEGRAM, '-100', '7', None,
                                               'send', content='must not send'))
    assert result.success is False
    native.send.assert_not_awaited()
