"""Managed connector security contracts over real loopback Relay sockets."""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import sqlite3
import subprocess
import sys

import pytest
from aiohttp import ClientSession, WSMsgType, web
from websockets.exceptions import ConnectionClosed

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.relay.auth import make_upgrade_token
from gateway.relay.ws_transport import WebSocketRelayTransport
from gateway.session import SessionSource


def binding(gateway_id="domain-a", chat_id="chat-a", thread_id="topic-a"):
    from gateway.isolated_relay import RelayBinding
    return RelayBinding(gateway_id=gateway_id, platform=Platform.TELEGRAM,
                        chat_id=chat_id, thread_id=thread_id, bot_id="test-bot")


def event(b, message_id="message-1"):
    return MessageEvent(text="hello", message_id=message_id,
                        source=SessionSource(platform=b.platform, chat_id=b.chat_id,
                                             thread_id=b.thread_id, user_id="actor-1",
                                             user_id_alt="actor-alt", chat_type="group"))


@asynccontextmanager
async def server(bindings, callback, **kwargs):
    from gateway.isolated_relay import ManagedRelayConnector
    connector = ManagedRelayConnector(bindings, callback, **kwargs)
    runner = web.AppRunner(connector.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        yield connector, f"ws://127.0.0.1:{port}/relay"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def client(url, b, inbound=None, **overrides):
    options: dict = dict(gateway_id=b.gateway_id, upgrade_secret=b.secret,
                   connect_timeout_s=2, outbound_timeout_s=2, reconnect=False)
    options.update(overrides)
    transport = WebSocketRelayTransport(url, b.platform.value, b.bot_id, **options)
    if inbound is not None:
        transport.set_inbound_handler(inbound)
    try:
        await transport.connect()
        yield transport
    finally:
        await transport.disconnect()


def action(b, op="send", **extra):
    return dict(op=op, chat_id=b.chat_id, metadata={"thread_id": b.thread_id}, **extra)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_to", [None, "message-1"])
@pytest.mark.parametrize("notify", [False, True])
async def test_actual_gateway_final_reply_shape_is_accepted(reply_to, notify):
    a, calls = binding(), []
    async def native(request):
        calls.append(request)
        return SendResult(True, message_id="reply-1")
    async with server([a], native) as (connector, url):
        connector.enqueue(a.gateway_id, "delivery-1", event(a))
        async with client(url, a) as transport:
            await transport.handshake()
            payload = action(a, content="reply", reply_to=reply_to)
            payload["metadata"] = {"thread_id": a.thread_id, "notify": notify, "user_id": "actor-1"}
            result = await transport.send_outbound(payload)
            assert result["success"] is True
    assert len(calls) == 1
    assert calls[0].notify is notify
    assert calls[0].reply_to == reply_to


@pytest.mark.asyncio
@pytest.mark.parametrize("emoji", [None, ""])
async def test_uniform_unreact_accepts_unspecified_emoji(emoji):
    a, calls = binding(), []
    async def native(request):
        calls.append(request)
        return SendResult(True)
    async with server([a], native) as (connector, url):
        connector.enqueue(a.gateway_id, "delivery-1", event(a))
        async with client(url, a) as transport:
            await transport.handshake()
            result = await transport.send_outbound(action(
                a, "react", message_id="message-1", emoji=emoji, remove=True))
            assert result["success"] is True
    assert calls[0].remove is True


@pytest.mark.asyncio
async def test_real_client_roundtrip_preserves_identity_and_scopes_native_actions():
    from gateway.isolated_relay import RelayAction
    a, b = binding(), binding("domain-b", "chat-b")
    calls, received_a, received_b = [], asyncio.Queue(), asyncio.Queue()

    async def native(request):
        calls.append(request)
        return SendResult(success=True, message_id="sent-1")

    original = event(a)
    before = deepcopy(original)
    async with server([a, b], native) as (connector, url):
        connector.enqueue(a.gateway_id, "delivery-1", original)
        async with client(url, a, inbound=received_a.put) as ta, client(url, b, inbound=received_b.put) as tb:
            ta.set_inbound_handler(received_a.put)
            tb.set_inbound_handler(received_b.put)
            descriptor = await ta.handshake()
            await tb.handshake()
            got = await asyncio.wait_for(received_a.get(), 2)
            assert got.message_id == original.message_id
            assert got.source.user_id == "actor-1"
            assert got.source.user_id_alt == "actor-alt"
            assert got.source.thread_id == a.thread_id
            assert got.source.delivered_via_upstream_relay
            for op, fields in [("send", {"content": "response"}),
                               ("send", {"content": "reply", "reply_to": "message-1"}),
                               ("edit", {"content": "edited", "message_id": "sent-1"}),
                               ("react", {"message_id": "message-1", "emoji": "👀"}),
                               ("react", {"message_id": "message-1", "emoji": "👀", "remove": True}),
                               ("typing", {})]:
                assert descriptor.supports_op(op)
                result = await ta.send_outbound(action(a, op, **fields), platform=a.platform.value)
                assert result["success"] is True
            # Barrier through the second socket, not a negative sleep race.
            assert (await tb.send_outbound(action(b, "typing")))["success"]
            assert received_b.empty()
            assert all(isinstance(call, RelayAction) for call in calls)
            assert all((call.gateway_id, call.platform, call.chat_id, call.thread_id) ==
                       (a.gateway_id, a.platform, a.chat_id, a.thread_id) for call in calls[:-1])
            assert calls[1].reply_to == "message-1"
            assert calls[4].remove is True
            for unsupported in ("send_media", "prompt", "follow_up", "thread_create", "get_chat_info"):
                assert not descriptor.supports_op(unsupported)
            assert not descriptor.supports_threads
            assert not descriptor.supports_draft_streaming
    assert original == before
    assert a.secret != b.secret
    assert a.secret not in repr(a)
    with pytest.raises(FrozenInstanceError):
        setattr(a, "chat_id", "other")


@pytest.mark.asyncio
@pytest.mark.parametrize("credentials", ["wrong-secret", "other-id", "cross-secret", "missing"])
async def test_authentication_rejects_without_callback(credentials):
    a, b = binding(), binding("domain-b", "chat-b")
    calls = []

    async def native(request):
        calls.append(request)
        return SendResult(True)

    opts = {"wrong-secret": {"upgrade_secret": "wrong"},
            "other-id": {"gateway_id": "other-domain"},
            "cross-secret": {"gateway_id": b.gateway_id},
            "missing": {"upgrade_secret": None}}[credentials]
    async with server([a, b], native) as (_, url):
        with pytest.raises((ConnectionClosed, ConnectionError, asyncio.TimeoutError)):
            async with client(url, a, **opts) as t:
                await t.handshake()
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("attack", [
    {"chat_id": "other"}, {"metadata": {"thread_id": "other"}},
    {"metadata": {}}, {"thread_id": "other"}, {"platform": "discord"},
    {"gatewayId": "domain-b"}, {"op": "unknown"}, {"op": "send_media"},
    {"op": "prompt"}, {"metadata": {"thread_id": "topic-a", "chat_id": "other"}},
    {"metadata": {"thread_id": "topic-a", "thread_ts": "other"}},
    {"op": "react", "message_id": "foreign", "emoji": "👀"},
    {"reply_to": "foreign"},
])
async def test_forged_actions_never_reach_native_callback(attack):
    a, calls = binding(), []

    async def native(request):
        calls.append(request)
        return SendResult(True)

    async with server([a], native) as (_, url):
        async with client(url, a) as t:
            await t.handshake()
            forged = action(a, content="attempt")
            forged.update(attack)
            result = await t.send_outbound(forged)
            assert result == {"success": False, "error": "request rejected"}
            assert a.secret not in str(result)
            result = await t.send_outbound(action(a, content="attempt"), platform="discord")
            assert result["success"] is False
    assert calls == []


@pytest.mark.asyncio
async def test_raw_frames_cannot_rebind_or_invoke_before_hello():
    import json
    a, calls = binding(), []

    async def native(request):
        calls.append(request)
        return SendResult(True)

    headers = {"Authorization": "Bearer " + make_upgrade_token(a.gateway_id, a.secret)}
    async with server([a], native) as (_, url), ClientSession() as http:
        for frame in [{"type": "hello", "platform": "discord", "botId": a.bot_id},
                      {"type": "hello", "platform": a.platform.value, "botId": "foreign"},
                      {"type": "outbound", "requestId": "1", "action": action(a, "typing")},
                      {"type": "hello", "platform": a.platform.value, "botId": a.bot_id,
                       "gatewayId": "domain-b"}]:
            async with http.ws_connect(url, headers=headers) as ws:
                await ws.send_str(json.dumps(frame) + "\n")
                response = await asyncio.wait_for(ws.receive(), 2)
                assert response.type == WSMsgType.CLOSE
                assert response.data == 4401
    assert not calls


@pytest.mark.asyncio
async def test_enqueue_bound_duplicate_and_foreign_events_fail_predictably():
    a, calls = binding(), []

    async def native(request):
        calls.append(request)
        return SendResult(True)

    async with server([a], native, max_pending=1) as (connector, url):
        connector.enqueue(a.gateway_id, "delivery-1", event(a))
        with pytest.raises(ValueError, match="duplicate delivery"):
            connector.enqueue(a.gateway_id, "delivery-1", event(a))
        with pytest.raises(asyncio.QueueFull):
            connector.enqueue(a.gateway_id, "delivery-2", event(a, "message-2"))
        for field, value in [("platform", Platform.DISCORD), ("chat_id", "foreign"),
                             ("thread_id", None)]:
            foreign = event(a)
            setattr(foreign.source, field, value)
            with pytest.raises(ValueError, match="event rejected"):
                connector.enqueue(a.gateway_id, "foreign", foreign)
        with pytest.raises(ValueError, match="event rejected"):
            connector.enqueue("unknown", "foreign", event(a))
        got = asyncio.Queue()
        async with client(url, a) as t:
            t.set_inbound_handler(got.put)
            await t.handshake()
            await asyncio.wait_for(got.get(), 2)
            # Reader sends ack before handling this response; server processes FIFO.
            await t.send_outbound(action(a, "typing"))
            with pytest.raises(ValueError, match="duplicate delivery"):
                connector.enqueue(a.gateway_id, "delivery-1", event(a))
            connector.enqueue(a.gateway_id, "delivery-2", event(a, "message-2"))
            assert (await asyncio.wait_for(got.get(), 2)).message_id == "message-2"


@pytest.mark.asyncio
async def test_callback_errors_do_not_echo_secrets():
    a = binding()

    async def native(request):
        raise RuntimeError(a.secret)

    async with server([a], native) as (_, url):
        async with client(url, a) as t:
            await t.handshake()
            assert await t.send_outbound(action(a, content="hello")) == {
                "success": False, "error": "native action failed"}


@asynccontextmanager
async def raw_client(url, b):
    headers = {"Authorization": "Bearer " + make_upgrade_token(b.gateway_id, b.secret)}
    async with ClientSession() as http, http.ws_connect(url, headers=headers) as ws:
        await ws.send_json({"type": "hello", "platform": b.platform.value, "botId": b.bot_id})
        assert (await ws.receive_json(timeout=2))["type"] == "descriptor"
        yield ws


@pytest.mark.asyncio
async def test_restart_replays_fifo_only_to_owner_and_retains_ack_and_targets(tmp_path):
    a, b = binding(), binding("domain-b", "chat-b")
    path, calls = tmp_path / "queue.db", []

    async def native(request):
        calls.append(request)
        return SendResult(True, message_id="sent-by-" + request.gateway_id)

    async with server([a, b], native, store_path=path) as (c, url):
        c.enqueue(a.gateway_id, "acked", event(a, "inbound-acked"))
        got = asyncio.Queue()
        async with client(url, a, inbound=got.put) as t:
            await t.handshake()
            await asyncio.wait_for(got.get(), 2)
            assert (await t.send_outbound(action(a, content="before restart")))["success"]
        c.enqueue(a.gateway_id, "pending-a", event(a, "inbound-a"))
        c.enqueue(a.gateway_id, "pending-a2", event(a, "inbound-a2"))
        c.enqueue(b.gateway_id, "pending-b", event(b, "inbound-b"))
        # Put bytes on the real socket but deliberately do not ACK.
        async with raw_client(url, a) as ws:
            assert (await ws.receive_json(timeout=2))["bufferId"] == "pending-a"

    async with server([a, b], native, store_path=path) as (c, url):
        for delivery in ("acked", "pending-a"):
            with pytest.raises(ValueError, match="duplicate delivery"):
                c.enqueue(a.gateway_id, delivery, event(a))
        qa, qb = asyncio.Queue(), asyncio.Queue()
        async with client(url, a, inbound=qa.put) as ta, client(url, b, inbound=qb.put) as tb:
            await ta.handshake()
            await tb.handshake()
            assert (await asyncio.wait_for(qa.get(), 2)).message_id == "inbound-a"
            assert (await asyncio.wait_for(qa.get(), 2)).message_id == "inbound-a2"
            assert (await asyncio.wait_for(qb.get(), 2)).message_id == "inbound-b"
            for op, fields in [("send", {"content": "reply", "reply_to": "inbound-acked"}),
                               ("edit", {"content": "edit", "message_id": "sent-by-domain-a"}),
                               ("react", {"message_id": "inbound-a", "emoji": "yes"})]:
                assert (await ta.send_outbound(action(a, op, **fields)))["success"]
                count = len(calls)
                assert not (await tb.send_outbound(action(b, op, **fields)))["success"]
                assert len(calls) == count
            assert qa.empty() and qb.empty()
    # ACKs from the reopened server must themselves survive another restart.
    async with server([a, b], native, store_path=path) as (c, _):
        for domain, delivery in [(a, "pending-a"), (a, "pending-a2"), (b, "pending-b")]:
            with pytest.raises(ValueError, match="duplicate delivery"):
                c.enqueue(domain.gateway_id, delivery, event(domain))


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [{"chat_id": "other"}, {"thread_id": None},
    {"scope_id": "other"}, {"platform": Platform.DISCORD}, {"bot_id": "other"}])
async def test_reused_gateway_with_changed_binding_fails_closed(tmp_path, changed):
    from gateway.isolated_relay import ManagedRelayConnector
    a, path = binding(), tmp_path / "queue.db"

    async def native(request):
        return SendResult(True)

    async with server([a], native, store_path=path) as (c, _):
        c.enqueue(a.gateway_id, "pending", event(a))
    with pytest.raises(ValueError, match="binding"):
        ManagedRelayConnector([replace(a, **changed)], native, store_path=path)
    # Removing a domain temporarily does not forget its durable ownership.
    async with server([], native, store_path=path):
        pass
    with pytest.raises(ValueError, match="binding"):
        ManagedRelayConnector([replace(a, **changed)], native, store_path=path)
    async with server([replace(a, secret="rotated-test-secret")], native, store_path=path) as (c, url):
        q = asyncio.Queue()
        async with client(url, replace(a, secret="rotated-test-secret"), inbound=q.put) as t:
            await t.handshake()
            assert (await asyncio.wait_for(q.get(), 2)).source.chat_id == a.chat_id


@pytest.mark.asyncio
async def test_wrong_domain_ack_and_overflow_have_no_durable_side_effects(tmp_path):
    a, b = binding(), binding("domain-b", "chat-b")
    path, calls = tmp_path / "queue.db", []

    async def native(request):
        calls.append(request)
        return SendResult(True)

    async with server([a, b], native, store_path=path, max_pending=1) as (c, url):
        c.enqueue(a.gateway_id, "a-delivery", event(a, "a-target"))
        c.enqueue(b.gateway_id, "b-delivery", event(b, "b-target"))
        with pytest.raises(asyncio.QueueFull):
            c.enqueue(a.gateway_id, "overflow", event(a, "overflow-target"))
        async with raw_client(url, b) as ws:
            assert (await ws.receive_json(timeout=2))["bufferId"] == "b-delivery"
            await ws.send_json({"type": "inbound_ack", "bufferId": "a-delivery"})
            closed = await ws.receive(timeout=2)
            assert closed.type == WSMsgType.CLOSE and closed.data == 4401
    async with server([a, b], native, store_path=path, max_pending=1) as (c, url):
        with pytest.raises(asyncio.QueueFull):
            c.enqueue(a.gateway_id, "new", event(a))
        q = asyncio.Queue()
        async with client(url, a, inbound=q.put) as t:
            await t.handshake()
            assert (await asyncio.wait_for(q.get(), 2)).message_id == "a-target"
            assert not (await t.send_outbound(action(a, "react", message_id="overflow-target", emoji="x")))["success"]
            assert not calls
        async with raw_client(url, b) as ws:
            assert (await ws.receive_json(timeout=2))["bufferId"] == "b-delivery"


@pytest.mark.asyncio
async def test_serialized_scope_is_validated_and_caller_mutation_cannot_retarget(tmp_path):
    a, path = binding(), tmp_path / "queue.db"

    async def native(request):
        return SendResult(True)

    async with server([a], native, store_path=path) as (c, _):
        malicious = event(a)
        source = malicious.source.to_dict()
        source["chat_id"] = "foreign"
        malicious.source.to_dict = lambda: source
        with pytest.raises(ValueError, match="event rejected"):
            c.enqueue(a.gateway_id, "forged", malicious)
        original = event(a)
        c.enqueue(a.gateway_id, "valid", original)
        original.source.chat_id = "foreign"
        original.message_id = "foreign-target"
        original.text = "changed"
    async with server([a], native, store_path=path) as (c, url):
        q = asyncio.Queue()
        async with client(url, a, inbound=q.put) as t:
            await t.handshake()
            got = await asyncio.wait_for(q.get(), 2)
            assert (got.source.chat_id, got.message_id, got.text) == (a.chat_id, "message-1", "hello")
            assert not (await t.send_outbound(action(a, "react", message_id="foreign-target", emoji="x")))["success"]
            assert (await t.send_outbound(action(a, "react", message_id="message-1", emoji="x")))["success"]


@pytest.mark.asyncio
async def test_committed_enqueue_survives_abrupt_process_exit(tmp_path):
    a, path = binding(), tmp_path / "queue.db"
    script = '''
import os, sys
from gateway.isolated_relay import ManagedRelayConnector, RelayBinding
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
b = RelayBinding("domain-a", Platform.TELEGRAM, "chat-a", "topic-a", "test-bot", secret="child-test-secret")
c = ManagedRelayConnector([b], None, store_path=sys.argv[1])
c.enqueue(b.gateway_id, "child-delivery", MessageEvent(text="committed", message_id="child-message", source=SessionSource(platform=b.platform, chat_id=b.chat_id, thread_id=b.thread_id)))
os._exit(0)
'''
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert result.returncode == 0, result.stderr

    async def native(request):
        return SendResult(True)

    q = asyncio.Queue()
    async with server([a], native, store_path=path) as (_, url), client(url, a, inbound=q.put) as t:
        await t.handshake()
        got = await asyncio.wait_for(q.get(), 2)
        assert (got.message_id, got.text) == ("child-message", "committed")


@pytest.mark.asyncio
async def test_sqlite_failures_rollback_queue_authority_and_ack(tmp_path):
    a, path = binding(), tmp_path / "queue.db"

    async def native(request):
        return SendResult(True)

    async with server([a], native, store_path=path) as (c, url):
        with sqlite3.connect(path) as db:
            db.execute("CREATE TRIGGER fail_target BEFORE INSERT ON targets BEGIN SELECT RAISE(ABORT, 'fault'); END")
        with pytest.raises(sqlite3.Error):
            c.enqueue(a.gateway_id, "retryable", event(a))
        assert not c._ready[a.gateway_id].is_set()
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT COUNT(*) FROM pending").fetchone()[0] == 0
            assert db.execute("SELECT COUNT(*) FROM targets").fetchone()[0] == 0
            db.execute("DROP TRIGGER fail_target")
        c.enqueue(a.gateway_id, "retryable", event(a))
        with sqlite3.connect(path) as db:
            db.execute("CREATE TRIGGER fail_ack BEFORE INSERT ON acknowledgements BEGIN SELECT RAISE(ABORT, 'fault'); END")
        async with raw_client(url, a) as ws:
            assert (await ws.receive_json(timeout=2))["bufferId"] == "retryable"
            await ws.send_json({"type": "inbound_ack", "bufferId": "retryable"})
            assert (await ws.receive(timeout=2)).type == WSMsgType.CLOSE
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT delivery_id FROM pending").fetchall() == [("retryable",)]
            assert db.execute("SELECT COUNT(*) FROM acknowledgements").fetchone()[0] == 0
            db.execute("DROP TRIGGER fail_ack")
    async with server([a], native, store_path=path) as (c, url):
        q = asyncio.Queue()
        async with client(url, a, inbound=q.put) as t:
            await t.handshake()
            assert (await asyncio.wait_for(q.get(), 2)).message_id == "message-1"
            await t.send_outbound(action(a, "typing"))
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT COUNT(*) FROM pending").fetchone()[0] == 0
            assert db.execute("SELECT delivery_id FROM acknowledgements").fetchall() == [("retryable",)]


@pytest.mark.asyncio
async def test_history_limits_survive_restart_and_expired_targets_fail_closed(tmp_path):
    a, b = binding(), binding("domain-b", "chat-b")
    path = tmp_path / "queue.db"

    async def native(request):
        return SendResult(True)

    async with server([a, b], native, store_path=path, history_size=1) as (c, url):
        q = asyncio.Queue()
        async with client(url, a, inbound=q.put) as t:
            await t.handshake()
            for delivery in ("old", "new"):
                c.enqueue(a.gateway_id, delivery, event(a, delivery))
                await asyncio.wait_for(q.get(), 2)
                await t.send_outbound(action(a, "typing"))
        c.enqueue(b.gateway_id, "new", event(b, "new"))
    async with server([a, b], native, store_path=path, history_size=1) as (c, url):
        async with client(url, a) as t:
            await t.handshake()
            assert not (await t.send_outbound(action(a, "react", message_id="old", emoji="x")))["success"]
            assert (await t.send_outbound(action(a, "react", message_id="new", emoji="x")))["success"]
        with pytest.raises(ValueError, match="duplicate delivery"):
            c.enqueue(a.gateway_id, "new", event(a))
        c.enqueue(a.gateway_id, "old", event(a))
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT COUNT(*) FROM acknowledgements WHERE gateway_id=?", (a.gateway_id,)).fetchone()[0] == 1
            assert db.execute("SELECT COUNT(*) FROM targets WHERE gateway_id=?", (a.gateway_id,)).fetchone()[0] == 1
