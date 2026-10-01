"""Broker budgets over real loopback HTTP; no provider or real auth store."""
import asyncio
import hashlib
import threading
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


PAYLOAD = {"model": "test-model", "input": "hi"}


def headers(domain="a"):
    return {"Authorization": f"Bearer guest-{domain}"}


@asynccontextmanager
async def limited_broker(*, options=None, loader=None, upstream=None, handlers=None):
    from gateway.isolated_inference import BrokerLimits, create_app

    calls = []
    policy = {
        hashlib.sha256(f"guest-{domain}".encode()).hexdigest(): {
            "domain": domain, "models": ["test-model"],
        } for domain in "abcd"
    }

    async def record(request):
        calls.append(await request.json())
        if upstream:
            return await upstream(request)
        return web.json_response({"output": [{"type": "function_call", "arguments": "{}"}]})

    app = web.Application()
    app.router.add_post("/responses", record)
    async with TestServer(app) as server:
        credentials = {"api_key": "synthetic-only", "base_url": str(server.make_url(""))}
        broker = create_app(
            lambda: policy, (lambda: loader(credentials)) if loader else (lambda: credentials),
            limits=BrokerLimits(**(options or {})),
        )
        if handlers is not None:
            @web.middleware
            async def track(request, handler):
                handlers.put_nowait(asyncio.current_task())
                return await handler(request)
            broker.middlewares.append(track)
        async with TestClient(TestServer(broker, shutdown_timeout=0.1)) as client:
            yield client, calls, policy


@pytest.mark.parametrize("options", [
    {"request_timeout_seconds": 0}, {"request_timeout_seconds": float("inf")},
    {"request_timeout_seconds": float("nan")}, {"request_timeout_seconds": True},
    {"credential_timeout_seconds": -1}, {"credential_timeout_seconds": "10"},
    {"max_response_bytes": 0}, {"max_response_bytes": True}, {"max_response_bytes": 1.5},
    {"default_output_tokens": 0}, {"max_output_tokens": None},
    {"default_output_tokens": 9, "max_output_tokens": 8},
    {"output_mode": "auto"}, {"output_mode": None},
    {"max_credential_loads": 0}, {"max_credential_loads": 3},
])
def test_operator_limits_fail_closed(options):
    from gateway.isolated_inference import BrokerLimits

    with pytest.raises(ValueError):
        BrokerLimits(**options)


@pytest.mark.asyncio
async def test_valid_operator_limits_keep_normal_json_tool_output():
    async with limited_broker(options={"output_mode": "responses", "default_output_tokens": 8,
                                       "max_output_tokens": 16}) as (client, calls, _):
        response = await client.post("/v1/responses", json=PAYLOAD, headers=headers())
        assert response.status == 200
        assert (await response.json())["output"][0]["type"] == "function_call"
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_stalled_body_expires_and_releases_domain_without_loading_credentials():
    release = asyncio.Event()
    sent = asyncio.Event()
    loaded = []

    async def body():
        yield b'{'
        sent.set()
        await release.wait()
        yield b'}'

    async with limited_broker(options={"request_timeout_seconds": 0.2},
                              loader=lambda credentials: loaded.append(True) or credentials) as (client, calls, _):
        pending = asyncio.create_task(client.post("/v1/responses", data=body(), headers=headers()))
        try:
            await asyncio.wait_for(sent.wait(), 2)
            duplicate = await client.post("/v1/responses", json=PAYLOAD, headers=headers())
            assert duplicate.status == 429
            response = await asyncio.wait_for(asyncio.shield(pending), 3)
            assert response.status == 504
            assert not loaded and not calls
            valid = await client.post("/v1/responses", json=PAYLOAD, headers=headers())
            assert valid.status == 200
            await valid.read()
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [
    {"request_timeout_seconds": 0.2, "credential_timeout_seconds": 10},
    {"request_timeout_seconds": 10, "credential_timeout_seconds": 0.2},
])
async def test_loader_timeout_keeps_outstanding_work_bounded_until_threads_finish(options):
    loop = asyncio.get_running_loop()
    entered = asyncio.Queue()
    release = threading.Event()
    started = []

    def loader(credentials):
        started.append(True)
        loop.call_soon_threadsafe(entered.put_nowait, True)
        assert release.wait(15), "test did not release synthetic loader"
        return credentials

    async with limited_broker(options=options, loader=loader) as (client, calls, _):
        pending = []
        try:
            for domain in "ab":
                pending.append(asyncio.create_task(client.post("/v1/responses", json=PAYLOAD,
                                                               headers=headers(domain))))
                await asyncio.wait_for(entered.get(), 2)
            for task in pending:
                response = await asyncio.wait_for(asyncio.shield(task), 3)
                assert response.status == 504
            for domain in "abcd" * 3:
                response = await asyncio.wait_for(client.post("/v1/responses", json=PAYLOAD,
                                                              headers=headers(domain)), 2)
                assert response.status == 429
            assert len(started) == 2
            assert not calls
        finally:
            release.set()
            await asyncio.gather(*pending, return_exceptions=True)
        # Completion of actual threads, not cancellation of awaiters, restores capacity.
        async with asyncio.timeout(3):
            while True:
                response = await client.post("/v1/responses", json=PAYLOAD, headers=headers("c"))
                if response.status != 429:
                    break
                await asyncio.sleep(0.02)
        assert response.status == 200
        await response.read()
        assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, 0, -1, True, 1.5, "8", 17])
async def test_invalid_or_excess_responses_token_budget_never_loads_credentials(budget):
    loaded = []
    async with limited_broker(options={"output_mode": "responses", "default_output_tokens": 8,
                                       "max_output_tokens": 16},
                              loader=lambda credentials: loaded.append(True) or credentials) as (client, calls, _):
        response = await client.post("/v1/responses", json={**PAYLOAD, "max_output_tokens": budget},
                                     headers=headers())
        assert response.status == 400
        assert not calls and not loaded


@pytest.mark.asyncio
@pytest.mark.parametrize("supplied", [{}, {"max_output_tokens": 1}, {"max_output_tokens": 16}])
async def test_responses_token_policy_injects_missing_budget_and_preserves_valid_budget(supplied):
    async with limited_broker(options={"output_mode": "responses", "default_output_tokens": 8,
                                       "max_output_tokens": 16}) as (client, calls, _):
        response = await client.post("/v1/responses", json={**PAYLOAD, **supplied}, headers=headers())
        assert response.status == 200
        await response.read()
        assert calls[0]["max_output_tokens"] == supplied.get("max_output_tokens", 8)


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, 8, 999999])
async def test_codex_does_not_claim_unsupported_token_enforcement(budget):
    async with limited_broker() as (client, calls, _):
        response = await client.post("/v1/responses", json={**PAYLOAD, "max_output_tokens": budget},
                                     headers=headers())
        assert response.status == 400
        assert not calls
        response = await client.post("/v1/responses", json=PAYLOAD, headers=headers())
        assert response.status == 200
        await response.read()
        assert "max_output_tokens" not in calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["application/json", "text/event-stream", "headerless"])
@pytest.mark.parametrize("excess", [False, True])
async def test_aggregate_response_byte_limit_includes_prefix_and_tool_stream(kind, excess):
    if kind == "application/json":
        first = b'{"output":[{"type":"function_call","arguments":"'
        rest = b'{}"}]}'
    else:
        first = b'event: response.function_call_arguments.delta\n'
        rest = b'data: {"type":"response.function_call_arguments.delta","delta":"{}"}\n\n'
    wire = first + rest
    cap = len(wire) - (1 if excess else 0)

    async def upstream(request):
        class Headerless(web.StreamResponse):
            async def _prepare_headers(self):
                await super()._prepare_headers()
                self.headers.pop("Content-Type", None)

        response = Headerless() if kind == "headerless" else web.StreamResponse(
            headers={"Content-Type": kind})
        await response.prepare(request)
        await response.write(first)
        await asyncio.sleep(0.01)
        await response.write(rest)
        return response

    async with limited_broker(options={"max_response_bytes": cap}, upstream=upstream) as (client, calls, _):
        try:
            response = await client.post("/v1/responses", json={**PAYLOAD, "stream": True}, headers=headers())
            if response.status != 200:
                assert excess and response.status == 502
            elif excess:
                with pytest.raises(aiohttp.ClientPayloadError):
                    await response.read()
            else:
                assert await response.read() == wire
        except aiohttp.ServerDisconnectedError:
            assert excess
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_deadline_aborts_prepared_stream_without_forged_success():
    release = asyncio.Event()

    async def upstream(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"type":"response.output_text.delta"}\n\n')
        await release.wait()
        return response

    async with limited_broker(options={"request_timeout_seconds": 0.2}, upstream=upstream) as (client, _, _):
        try:
            response = await client.post("/v1/responses", json=PAYLOAD, headers=headers())
            with pytest.raises(aiohttp.ClientPayloadError):
                await asyncio.wait_for(response.read(), 3)
        finally:
            release.set()


@pytest.mark.asyncio
async def test_handler_cancellation_cannot_recycle_running_credential_work():
    handlers = asyncio.Queue()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    release = threading.Event()
    started = []

    def loader(credentials):
        started.append(True)
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(15)
        return credentials

    async with limited_broker(options={"max_credential_loads": 1}, loader=loader,
                              handlers=handlers) as (client, calls, _):
        pending = asyncio.create_task(client.post("/v1/responses", json=PAYLOAD, headers=headers()))
        try:
            handler = await asyncio.wait_for(handlers.get(), 2)
            await asyncio.wait_for(entered.wait(), 2)
            handler.cancel()
            await asyncio.gather(handler, return_exceptions=True)
            for domain in "abcd" * 3:
                response = await asyncio.wait_for(client.post("/v1/responses", json=PAYLOAD,
                                                              headers=headers(domain)), 2)
                assert response.status == 429
            assert len(started) == 1
            assert not calls
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_slow_reader_backpressure_is_inside_whole_request_deadline():
    stopped = asyncio.Event()
    writes = []

    async def upstream(request):
        if (await request.json())["input"] != "flood":
            return web.json_response({"ok": True})
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        try:
            # No upstream stall: only the non-reading downstream stops progress.
            for _ in range(8192):
                await response.write(b":" + b"x" * 65533 + b"\n\n")
                writes.append(True)
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            stopped.set()
        return response

    async with limited_broker(options={"request_timeout_seconds": 0.4,
                                       "max_response_bytes": 1024 * 1024 * 1024},
                              upstream=upstream) as (client, calls, _):
        url = client.make_url("/v1/responses")
        reader, writer = await asyncio.open_connection(url.host, url.port, limit=1024)
        data = b'{"model":"test-model","input":"flood"}'
        writer.write(b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n"
                     b"Authorization: Bearer guest-a\r\nContent-Type: application/json\r\n"
                     + f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
        await writer.drain()
        try:
            # Keep the reader alive but do not consume headers or response bytes.
            await asyncio.wait_for(stopped.wait(), 3)
            assert 0 < len(writes) < 8192  # upstream terminated, not exhausted
            response = await client.post("/v1/responses", json=PAYLOAD, headers=headers())
            assert response.status == 200
            assert await response.json() == {"ok": True}
            assert len(calls) == 2
        finally:
            writer.close()
            await writer.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["body", "credentials"])
async def test_grant_revocation_while_waiting_never_reaches_upstream(stage):
    entered = asyncio.Event()
    release_body = asyncio.Event()
    release_loader = threading.Event()
    loop = asyncio.get_running_loop()

    def loader(credentials):
        if stage == "credentials":
            loop.call_soon_threadsafe(entered.set)
            assert release_loader.wait(15)
        return credentials

    async def body():
        yield b'{"model":"test-model",'
        entered.set()
        await release_body.wait()
        yield b'"input":"hi"}'

    async with limited_broker(loader=loader) as (client, calls, policy):
        if stage == "body":
            pending = asyncio.create_task(client.post("/v1/responses", headers=headers(), data=body()))
        else:
            pending = asyncio.create_task(client.post("/v1/responses", headers=headers(), json=PAYLOAD))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            del policy[hashlib.sha256(b"guest-a").hexdigest()]
            release_body.set()
            release_loader.set()
            response = await asyncio.wait_for(pending, 3)
            assert response.status == 401
            assert not calls
            response = await client.post("/v1/responses", json=PAYLOAD, headers=headers("b"))
            assert response.status == 200
        finally:
            release_body.set()
            release_loader.set()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_compressed_response_is_rejected_without_inflation_or_forwarding():
    import gzip

    async def upstream(request):
        return web.Response(body=gzip.compress(b'x' * 100000), content_type="application/json",
                            headers={"Content-Encoding": "gzip"})

    async with limited_broker(options={"max_response_bytes": 1000}, upstream=upstream) as (client, calls, _):
        response = await client.post("/v1/responses", json=PAYLOAD, headers=headers())
        assert response.status == 502
        assert await response.text() == "Invalid inference response"
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_operator_limits_file_is_used_by_broker_entrypoint(tmp_path, monkeypatch):
    import json
    import sys
    from gateway.isolated_inference import main

    policy = tmp_path / "policy.json"
    limits = tmp_path / "limits.json"
    policy.write_text(json.dumps({hashlib.sha256(b"guest-a").hexdigest(): {
        "domain": "a", "models": ["test-model"],
    }}), encoding="utf-8")
    limits.write_text(json.dumps({"max_response_bytes": 5}), encoding="utf-8")
    apps = []
    monkeypatch.setattr(sys, "argv", ["broker", "--policy", str(policy), "--port", "12345",
                                     "--limits", str(limits)])
    # Intercept only the blocking CLI server launcher; exercise that exact app
    # via real loopback HTTP below, with an explicitly synthetic credential loader.
    monkeypatch.setattr(web, "run_app", lambda app, **kwargs: apps.append(app))
    upstream = web.Application()
    async def respond(request):
        return web.Response(body=b"0123456789", content_type="application/json")
    upstream.router.add_post("/responses", respond)
    async with TestServer(upstream) as server:
        monkeypatch.setattr("hermes_cli.auth.resolve_codex_runtime_credentials", lambda: {
            "api_key": "synthetic-only", "base_url": str(server.make_url("")),
        })
        main()
        async with TestClient(TestServer(apps[0])) as client:
            response = await client.post("/v1/responses", json=PAYLOAD, headers=headers())
            with pytest.raises(aiohttp.ClientPayloadError):
                await response.read()


@pytest.mark.asyncio
async def test_upstream_headers_stall_times_out_then_domain_can_retry():
    release = asyncio.Event()
    async def upstream(request):
        if (await request.json())["input"] == "stall":
            await release.wait()
        return web.json_response({"ok": True})

    async with limited_broker(options={"request_timeout_seconds": 0.2}, upstream=upstream) as (client, calls, _):
        try:
            response = await asyncio.wait_for(client.post("/v1/responses", json={**PAYLOAD, "input": "stall"},
                                                         headers=headers()), 3)
            assert response.status == 504
            assert await response.text() == "Inference deadline exceeded"
            response = await client.post("/v1/responses", json=PAYLOAD, headers=headers())
            assert response.status == 200
            await response.read()
        finally:
            release.set()


@pytest.mark.asyncio
async def test_redirect_never_replays_body_or_credentials():
    targets = []
    app = web.Application()
    async def target(request):
        targets.append(True)
        return web.json_response({"ok": True})
    app.router.add_post("/stolen", target)
    async with TestServer(app) as target_server:
        async def redirect(request):
            return web.Response(status=307, headers={"Location": str(target_server.make_url("/stolen"))})
        async with limited_broker(upstream=redirect) as (client, calls, _):
            response = await client.post("/v1/responses", json=PAYLOAD, headers=headers())
            assert response.status == 502
            assert await response.text() == "Inference temporarily unavailable"
            assert len(calls) == 1 and not targets


@pytest.mark.asyncio
async def test_stalled_raw_upload_does_not_linger_after_deadline_response():
    loaded = []
    async with limited_broker(options={"request_timeout_seconds": 0.2},
                              loader=lambda credentials: loaded.append(True) or credentials) as (client, calls, _):
        url = client.make_url("/v1/responses")
        reader, writer = await asyncio.open_connection(url.host, url.port)
        writer.write(b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n"
                     b"Authorization: Bearer guest-a\r\nContent-Type: application/json\r\n"
                     b"Content-Length: 100\r\n\r\n{")
        await writer.drain()
        try:
            # Unlike aiohttp's cooperative upload iterator, this client never
            # finishes/cancels its upload upon seeing the error response.
            wire = await asyncio.wait_for(reader.read(), 3)
            assert b"504 Gateway Timeout" in wire
            assert b"Inference deadline exceeded" in wire
            assert not loaded and not calls
        finally:
            writer.close()
            await writer.wait_closed()
