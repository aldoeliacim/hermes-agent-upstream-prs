"""Admitted response lifecycle over real aiohttp loopback, with controlled I/O stalls."""
import asyncio

import aiohttp
import pytest
from aiohttp import web

from tests.gateway.test_isolated_inference_limits import PAYLOAD, headers, limited_broker


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "validation", "credential", "upstream", "unknown", "timeout"])
@pytest.mark.parametrize("stage", ["prepare", "eof"])
async def test_every_outcome_finalization_has_a_deadline(monkeypatch, outcome, stage):
    entered = asyncio.Event()
    ended = asyncio.Event()
    release = asyncio.Event()
    release_upstream = asyncio.Event()
    responses: list[web.StreamResponse] = []
    statuses = []
    original_prepare = web.StreamResponse.prepare
    original_eof = web.StreamResponse.write_eof

    async def stall():
        entered.set()
        try:
            await release.wait()
        finally:
            ended.set()

    async def prepare(self, request):
        if request.path == "/v1/responses" and request.headers.get("session_id") == "stall":
            responses.append(self)
            statuses.append(self.status)
            if stage == "prepare":
                await stall()
        return await original_prepare(self, request)

    async def eof(self, data=b""):
        if any(self is response for response in responses) and stage == "eof":
            await stall()
        return await original_eof(self, data)

    def loader(credentials):
        if outcome == "credential":
            from hermes_cli.auth import AuthError
            raise AuthError("synthetic credential failure")
        if outcome == "unknown":
            raise RuntimeError("synthetic private detail")
        return credentials

    async def upstream(request):
        if outcome == "timeout" and (await request.json())["input"] == "stall":
            await release_upstream.wait()
        return web.json_response({"ok": True}, status=500 if outcome == "upstream" else 200)

    monkeypatch.setattr(web.StreamResponse, "prepare", prepare)
    monkeypatch.setattr(web.StreamResponse, "write_eof", eof)
    async with limited_broker(options={"request_timeout_seconds": 0.15}, loader=loader,
                              upstream=upstream) as (client, _, _):
        payload: dict[str, object] = {**PAYLOAD, "input": "stall"}
        if outcome == "validation":
            payload["max_output_tokens"] = 1

        async def consume():
            response = await client.post("/v1/responses", json=payload,
                                         headers={**headers(), "session_id": "stall"})
            await response.read()

        pending = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            # A client seeing Content-Length bytes alone is not EOF completion:
            # require the actual stalled server finalizer to terminate too.
            await asyncio.wait_for(ended.wait(), 2)
            try:
                await asyncio.wait_for(asyncio.shield(pending), 2)
            except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError):
                pass
            assert statuses[0] == {
                "success": 200, "validation": 400, "credential": 503,
                "upstream": 502, "unknown": 502, "timeout": 504,
            }[outcome]
        finally:
            release.set()
            release_upstream.set()
            await asyncio.gather(pending, return_exceptions=True)
        # Stop injecting failures, then prove same-domain normal use survives.
        outcome = "success"
        healthy = await client.post("/v1/responses", json=PAYLOAD, headers=headers())
        assert healthy.status == 200
        assert await healthy.json() == {"ok": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["prepare", "eof"])
@pytest.mark.parametrize("exit_kind", ["complete", "cancel", "failure"])
async def test_error_delivery_owns_admission_until_completion_or_abort(monkeypatch, stage, exit_kind):
    entered = asyncio.Queue()
    release = asyncio.Event()
    transports = []
    handlers = asyncio.Queue()
    responses: list[web.StreamResponse] = []
    original_prepare = web.StreamResponse.prepare
    original_eof = web.StreamResponse.write_eof

    async def intercept():
        entered.put_nowait(True)
        await release.wait()
        if exit_kind == "failure":
            raise RuntimeError("synthetic transport failure")

    async def prepare(self, request):
        if request.path == "/v1/responses" and self.status == 400:
            responses.append(self)
            transports.append(request.transport)
            if stage == "prepare":
                await intercept()
        return await original_prepare(self, request)

    async def eof(self, data=b""):
        if any(self is response for response in responses) and stage == "eof":
            await intercept()
        return await original_eof(self, data)

    monkeypatch.setattr(web.StreamResponse, "prepare", prepare)
    monkeypatch.setattr(web.StreamResponse, "write_eof", eof)
    async with limited_broker(options={"request_timeout_seconds": 10},
                              handlers=handlers) as (client, calls, _):
        pending = []
        server_tasks = []

        async def consume(domain):
            response = await client.post("/v1/responses", json={**PAYLOAD, "max_output_tokens": 1},
                                         headers=headers(domain))
            assert response.status == 400
            await response.read()

        async def post(domain, status):
            response = await client.post("/v1/responses", json=PAYLOAD, headers=headers(domain))
            await handlers.get()
            assert response.status == status
            await response.read()

        try:
            pending.append(asyncio.create_task(consume("a")))
            server_tasks.append(await asyncio.wait_for(handlers.get(), 2))
            await asyncio.wait_for(entered.get(), 2)
            await post("a", 429)
            await post("b", 200)  # Other domain still has independent capacity.
            pending.append(asyncio.create_task(consume("b")))
            server_tasks.append(await asyncio.wait_for(handlers.get(), 2))
            await asyncio.wait_for(entered.get(), 2)
            await post("c", 429)  # Two finalizers retain both global slots.
            assert len(calls) == 1
            if exit_kind == "cancel":
                for task in server_tasks:
                    task.cancel()
            else:
                release.set()
            await asyncio.wait_for(asyncio.gather(*server_tasks, return_exceptions=True), 2)
            if exit_kind != "complete":
                assert all(transport.is_closing() for transport in transports)
            results = await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 2)
            if exit_kind == "complete":
                assert results == [None, None]
            await post("a", 200)
            await post("b", 200)
            assert len(calls) == 3
        finally:
            release.set()
            for task in server_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*server_tasks, *pending, return_exceptions=True)
