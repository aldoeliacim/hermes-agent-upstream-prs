"""The credential broker performs inference only; it never runs an agent.

Fake upstream HTTP server, synthetic tokens and two trust domains. No provider
account or live service is contacted by these tests.
"""
import asyncio
import hashlib
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@asynccontextmanager
async def broker(headerless_body=None, credentials_override=None, entered=None, release=None):
    from gateway.isolated_inference import create_app

    calls = []
    policy = {
        hashlib.sha256(b"guest-a-token").hexdigest(): {"domain": "a", "models": ["test-model"]},
        hashlib.sha256(b"guest-b-token").hexdigest(): {"domain": "b", "models": ["test-model"]},
    }

    async def upstream(request):
        calls.append((dict(request.headers), await request.json()))
        if entered is not None:
            entered.put_nowait(True)
        if release is not None:
            await release.wait()
        if headerless_body is not None:
            class HeaderlessSSE(web.StreamResponse):
                async def _prepare_headers(self):
                    await super()._prepare_headers()
                    self.headers.pop("Content-Type", None)

            response = HeaderlessSSE()
            await response.prepare(request)
            await response.write(headerless_body)
            await response.write_eof()
            return response
        return web.Response(
            text='data: {"type":"response.completed"}\n\n',
            content_type="text/event-stream",
            headers={"X-Private-Account": "must-not-cross"},
        )

    app = web.Application()
    app.router.add_post("/codex/responses", upstream)
    async with TestServer(app) as server:
        credentials = lambda: {"api_key": "fake-upstream-only", "base_url": str(server.make_url("/codex"))}
        async with TestClient(TestServer(create_app(lambda: policy, credentials_override or credentials))) as client:
            yield client, calls, policy


@pytest.mark.asyncio
async def test_scoped_inference_keeps_credentials_and_account_headers_out_of_worker():
    async with broker() as (client, calls, _):
        response = await client.post("/v1/responses", json={"model": "test-model", "input": "hi", "store": True}, headers={"Authorization": "Bearer guest-a-token", "ChatGPT-Account-ID": "foreign", "session_id": "owner-session"})
        assert response.status == 200
        assert "response.completed" in await response.text()
        assert "X-Private-Account" not in response.headers
        headers, body = calls[0]
        assert headers["Authorization"] == "Bearer fake-upstream-only"
        assert headers.get("ChatGPT-Account-ID") != "foreign"
        assert headers["session_id"] != "owner-session"
        assert body["store"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("wire,status", [
    (b'event: response.completed\ndata: {"type":"response.completed"}\n\n', 200),
    (b'<html>not an inference response</html>', 502),
])
async def test_native_codex_headerless_stream_is_validated_before_forwarding(wire, status):
    async with broker(headerless_body=wire) as (client, _, _):
        response = await client.post("/v1/responses", json={"model": "test-model", "input": "hi", "stream": True}, headers={"Authorization": "Bearer guest-a-token"})
        assert response.status == status
        if status == 200:
            assert await response.read() == wire
            assert response.headers["Content-Type"] == "text/event-stream"


@pytest.mark.asyncio
async def test_equal_session_ids_are_partitioned_by_domain():
    async with broker() as (client, calls, _):
        for token in ("guest-a-token", "guest-b-token", "guest-a-token"):
            response = await client.post("/v1/responses", json={"model": "test-model", "input": "hi"}, headers={"Authorization": f"Bearer {token}", "session_id": "shared-name"})
            assert response.status == 200
            await response.read()
        assert calls[0][0]["session_id"] == calls[2][0]["session_id"]
        assert calls[0][0]["session_id"] != calls[1][0]["session_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"model": "ungranted-model", "input": "hi"},
    {"model": "test-model", "previous_response_id": "private-response"},
    {"model": "test-model", "conversation": "private-conversation"},
    {"model": "test-model", "tools": [{"type": "file_search", "vector_store_ids": ["private"]}]},
    {"model": "test-model", "tools": [{"type": "mcp", "server_url": "http://private"}]},
    {"model": "test-model", "input": [{"type": "item_reference", "id": "private-item"}]},
    {"model": "test-model", "input": [{"role": "user", "content": [{"type": "input_file", "file_id": "private-file"}]}]},
    {"model": "test-model", "background": True},
    {"model": "test-model", "input": "hi", "prompt": {"id": "pmpt_foreign", "variables": {"file": {"type": "input_file", "file_id": "foreign"}}}},
    {"model": "test-model", "input": "hi", "unknown_future_resource": "foreign"},
    {"model": "test-model", "input": [{"role": "user", "content": [{"type": "input_file", "file_url": "https://private.example/file", "file_data": ""}]}]},
    {"model": "test-model", "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://private.example/file"}]}]},
    {"model": "test-model", "input": [{"type": "function_call_output", "call_id": "call_1", "output": [{"type": "input_file", "file_id": "foreign"}]}]},
])
async def test_resource_references_and_ungranted_models_do_not_reach_upstream(payload):
    async with broker() as (client, calls, _):
        response = await client.post("/v1/responses", json=payload, headers={"Authorization": "Bearer guest-a-token"})
        assert response.status in {400, 403}
        assert not calls


@pytest.mark.asyncio
async def test_revocation_applies_to_existing_client_without_restart():
    async with broker() as (client, calls, policy):
        headers = {"Authorization": "Bearer guest-a-token"}
        payload = {"model": "test-model", "input": "hi"}
        response = await client.post("/v1/responses", json=payload, headers=headers)
        assert response.status == 200
        await response.read()
        policy.clear()
        response = await client.post("/v1/responses", json=payload, headers=headers)
        assert response.status == 401
        assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path,token", [
    ("post", "/v1/responses", "invalid"),
    ("get", "/v1/responses/private", "guest-a-token"),
    ("post", "/v1/chat/completions", "guest-a-token"),
    ("get", "/account", "guest-a-token"),
])
async def test_no_agent_admin_or_stored_response_surface(method, path, token):
    async with broker() as (client, calls, _):
        response = await getattr(client, method)(path, headers={"Authorization": f"Bearer {token}"})
        assert response.status in {401, 404, 405}
        assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("quota", [False, True])
async def test_auth_loader_failure_is_sanitized_and_classified(quota):
    from hermes_cli.auth import AuthError, CODEX_RATE_LIMITED_CODE

    def broken_credentials():
        raise AuthError("DO_NOT_ECHO_FAKE_SECRET", code=CODEX_RATE_LIMITED_CODE if quota else "refresh_failed")

    async with broker(credentials_override=broken_credentials) as (client, calls, _):
        response = await client.post("/v1/responses", json={"model": "test-model", "input": "hi"}, headers={"Authorization": "Bearer guest-a-token"})
        assert response.status == (429 if quota else 503)
        assert "DO_NOT_ECHO" not in await response.text()
        assert not calls


@pytest.mark.asyncio
async def test_admission_limits_apply_before_queueing_upstream_and_release_after_finish():
    entered = asyncio.Queue()
    release = asyncio.Event()
    async with broker(entered=entered, release=release) as (client, calls, policy):
        policy[hashlib.sha256(b"guest-c-token").hexdigest()] = {"domain": "c", "models": ["test-model"]}
        payload = {"model": "test-model", "input": "hi"}

        async def post(token):
            return await client.post("/v1/responses", json=payload, headers={"Authorization": "Bearer " + token})

        first = asyncio.create_task(post("guest-a-token"))
        try:
            await asyncio.wait_for(entered.get(), 2)
            duplicate = await asyncio.wait_for(post("guest-a-token"), 1)
            assert duplicate.status == 429
            second = asyncio.create_task(post("guest-b-token"))
            try:
                await asyncio.wait_for(entered.get(), 2)
                saturated = await asyncio.wait_for(post("guest-c-token"), 1)
                assert saturated.status == 429
                assert len(calls) == 2
            finally:
                release.set()
                await second
        finally:
            release.set()
            await first
        again = await post("guest-a-token")
        assert again.status == 200


@pytest.mark.asyncio
async def test_hourly_request_budget_is_per_domain_not_one_budget_for_all_contacts():
    async with broker() as (client, calls, _):
        payload = {"model": "test-model", "input": "hi"}
        for _ in range(120):
            response = await client.post("/v1/responses", json=payload, headers={"Authorization": "Bearer guest-a-token"})
            assert response.status == 200
            await response.read()
        response = await client.post("/v1/responses", json=payload, headers={"Authorization": "Bearer guest-a-token"})
        assert response.status == 429
        other = await client.post("/v1/responses", json=payload, headers={"Authorization": "Bearer guest-b-token"})
        assert other.status == 200
        assert len(calls) == 121
