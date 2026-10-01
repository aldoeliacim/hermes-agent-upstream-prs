"""Inference-only credential boundary for isolated agent processes.

Workers receive independently revocable, model-scoped tokens, never the
operator's OAuth credentials. This process has no agent loop, tool dispatch,
workspace, artifact, account-inspection or stored-response endpoints. Its
only upstream operation is a non-persistent Responses request.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from aiohttp import web

from agent.codex_headers import codex_cloudflare_headers

_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_CLIENT = web.AppKey("inference_http_client", aiohttp.ClientSession)


@dataclass(frozen=True)
class BrokerLimits:
    """Trusted, startup-only policy; never inferred from worker body or URL.

    Defaults are explicit safety ceilings, not measured service capacity:
    600s whole request, 30s waiting for credentials, two outstanding loaders,
    8 MiB aggregate response body (including SSE framing and tool arguments).
    ``responses`` supplies a missing token budget (4096) and rejects budgets
    above 16384. Choose it ONLY for a trusted endpoint supporting that field.
    Default ``codex`` rejects explicit max_output_tokens rather than claiming
    to enforce it: native agent/transports/codex.py omits this parameter.
    Byte/time limits are NOT provider token, reasoning, or billing guarantees;
    disconnecting cannot guarantee that remote generation stops. After the
    request deadline, an unstarted 504 gets at most min(100ms, request timeout)
    to finish sending; otherwise the transport is aborted. Admission is held
    through normal/error response EOF or abort, including that final send.

    Embedders pass ``limits=BrokerLimits(...)`` to create_app. The CLI accepts
    ``--limits /operator/limits.json``, e.g. {"request_timeout_seconds": 300,
    "credential_timeout_seconds": 20, "max_response_bytes": 4194304}.
    Unknown fields or invalid values prevent startup. Missing fields retain
    defaults. Grant revocation stays live; changing limits requires rebuilding
    the app. Keep codex mode with the CLI's native Codex credential resolver.
    Compressed upstream responses are rejected (identity encoding requested).
    """

    request_timeout_seconds: float = 600
    credential_timeout_seconds: float = 30
    max_credential_loads: int = 2
    max_response_bytes: int = 8 * 1024 * 1024
    output_mode: str = "codex"
    default_output_tokens: int = 4096
    max_output_tokens: int = 16384

    def __post_init__(self):
        for value in (self.request_timeout_seconds, self.credential_timeout_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Broker timeouts must be finite positive numbers")
        for value in (self.max_credential_loads, self.max_response_bytes,
                      self.default_output_tokens, self.max_output_tokens):
            if type(value) is not int or value <= 0:
                raise ValueError("Broker budgets must be positive integers")
        if self.max_credential_loads > 2 or self.default_output_tokens > self.max_output_tokens:
            raise ValueError("Invalid broker budget range")
        if self.output_mode not in ("codex", "responses"):
            raise ValueError("Unknown broker output mode")


def _grant(policy_loader: Callable, authorization: str) -> dict:
    prefix = "Bearer "
    if not authorization.startswith(prefix) or len(authorization) > 4096:
        raise web.HTTPUnauthorized(text="Invalid inference credential")
    token = authorization[len(prefix):]
    if not token or token != token.strip():
        raise web.HTTPUnauthorized(text="Invalid inference credential")
    try:
        digest = hashlib.sha256(token.encode()).hexdigest()
        policy = policy_loader()
        grant = policy.get(digest) if isinstance(policy, dict) else None
        if (
            not isinstance(grant, dict)
            or not isinstance(grant.get("domain"), str)
            or not grant["domain"]
            or not isinstance(grant.get("models"), list)
            or not grant["models"]
            or not all(isinstance(model, str) and model for model in grant["models"])
        ):
            raise ValueError("Invalid grant")
        return {"domain": grant["domain"], "models": tuple(grant["models"])}
    except Exception:
        raise web.HTTPUnauthorized(text="Invalid inference credential") from None


def _validate_inline_parts(content: object) -> None:
    if isinstance(content, str):
        return
    if not isinstance(content, list):
        raise web.HTTPBadRequest(text="Invalid message content")
    fields = {
        "input_text": {"type", "text"}, "output_text": {"type", "text", "annotations", "logprobs"},
        "input_image": {"type", "image_url", "detail"},
        "input_file": {"type", "filename", "file_data"}, "refusal": {"type", "refusal"},
    }
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in fields:
            raise web.HTTPBadRequest(text="Invalid message content")
        if set(part) - fields[part["type"]]:
            raise web.HTTPForbidden(text="Only inline content is granted")
        if part["type"] == "input_file":
            if not isinstance(part.get("file_data"), str) or not part["file_data"]:
                raise web.HTTPForbidden(text="Only inline file content is granted")
        if part["type"] == "input_image":
            image = part.get("image_url")
            if not isinstance(image, str) or not image.startswith("data:image/"):
                raise web.HTTPForbidden(text="Only inline image content is granted")


def _validate_request(body: object, grant: dict) -> dict:
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(text="Expected a Responses object")
    # No passthrough for newly introduced resource-bearing fields (e.g. stored
    # prompt templates). Extend this closed wire contract only with tests.
    allowed_fields = {
        "model", "input", "instructions", "tools", "tool_choice", "parallel_tool_calls",
        "stream", "store", "background", "previous_response_id", "conversation",
        "max_output_tokens", "max_tool_calls", "temperature", "top_p", "text", "reasoning",
        "include", "metadata", "user", "safety_identifier", "prompt_cache_key",
        "prompt_cache_retention", "service_tier", "truncation", "top_logprobs", "stream_options",
    }
    if set(body) - allowed_fields:
        raise web.HTTPForbidden(text="Inference field is not granted")
    if body.get("model") not in grant["models"]:
        raise web.HTTPForbidden(text="Model is not granted")
    if any(body.get(field) is not None for field in ("previous_response_id", "conversation")):
        raise web.HTTPForbidden(text="Stored response access is not granted")
    if body.get("background"):
        raise web.HTTPForbidden(text="Background provider execution is not granted")
    tools = body.get("tools", [])
    if not isinstance(tools, list) or any(
        not isinstance(tool, dict) or tool.get("type") != "function" for tool in tools
    ):
        raise web.HTTPForbidden(text="Hosted resource tools are not granted")
    input_items = body.get("input")
    if not isinstance(input_items, (str, list)):
        raise web.HTTPBadRequest(text="Expected explicit conversation input")
    if isinstance(input_items, list):
        for item in input_items:
            if not isinstance(item, dict) or item.get("type", "message") not in {
                "message", "function_call", "function_call_output", "reasoning",
            }:
                raise web.HTTPForbidden(text="Stored input references are not granted")
            if item.get("type", "message") == "message":
                _validate_inline_parts(item.get("content", []))
            elif item.get("type") == "function_call_output":
                _validate_inline_parts(item.get("output"))
    # Never create provider-side state under the operator's account.
    return {**body, "store": False}


def create_app(
    policy_loader: Callable, credentials_loader: Callable, *, limits: BrokerLimits | None = None,
) -> web.Application:
    """Build a broker with injectable trusted loaders; no worker-chosen upstream.

    ``policy_loader`` returns SHA-256(token) -> {domain, models}. It is reread
    on every request, including after body parsing, so revocation does not
    wait for a connection, process, or session cache to expire. It MUST be a
    fast, nonblocking trusted lookup: asyncio deadlines cannot preempt Python
    blocking the event loop. Credential I/O runs in a dedicated bounded pool;
    cancellation bounds the HTTP wait, not the lifetime of trusted OS threads.
    Limits cover admitted HTTP handlers, not incomplete headers/idle sockets
    before aiohttp dispatch. Front-door connection/header limits remain the
    embedding server or reverse proxy's responsibility.
    """
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error

    limits = limits if limits is not None else BrokerLimits()
    if not isinstance(limits, BrokerLimits):
        raise ValueError("Expected validated BrokerLimits")
    # aiohttp otherwise drains an unfinished upload after the handler returns,
    # outside our deadline (even for a force_close response).
    app = web.Application(client_max_size=_MAX_REQUEST_BYTES, handler_args={"lingering_time": 0})
    # Count actual executor futures, NOT cancellable asyncio waiters. No work
    # is submitted beyond the pool budget; a timed-out domain cannot submit again
    # until its previous trusted loader has really returned.
    loading: dict[str, Future] = {}
    executor = ThreadPoolExecutor(max_workers=limits.max_credential_loads,
                                  thread_name_prefix="inference-credentials")
    active_domains: set[str] = set()
    attempts: dict[str, deque] = {}

    async def client_lifecycle(application):
        timeout = aiohttp.ClientTimeout(total=600, connect=20, sock_read=180)
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=False, auto_decompress=False,
                headers={"Accept-Encoding": "identity"},
            ) as client:
                application[_CLIENT] = client
                yield
        finally:
            # Python cannot kill a running trusted-loader thread. Do not block
            # the event loop on shutdown; stuck threads can still delay process
            # exit and require operator intervention. Bound work, not its OS
            # lifetime. The loader must bound its own I/O and not spawn work.
            executor.shutdown(wait=False, cancel_futures=True)

    async def load_credentials(domain):
        if domain in loading or len(loading) >= limits.max_credential_loads:
            raise web.HTTPTooManyRequests(text="Inference capacity is busy")
        loop = asyncio.get_running_loop()
        future = executor.submit(credentials_loader)
        loading[domain] = future

        def retire(_):
            def remove():
                if loading.get(domain) is future:
                    del loading[domain]
            try:
                loop.call_soon_threadsafe(remove)
            except RuntimeError:  # Loop already closed during broker shutdown.
                pass

        future.add_done_callback(retire)
        waiter = asyncio.wrap_future(future)
        # Retrieve late exceptions without logging trusted-loader error text.
        waiter.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        async with asyncio.timeout(limits.credential_timeout_seconds):
            return await asyncio.shield(waiter)

    app.cleanup_ctx.append(client_lifecycle)

    async def responses(request: web.Request, domain: str) -> web.StreamResponse:
        authorization = request.headers.get("Authorization", "")
        _grant(policy_loader, authorization)
        if request.query:
            raise web.HTTPBadRequest(text="Query parameters are not accepted")
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeError):
            raise web.HTTPBadRequest(text="Invalid JSON") from None
        grant = _grant(policy_loader, authorization)
        if grant["domain"] != domain:
            raise web.HTTPUnauthorized(text="Inference grant changed")
        body = _validate_request(body, grant)
        if limits.output_mode == "codex":
            if "max_output_tokens" in body:
                raise web.HTTPBadRequest(text="Output token parameter is not supported")
        else:
            budget = body.get("max_output_tokens", limits.default_output_tokens)
            if type(budget) is not int or not 0 < budget <= limits.max_output_tokens:
                raise web.HTTPBadRequest(text="Invalid inference output budget")
            body["max_output_tokens"] = budget
        try:
            credentials = await load_credentials(domain)
            grant = _grant(policy_loader, authorization)
            if grant["domain"] != domain:
                raise web.HTTPUnauthorized(text="Inference grant changed")
            body = _validate_request(body, grant)
            base_url = credentials["base_url"].rstrip("/")
            access_token = credentials["api_key"]
            if not isinstance(access_token, str) or not access_token:
                raise ValueError("Missing upstream credential")
            identity = hashlib.sha256(
                json.dumps([grant["domain"], request.headers.get("session_id", "")]).encode()
            ).hexdigest()
            headers = {
                **codex_cloudflare_headers(access_token, base_url=base_url),
                "Authorization": f"Bearer {access_token}",
                "session_id": identity,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
            # Prompt caching is also partitioned; the worker never selects an
            # owner cache namespace or supplies trusted upstream headers.
            body["prompt_cache_key"] = identity
            async with request.app[_CLIENT].post(
                base_url + "/responses", json=body, headers=headers, allow_redirects=False,
            ) as upstream:
                if upstream.status != 200:
                    status = upstream.status if upstream.status in {429, 503} else 502
                    raise _upstream_failure(status)
                # Avoid allocating an arbitrarily inflated compressed chunk
                # before the byte counter can see it. Native JSON/SSE needs no
                # compression; a provider ignoring identity fails closed.
                if upstream.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise web.HTTPBadGateway(text="Invalid inference response")
                content_type = upstream.headers.get("Content-Type", "").split(";", 1)[0]
                prefix = b""
                if not content_type and body.get("stream") is True:
                    # The native Codex endpoint can omit Content-Type. Sniff
                    # a bounded SSE line rather than trusting a 200 response
                    # or discarding bytes before forwarding the real stream.
                    prefix = await upstream.content.readline()
                    if prefix.startswith((b"event:", b"data:", b":")):
                        content_type = "text/event-stream"
                if content_type not in {"text/event-stream", "application/json"}:
                    raise web.HTTPBadGateway(text="Invalid inference response")
                if len(prefix) > limits.max_response_bytes:
                    raise web.HTTPBadGateway(text="Inference response limit exceeded")
                response = web.StreamResponse(status=200, headers={"Content-Type": content_type})
                try:
                    await response.prepare(request)
                    received = len(prefix)
                    if prefix:
                        await response.write(prefix)
                    async for chunk in upstream.content.iter_chunked(65536):
                        received += len(chunk)
                        if received > limits.max_response_bytes:
                            raise web.HTTPBadGateway(text="Inference response limit exceeded")
                        await response.write(chunk)
                    await response.write_eof()
                    return response
                except BaseException:
                    # Headers may already be on the wire. Never append a second
                    # HTTP response or a successful EOF to a truncated JSON/SSE.
                    # abort(), not close(), discards queued slow-reader output.
                    if request.transport is not None:
                        request.transport.abort()
                    raise
        except web.HTTPException:
            raise
        except AuthError as error:
            raise _upstream_failure(429 if is_rate_limited_auth_error(error) else 503) from None
        except TimeoutError:
            raise
        except (aiohttp.ClientError, KeyError, ValueError, TypeError):
            raise web.HTTPBadGateway(text="Inference unavailable") from None

    async def send_error(request: web.Request, error: web.HTTPException) -> web.StreamResponse:
        # Do not raise HTTPException back to aiohttp: its protocol finalizes
        # raised errors *after* our deadline and admission scope have exited.
        response = web.Response(status=error.status, body=error.body, headers=error.headers)
        response.force_close()  # Do not drain an unfinished upload for reuse.
        await response.prepare(request)
        await response.write_eof()
        return response

    async def admitted_response(request: web.Request) -> web.StreamResponse:
        deadline = asyncio.get_running_loop().time() + limits.request_timeout_seconds
        domain = _grant(policy_loader, request.headers.get("Authorization", ""))["domain"]
        # No waiting queue and no body/credential allocation before admission.
        # One request per domain, two globally, 120 attempts/domain/sliding hour.
        # These are explicit canary safety limits, not a token-cost estimate.
        if (domain in active_domains or len(active_domains) >= 2
                or domain in loading or len(loading) >= limits.max_credential_loads):
            raise web.HTTPTooManyRequests(text="Inference capacity is busy")
        now = time.monotonic()
        history = attempts.setdefault(domain, deque())
        while history and history[0] <= now - 3600:
            history.popleft()
        if len(history) >= 120:
            raise web.HTTPTooManyRequests(text="Inference request budget exhausted")
        history.append(now)
        active_domains.add(domain)
        sending_error = False
        try:
            try:
                # One absolute deadline covers upload, credentials, upstream and
                # all normal/error prepare/write/EOF, while retaining admission.
                async with asyncio.timeout_at(deadline):
                    try:
                        return await responses(request, domain)
                    except web.HTTPException as error:
                        failure = error
                    except TimeoutError:
                        raise
                    except Exception:
                        failure = web.HTTPBadGateway(text="Inference unavailable")
                    if request.transport is None or request.transport.is_closing():
                        raise asyncio.CancelledError
                    sending_error = True
                    return await send_error(request, failure)
            except TimeoutError:
                if sending_error or request.transport is None or request.transport.is_closing():
                    raise asyncio.CancelledError
                # The original deadline is never renewed for inference. Only an
                # unstarted timeout response gets a final, bounded send attempt:
                # at most 100ms (and no more than the configured request budget).
                # Admission remains owned until EOF or transport abort.
                async with asyncio.timeout(min(0.1, limits.request_timeout_seconds)):
                    return await send_error(
                        request, web.HTTPGatewayTimeout(text="Inference deadline exceeded"),
                    )
        except BaseException:
            # This includes cancellation and prepare/EOF failures. Abort before
            # releasing admission, and terminate the aiohttp request task rather
            # than letting its protocol attempt another unbounded error response.
            if request.transport is not None:
                request.transport.abort()
            raise asyncio.CancelledError from None
        finally:
            active_domains.remove(domain)

    app.router.add_post("/v1/responses", admitted_response)
    return app


def _upstream_failure(status: int) -> web.HTTPException:
    if status == 429:
        return web.HTTPTooManyRequests(text="Inference temporarily unavailable")
    if status == 503:
        return web.HTTPServiceUnavailable(text="Inference temporarily unavailable")
    return web.HTTPBadGateway(text="Inference temporarily unavailable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--limits", type=Path,
                        help="Trusted JSON object of BrokerLimits fields (startup-only)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    from hermes_cli.auth import resolve_codex_runtime_credentials

    limits = BrokerLimits(**json.loads(args.limits.read_text(encoding="utf-8"))) if args.limits else BrokerLimits()
    app = create_app(
        lambda: json.loads(args.policy.read_text(encoding="utf-8")),
        resolve_codex_runtime_credentials,
        limits=limits,
    )
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
