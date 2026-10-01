"""Managed, preprovisioned native Relay endpoint; no enrollment or agent runtime.

The parent supplies immutable bindings and an async native-platform callback.
Run ``connector.app`` with aiohttp, and call ``enqueue`` on its event loop with
an already-authorized native MessageEvent. /relay is text-only by default.
Supplying media_store_path enables the native wire under
/gateways/{gateway_id}/relay; use that dial URL so native media_base_url keeps
its gateway prefix. TLS/network containment and credentials belong to the parent.

Delivery is bounded, FIFO and ACK-gated. Supply ``store_path`` for durable
restart/reconnect replay; omitted paths are ephemeral (test/legacy use only).
Production parents must supply a persistent, private per-connector file path.
Duplicate IDs and message targets have bounded ``history_size`` windows.
ACK means transport acceptance, NOT agent completion or exactly-once execution.
A native send and its authority commit are not a distributed transaction; a
crash between them can leave an untracked send (subsequent edits fail closed).
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import re
import secrets
from pathlib import Path
import sqlite3
from typing import Any, Awaitable, Callable, Literal, Sequence, cast
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.isolated_relay_store import RelayStore
from gateway.isolated_relay_media import (RelayMediaStore, OwnedMedia,
    DEFAULT_MEDIA_MAX_BYTES, MEDIA_KINDS, validate_filename)
from gateway.relay.auth import verify_token
from gateway.relay.descriptor import CapabilityDescriptor, CONTRACT_VERSION


@dataclass(frozen=True)
class RelayBinding:
    gateway_id: str
    platform: Platform
    chat_id: str
    thread_id: str | None
    bot_id: str
    scope_id: str | None = None
    secret: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)

    def __post_init__(self):
        if (not isinstance(self.platform, Platform)
                or any(not isinstance(v, str) or not v for v in
                       (self.gateway_id, self.chat_id, self.bot_id, self.secret))
                or any(v is not None and (not isinstance(v, str) or not v)
                       for v in (self.thread_id, self.scope_id))):
            raise ValueError("invalid binding")


@dataclass(frozen=True)
class RelayAction:
    """Validated native action. No credentials or worker-controlled routing map."""
    gateway_id: str
    platform: Platform
    chat_id: str
    thread_id: str | None
    scope_id: str | None
    op: Literal["send", "edit", "react", "typing", "send_media"]
    content: str | None = None
    message_id: str | None = None
    reply_to: str | None = None
    emoji: str | None = None
    remove: bool = False
    notify: bool = False
    media: OwnedMedia | None = None
    media_kind: str | None = None


class ManagedRelayConnector:
    def __init__(self, bindings: Sequence[RelayBinding],
                 callback: Callable[[RelayAction], Awaitable[SendResult]], *,
                 max_pending: int = 128, history_size: int = 4096,
                 store_path: str | Path | None = None,
                 media_store_path: str | Path | None = None,
                 media_max_bytes: int = DEFAULT_MEDIA_MAX_BYTES,
                 supported_ops: Callable[[RelayBinding], Sequence[str]] | None = None):
        if (type(max_pending) is not int or type(history_size) is not int
                or max_pending < 1 or history_size < 1):
            raise ValueError("invalid queue limits")
        self._bindings = {b.gateway_id: b for b in bindings}
        if len(self._bindings) != len(bindings) or len({b.secret for b in bindings}) != len(bindings):
            raise ValueError("duplicate binding or secret")
        self._callback = callback
        self._supported_ops = supported_ops
        self._store = RelayStore(store_path, bindings, max_pending, history_size)
        self._media = (RelayMediaStore(media_store_path, bindings, media_max_bytes)
                       if media_store_path is not None else None)
        self._ready = {key: asyncio.Event() for key in self._bindings}
        self._sockets: dict[str, web.WebSocketResponse] = {}
        self.app = web.Application()
        self.app.router.add_get("/relay", self._relay)
        if self._media is not None:
            prefix = "/gateways/{gateway_id}/relay"
            self.app.router.add_get(prefix, self._relay)
            self.app.router.add_post(prefix + "/media", self._media.upload)
            self.app.router.add_get(prefix + "/media/{media_id}", self._media.download)
        self.app.on_shutdown.append(self._shutdown)
        self.app.on_cleanup.append(self._cleanup)

    async def _cleanup(self, app):
        self._store.close()
        if self._media is not None:
            self._media.close()

    async def export_media(self, gateway_id, source, *, filename, content_type):
        """Export only trusted parent-supplied bytes/stream, never event paths."""
        if self._media is None:
            raise ValueError("media disabled")
        return await self._media.export(gateway_id, source, filename=filename,
                                        content_type=content_type)

    def enqueue(self, gateway_id: str, delivery_id: str, event: MessageEvent, *,
                media: Sequence[OwnedMedia] = (), media_base_url: str | None = None) -> None:
        """Snapshot an authorized event; reject foreign scope before queueing.

        Media must be explicit OwnedMedia exports from export_media(). The
        parent supplies its public /gateways/{gateway_id} base, NOT worker
        input. Clear the original event's media_urls on a copy: platform paths
        are never export authority. Uploaded objects cannot masquerade as
        trusted inbound exports. Store metadata, not caller overrides, is sent.

        Raises ValueError for a rejected event/duplicate ID, asyncio.QueueFull
        for backpressure. Caller must retain/retry rejected overflow deliveries.
        Raw platform payload, local paths, prompt controls and metadata never
        cross this boundary. The caller's event/source are not modified.
        """
        b = self._bindings.get(gateway_id)
        exported = []
        if media:
            if self._media is None or not isinstance(media_base_url, str):
                raise ValueError("media unavailable")
            base = urlsplit(media_base_url)
            if (base.scheme not in ("http", "https") or not base.netloc
                    or base.username or base.password or base.query or base.fragment
                    or base.path != f"/gateways/{gateway_id}"):
                raise ValueError("invalid media base URL")
            for supplied in media:
                if not isinstance(supplied, OwnedMedia):
                    raise ValueError("media unavailable")
                owned = self._media.resolve(gateway_id, supplied.id)
                if owned.uploaded or owned != supplied:
                    raise ValueError("media unavailable")
                exported.append(owned)
        event = deepcopy(event)
        src = event.source
        if (b is None or src is None
                or (src.platform, src.chat_id, src.thread_id, src.scope_id) !=
                   (b.platform, b.chat_id, b.thread_id, b.scope_id)
                or not isinstance(delivery_id, str) or not delivery_id
                or not isinstance(event.message_id, str) or not event.message_id
                or not isinstance(event.text, str)
                or (not exported and event.message_type not in (MessageType.TEXT, MessageType.COMMAND))
                or (exported and event.message_type not in (MessageType.TEXT, MessageType.PHOTO, MessageType.VOICE,
                                                            MessageType.AUDIO, MessageType.VIDEO, MessageType.DOCUMENT))
                or event.media_urls or event.prompt_response):
            raise ValueError("event rejected")
        # Project only public source fields; do not forward profile-routing or
        # internal authorization overrides from a multiplexed parent process.
        source = {k: v for k, v in src.to_dict().items() if k in {
            "platform", "chat_id", "chat_type", "chat_name", "chat_topic",
            "thread_id", "scope_id", "parent_chat_id", "user_id", "user_name",
            "user_id_alt", "chat_id_alt", "message_id"}}
        source["message_id"] = event.message_id
        event_frame: dict[str, Any] = {
            "source": source, "text": event.text, "message_type": event.message_type.value,
            "message_id": event.message_id, "reply_to_message_id": event.reply_to_message_id,
            "reply_to": {"text": event.reply_to_text, "author": event.reply_to_author_name,
                         "is_own": event.reply_to_is_own_message}}
        if exported:
            urls = [f"{media_base_url}/relay/media/{m.id}" for m in exported]
            event_frame["media_urls"] = urls
            event_frame["media"] = [
                {"url": url, "mime": m.content_type, "size": m.size,
                 "filename": m.filename, "sha256": m.sha256}
                for url, m in zip(urls, exported)]
        frame = {"type": "inbound", "bufferId": delivery_id, "event": event_frame}
        payload = json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n"
        # Validate the exact serialized projection, including custom to_dict
        # implementations. Never authorize one object and transmit another.
        snapshot = json.loads(payload)["event"]
        wire_source = snapshot["source"]
        if tuple(wire_source.get(k) for k in ("platform", "chat_id", "thread_id", "scope_id")) != (
                b.platform.value, b.chat_id, b.thread_id, b.scope_id):
            raise ValueError("event rejected")
        self._store.enqueue(gateway_id, delivery_id, payload, snapshot["message_id"])
        self._ready[gateway_id].set()

    async def _shutdown(self, app):
        for ws in list(self._sockets.values()):
            await ws.close(code=1001, message=b"shutdown")

    async def _relay(self, request: web.Request):
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        # Verify both signature AND exact gatewayId. A valid secret for A never
        # authenticates B, even if the hello or signed payload claims otherwise.
        b = next((b for b in self._bindings.values()
                  if verify_token(token, [b.secret]) == b.gateway_id), None)
        ws = web.WebSocketResponse(max_msg_size=65536, heartbeat=30)
        await ws.prepare(request)
        if (b is None or request.match_info.get("gateway_id", b.gateway_id) != b.gateway_id):
            await ws.close(code=4401, message=b"unauthorized")
            return ws
        # 4401 means permanent credential revocation to the native client.
        # Occupancy and not-yet-connected native adapters are transient: do
        # not advertise an empty (legacy-default) descriptor or replay early.
        if b.gateway_id in self._sockets or not self.operations(b):
            await ws.close(code=1013, message=b"temporarily unavailable")
            return ws
        self._sockets[b.gateway_id] = ws
        pump = None
        sent_id = None
        try:
            async def deliver():
                nonlocal sent_id
                while True:
                    await self._ready[b.gateway_id].wait()
                    self._ready[b.gateway_id].clear()
                    pending = self._store.peek(b.gateway_id)
                    if pending and sent_id is None:
                        sent_id, payload = pending
                        await ws.send_str(payload)

            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    break
                try:
                    for line in msg.data.splitlines():
                        if not line.strip():
                            continue
                        frame = json.loads(line)
                        if not isinstance(frame, dict):
                            raise ValueError
                        kind = frame.get("type")
                        if pump is None:
                            if (kind != "hello" or frame.get("platform") != b.platform.value
                                    or frame.get("botId") != b.bot_id
                                    or set(frame) - {"type", "platform", "botId", "command_manifest"}):
                                raise ValueError
                            desc = CapabilityDescriptor(
                                contract_version=CONTRACT_VERSION, platform=str(b.platform.value),
                                label="Managed Relay", max_message_length=4096,
                                supports_draft_streaming=False, supports_edit='edit' in self.operations(b),
                                supports_threads=False, markdown_dialect="plain", len_unit="chars",
                                supported_ops=self.operations(b))
                            await ws.send_str(json.dumps({"type": "descriptor", "descriptor": asdict(desc)}) + "\n")
                            pump = asyncio.create_task(deliver())
                            self._ready[b.gateway_id].set()
                        elif kind == "inbound_ack":
                            if set(frame) != {"type", "bufferId"} or frame.get("bufferId") != sent_id or sent_id is None:
                                raise ValueError
                            self._store.ack(b.gateway_id, sent_id)
                            sent_id = None
                            self._ready[b.gateway_id].set()
                        elif kind == "outbound":
                            result = await self._outbound(b, frame, f"{request.scheme}://{request.host}")
                            await ws.send_str(json.dumps({"type": "outbound_result",
                                                "requestId": frame.get("requestId"), "result": result}) + "\n")
                        else:
                            raise ValueError
                except (ValueError, TypeError, sqlite3.Error):
                    await ws.close(code=4401, message=b"unauthorized")
                    break
        finally:
            if pump is not None:
                pump.cancel()
                with suppress(asyncio.CancelledError, ConnectionError):
                    await pump
            self._sockets.pop(b.gateway_id, None)
        return ws

    def operations(self, binding):
        supported = ("send", "edit", "react", "typing") + (
            ("send_media",) if self._media is not None else ())
        if self._supported_ops is None:
            return supported
        native = self._supported_ops(binding)
        return tuple(op for op in supported if op in native)

    async def _outbound(self, b: RelayBinding, frame: dict, origin: str = "") -> dict:
        rejected = {"success": False, "error": "request rejected"}
        action = frame.get("action")
        if (set(frame) - {"type", "requestId", "action", "platform", "botId"}
                or not isinstance(frame.get("requestId"), str)
                or frame.get("platform", b.platform.value) != b.platform.value
                or frame.get("botId", b.bot_id) != b.bot_id
                or not isinstance(action, dict)):
            return rejected
        op = action.get("op")
        allowed = {"send": {"content", "reply_to"}, "edit": {"content", "message_id"},
                   "react": {"message_id", "emoji", "remove"}, "typing": {"content"}}
        if self._media is not None:
            allowed["send_media"] = {"source_url", "media_kind", "content", "filename", "reply_to"}
        if not isinstance(op, str) or op not in allowed or op not in self.operations(b):
            return rejected
        if set(action) - ({"op", "chat_id", "metadata"} | allowed[op]):
            return rejected
        md = action.get("metadata", {})
        if (action.get("chat_id") != b.chat_id or not isinstance(md, dict)
                or set(md) - {"thread_id", "scope_id", "user_id", "notify", "job_id"}
                or type(md.get("notify", False)) is not bool
                or md.get("thread_id", b.thread_id if op in {'react', 'edit'} else None) != b.thread_id
                or md.get("scope_id", b.scope_id) != b.scope_id):
            return rejected
        # Native cron adds this correlation tag. It is never routing authority
        # and is intentionally not forwarded into the native callback.
        if 'job_id' in md and (not isinstance(md['job_id'], str)
                              or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', md['job_id'])):
            return rejected
        for key in ("content", "message_id", "reply_to", "emoji"):
            if key in {"reply_to", "emoji"} and action.get(key) is None:
                continue
            if key in action and not isinstance(action[key], str):
                return rejected
        if op in ("send", "edit") and not action.get("content"):
            return rejected
        if op == "react" and (type(action.get("remove", False)) is not bool
                              or (not action.get("remove", False) and not action.get("emoji"))):
            return rejected
        target = action.get("message_id") if op in ("react", "edit") else action.get("reply_to")
        if op in ("react", "edit") or action.get("reply_to") is not None:
            ownership = self._store.target(b.gateway_id, target) if target else None
            if ownership is None or (op == "edit" and not ownership):
                return rejected
        media = None
        if op == "send_media":
            if self._media is None:
                return rejected
            try:
                if not isinstance(action.get("media_kind"), str) or action["media_kind"] not in MEDIA_KINDS:
                    return rejected
                media = self._media.resolve_url(b.gateway_id, action.get("source_url"), origin)
                # Preserve uploaded metadata; action metadata cannot rename or retype it.
                if "filename" in action and validate_filename(action["filename"]) != media.filename:
                    return rejected
            except (ValueError, TypeError, OSError):
                return rejected
        native = RelayAction(gateway_id=b.gateway_id, platform=b.platform, chat_id=b.chat_id,
                             thread_id=b.thread_id, scope_id=b.scope_id,
                             op=cast(Literal["send", "edit", "react", "typing", "send_media"], op),
                             media=media,
                             notify=md.get("notify", False),
                             **{key: action[key] for key in allowed[op]
                                if key in action and key not in {"source_url", "filename"}})
        try:
            result = await self._callback(native)
            if not isinstance(result, SendResult) or result.success is not True:
                return {"success": False, "error": "native action failed"}
            response: dict = {"success": True}
            if isinstance(result.message_id, str) and result.message_id:
                response["message_id"] = result.message_id
                if op in ("send", "send_media"):
                    self._store.remember_sent(b.gateway_id, result.message_id)
            return response
        except Exception:
            # Native errors can contain platform credentials or request payloads.
            # Neither exception text nor raw_response is ever returned to workers.
            return {"success": False, "error": "native action failed"}
