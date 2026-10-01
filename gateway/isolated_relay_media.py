"""Private, ownership-bound media plane for the managed native Relay.

Native wire, mounted beneath /gateways/{gateway_id}: POST /relay/media raw
bytes -> {id,size,...}; GET /relay/media/{id}. Dial that prefix's /relay with
unchanged RelayMediaClient/media_base_url. No public-URL or host-path resolver.

The parent supplies a persistent PRIVATE directory (never a platform cache),
protects it from worker mounts, and owns retention/quotas and platform-cache
binding. 256 MiB is deployment policy, not a base64/protocol bound. The existing
native client has its own 25 MiB/buffered implementation; this broker streams.
No parsing, rendering, thumbnailing or transcoding occurs here. Native callback
consumers must sandbox those operations rather than execute them as root.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import time
import unicodedata
from typing import AsyncIterable
from urllib.parse import quote, unquote

from aiohttp import web
from gateway.relay.auth import verify_token

CHUNK_SIZE = 65536
DEFAULT_MEDIA_MAX_BYTES = 256 * 1024 * 1024
MEDIA_KINDS = frozenset({'image', 'voice', 'audio', 'video', 'document'})


def validate_filename(name: str) -> str:
    if (not isinstance(name, str) or not name or len(name) > 255
            or name in {'.', '..'} or any(c in name for c in '/\\";')
            or len(name.encode('utf-8')) > 255
            or any(unicodedata.category(c).startswith('C') for c in name)):
        raise ValueError('invalid media filename')
    return name


def _content_type(value: str) -> str:
    if (not isinstance(value, str) or len(value) > 255
            or not re.fullmatch(r'[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+', value)):
        raise ValueError('invalid content type')
    return value


@dataclass(frozen=True)
class OwnedMedia:
    """Resolved immutable object, never a worker-supplied filesystem path."""
    id: str
    gateway_id: str
    filename: str
    content_type: str
    size: int
    sha256: str
    uploaded: bool
    _path: Path = field(repr=False, compare=False)

    def open(self):
        fd = os.open(self._path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size != self.size:
                raise ValueError('media unavailable')
            return os.fdopen(fd, 'rb')
        except BaseException:
            os.close(fd)
            raise


class RelayMediaStore:
    def __init__(self, path, bindings, max_bytes=DEFAULT_MEDIA_MAX_BYTES, *, max_objects=128):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('invalid media cap')
        self.max_bytes = max_bytes
        if type(max_objects) is not int or max_objects < 1:
            raise ValueError('invalid media object cap')
        self.max_objects = max_objects
        self._reserved = {}
        self.bindings = {b.gateway_id: b for b in bindings}
        if any(not re.fullmatch(r'[A-Za-z0-9_-]+', key) for key in self.bindings):
            raise ValueError('media gateway id must be a URL-safe identifier')
        self.path = Path(path)
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.path.lstat()
        # POSIX-only by design: ownership is checked by uid, not by name.
        owner_uid = os.getuid() if hasattr(os, 'getuid') else None
        if owner_uid is None:
            raise ValueError('media store requires a POSIX private owned directory')
        if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid != owner_uid:
            raise ValueError('media store must be a private owned directory')
        db_path = self.path / 'media.sqlite3'
        fd = os.open(db_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        self.db = sqlite3.connect(db_path)
        try:
            self.db.executescript('''
                PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS bindings (gateway_id TEXT PRIMARY KEY, identity TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS media (
                    id TEXT PRIMARY KEY, gateway_id TEXT NOT NULL, filename TEXT NOT NULL,
                    content_type TEXT NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL,
                    uploaded INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS media_lifetime (id TEXT PRIMARY KEY, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS native_exports (
                    gateway_id TEXT NOT NULL, source_id TEXT NOT NULL, media_id TEXT NOT NULL,
                    PRIMARY KEY(gateway_id,source_id));
            ''')
            with self.db:
                self.db.execute('INSERT OR IGNORE INTO media_lifetime SELECT id,? FROM media',
                                (time.time() + 7 * 86400,))
                for b in bindings:
                    identity = json.dumps([b.platform.value, b.chat_id, b.thread_id, b.scope_id, b.bot_id])
                    row = self.db.execute('SELECT identity FROM bindings WHERE gateway_id=?', (b.gateway_id,)).fetchone()
                    if row and row[0] != identity:
                        raise ValueError('stored media binding mismatch')
                    self.db.execute('INSERT OR IGNORE INTO bindings VALUES (?,?)', (b.gateway_id, identity))
            # Interrupted writes were never accepted; only private orphan leaves.
            for blob in self.path.glob('*.blob'):
                if re.fullmatch(r'[0-9a-f]{32}', blob.stem) and not self.db.execute(
                        'SELECT 1 FROM media WHERE id=?', (blob.stem,)).fetchone():
                    blob.unlink()
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def prune(self):
        expired = [r[0] for r in self.db.execute('SELECT id FROM media_lifetime WHERE expires<=?', (time.time(),))]
        with self.db:
            for ident in expired:
                (self.path / (ident + '.blob')).unlink(missing_ok=True)
                self.db.execute('DELETE FROM media WHERE id=?', (ident,))
                self.db.execute('DELETE FROM native_exports WHERE media_id=?', (ident,))
                self.db.execute('DELETE FROM media_lifetime WHERE id=?', (ident,))

    def native_export(self, gateway_id, source_id):
        self.prune()
        row = self.db.execute('SELECT media_id FROM native_exports WHERE gateway_id=? AND source_id=?',
                              (gateway_id, source_id)).fetchone()
        return self.resolve(gateway_id, row[0]) if row else None

    def remember_native(self, gateway_id, source_id, media_id):
        owned = self.resolve(gateway_id, media_id)
        if owned.uploaded or not isinstance(source_id, str) or len(source_id) > 1024:
            raise ValueError('invalid native provenance')
        with self.db:
            self.db.execute('INSERT INTO native_exports VALUES (?,?,?)', (gateway_id, source_id, media_id))

    def resolve(self, gateway_id, media_id, *, uploaded_only=False):
        if gateway_id not in self.bindings or not isinstance(media_id, str) or not re.fullmatch(r'[0-9a-f]{32}', media_id):
            raise ValueError('media unavailable')
        self.prune()
        row = self.db.execute('SELECT id,gateway_id,filename,content_type,size,sha256,uploaded FROM media WHERE id=? AND gateway_id=?',
                              (media_id, gateway_id)).fetchone()
        if row is None or row[4] > self.max_bytes or (uploaded_only and not row[6]):
            raise ValueError('media unavailable')
        return OwnedMedia(id=row[0], gateway_id=row[1], filename=row[2],
                          content_type=row[3], size=row[4], sha256=row[5],
                          uploaded=bool(row[6]), _path=self.path / (media_id + '.blob'))

    def resolve_url(self, gateway_id, url, origin):
        prefix = f'{origin}/gateways/{gateway_id}/relay/media/'
        if not isinstance(url, str) or not url.startswith(prefix):
            raise ValueError('media unavailable')
        return self.resolve(gateway_id, url[len(prefix):], uploaded_only=True)

    async def put(self, gateway_id, chunks: AsyncIterable[bytes | bytearray | memoryview], *, filename, content_type, uploaded):
        if gateway_id not in self.bindings:
            raise ValueError('unknown media owner')
        self.prune()
        count, size = self.db.execute('SELECT count(*),coalesce(sum(size),0) FROM media WHERE gateway_id=?',
                                      (gateway_id,)).fetchone()
        reserved = self._reserved.get(gateway_id, 0)
        if count + reserved >= self.max_objects or size + (reserved + 1) * self.max_bytes > 4 * self.max_bytes:
            raise ValueError('media capacity exhausted')
        self._reserved[gateway_id] = reserved + 1
        try:
            return await self._put(gateway_id, chunks, filename=filename, content_type=content_type, uploaded=uploaded)
        finally:
            self._reserved[gateway_id] -= 1

    async def _put(self, gateway_id, chunks, *, filename, content_type, uploaded):
        if gateway_id not in self.bindings:
            raise ValueError('unknown media owner')
        filename, content_type = validate_filename(filename), _content_type(content_type)
        media_id = secrets.token_hex(16)
        path = self.path / (media_id + '.blob')
        # Exclusive creation: neither a collision nor an attacker can replace an asset.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        size, digest = 0, hashlib.sha256()
        try:
            with os.fdopen(fd, 'wb') as output:
                async for chunk in chunks:
                    if not isinstance(chunk, (bytes, bytearray, memoryview)) or len(chunk) > CHUNK_SIZE:
                        raise ValueError('invalid stream chunk')
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise web.HTTPRequestEntityTooLarge(max_size=self.max_bytes, actual_size=size)
                    output.write(chunk)
                    digest.update(chunk)
                if size == 0:
                    raise ValueError('empty media')
                output.flush()
                os.fsync(output.fileno())
            directory = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            with self.db:
                self.db.execute('INSERT INTO media VALUES (?,?,?,?,?,?,?)',
                    (media_id, gateway_id, filename, content_type, size, digest.hexdigest(), bool(uploaded)))
                self.db.execute('INSERT INTO media_lifetime VALUES (?,?)', (media_id, time.time() + 7 * 86400))
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return self.resolve(gateway_id, media_id)

    async def export(self, gateway_id, source, *, filename, content_type):
        """Trusted parent supplies explicit bytes or a binary read(size) stream.

        Never accepts a path/URL or interprets MessageEvent.media_urls as authority.
        Parent retains responsibility for authorizing the platform cache source.
        """
        if not isinstance(source, bytes) and (isinstance(source, (str, os.PathLike)) or not callable(getattr(source, 'read', None))):
            raise TypeError('export requires bytes or a binary stream')
        async def chunks():
            if isinstance(source, bytes):
                view = memoryview(source)
                for offset in range(0, len(view), CHUNK_SIZE):
                    yield view[offset:offset + CHUNK_SIZE]
            else:
                while True:
                    part = source.read(CHUNK_SIZE)
                    if not part:
                        break
                    yield part
        return await self.put(gateway_id, chunks(), filename=filename, content_type=content_type, uploaded=False)

    def authenticate(self, request):
        gateway_id = request.match_info['gateway_id']
        b = self.bindings.get(gateway_id)
        header = request.headers.get('Authorization', '')
        token = header[7:] if header.startswith('Bearer ') else ''
        if b is None or verify_token(token, [b.secret]) != gateway_id:
            raise web.HTTPUnauthorized()
        return gateway_id

    async def upload(self, request):
        try:
            return await self._upload(request)
        except web.HTTPException as error:
            # Rejected bodies are never drained into application memory. Do not
            # reuse a connection containing an unread/oversized request body.
            error.force_close()
            raise

    async def _upload(self, request):
        gateway_id = self.authenticate(request)  # before any application body read
        if request.headers.get('Content-Encoding'):
            raise web.HTTPBadRequest()
        if request.content_length is not None and request.content_length > self.max_bytes:
            raise web.HTTPRequestEntityTooLarge(max_size=self.max_bytes, actual_size=request.content_length)
        try:
            filename = request.headers.get('X-Media-Filename', 'attachment.bin')
            encoded = request.headers.get('X-Media-Filename*')
            if encoded is not None:
                if not encoded.startswith("UTF-8''") or re.search(r'%(?![0-9A-Fa-f]{2})', encoded):
                    raise ValueError('invalid media filename')
                filename = unquote(encoded[7:], encoding='utf-8', errors='strict')
            media = await self.put(gateway_id, request.content.iter_chunked(CHUNK_SIZE),
                filename=filename,
                content_type=request.headers.get('Content-Type', 'application/octet-stream'), uploaded=True)
        except ValueError:
            raise web.HTTPBadRequest() from None
        return web.json_response({'id': media.id, 'size': media.size, 'sha256': media.sha256,
                                  'filename': media.filename, 'content_type': media.content_type})

    async def download(self, request):
        gateway_id = self.authenticate(request)
        try:
            media = self.resolve(gateway_id, request.match_info['media_id'])
            stream = media.open()  # ownership checked before opening or sending bytes
        except (ValueError, OSError):
            raise web.HTTPNotFound() from None
        with stream:
            response = web.StreamResponse(headers={
                'Content-Type': media.content_type, 'Content-Length': str(media.size),
                'Content-Disposition': "attachment; filename*=UTF-8''" + quote(media.filename, safe=''),
                'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'private, no-store',
                'ETag': '"' + media.sha256 + '"'})
            await response.prepare(request)
            while chunk := stream.read(CHUNK_SIZE):
                await response.write(chunk)
            await response.write_eof()
            return response
