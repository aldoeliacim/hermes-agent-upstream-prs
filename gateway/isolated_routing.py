"""Strict actor/audience routing into a parent-owned managed native Relay.

Optional gateway.isolated_agents configuration uses local_owners, workers and
routes. Each worker domain has gateway_id, secret_env and bot_id, and exactly
one route (platform/chat_id/chat_type/thread_id/sender_ids/domain, optional
scope_id). listen_host/listen_port and an absolute persistent store_path are
required. No HTTP proxy, enrollment, implicit owner, or worker agent runs here.

A supervisor-owned .isolated-agents-required sentinel beside config.yaml makes
missing configuration fatal to routing. Malformed or duplicate-key config is
always closed. Config and credentials are snapshotted once, restart required.
"""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

from aiohttp import web
from gateway.config import Platform
from gateway.isolated_relay import ManagedRelayConnector, RelayBinding
from gateway.isolated_relay_store import DuplicateDelivery
from gateway.platforms.base import BasePlatformAdapter, SendResult

logger = logging.getLogger(__name__)


class IsolationRejected(ValueError):
    pass


class AdmissionRetry(RuntimeError):
    """Not durably accepted. Native intake must retain its event, not ACK it."""


def _identifier(value):
    if (not isinstance(value, str) or not value or value != value.strip()
            or len(value) > 512 or '*' in value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise IsolationRejected('Invalid identifier')
    return value


def _audience(platform, chat_type, chat_id, thread_id, scope_id=None):
    platform = Platform(_identifier(platform))
    if chat_type not in {'dm', 'group', 'channel', 'thread', 'forum'}:
        raise IsolationRejected('Invalid audience')
    return (platform, chat_type, _identifier(chat_id),
            _identifier(thread_id) if thread_id is not None else None,
            _identifier(scope_id) if scope_id is not None else None)


@dataclass(frozen=True)
class Route:
    audience: tuple
    sender_ids: frozenset
    binding: RelayBinding


@dataclass(frozen=True)
class IsolationPolicy:
    owners: tuple = ()
    routes: tuple = ()
    listen_host: str = ''
    listen_port: int = 0
    store_path: str = ''
    invalid: bool = False

    def resolve(self, source):
        if self.invalid or source is None:
            raise IsolationRejected('Isolation unavailable')
        try:
            audience = _audience(source.platform.value, source.chat_type,
                                 source.chat_id, source.thread_id, source.scope_id)
            sender = _identifier(source.user_id)
        except (AttributeError, TypeError, ValueError):
            raise IsolationRejected('Invalid source') from None
        if (audience[1] == 'dm' and audience[3:] == (None, None)
                and (audience[0], audience[2], sender) in self.owners):
            return None
        for route in self.routes:
            if route.audience == audience and sender in route.sender_ids:
                return route
        raise IsolationRejected('No authorized route')


def policy_from_config(raw):
    """Poison the entire policy on any malformed row, including missing secrets."""
    from agent.secret_scope import current_secret_scope, get_secret
    try:
        if not isinstance(raw, dict) or not isinstance(raw.get('gateway', {}), dict):
            raise IsolationRejected('Invalid config')
        gateway = raw.get('gateway', {})
        if 'isolated_agents' not in gateway:
            return None
        cfg = gateway['isolated_agents']
        if (not isinstance(cfg, dict) or not cfg or set(cfg) - {
                'local_owners', 'workers', 'routes', 'listen_host', 'listen_port', 'store_path'}):
            raise IsolationRejected('Invalid policy')
        host = _identifier(cfg['listen_host'])
        port = cfg['listen_port']
        path = _identifier(cfg['store_path'])
        if (type(port) is not int or not 0 <= port <= 65535
                or not Path(path).is_absolute() or path == ':memory:'):
            raise IsolationRejected('Invalid listener/store')
        if (not isinstance(cfg['workers'], dict) or not cfg['workers']
                or not isinstance(cfg['routes'], list) or not cfg['routes']
                or not isinstance(cfg.get('local_owners', []), list)):
            raise IsolationRejected('Invalid tables')
        workers, ids, secrets = {}, set(), set()
        for domain, item in cfg['workers'].items():
            _identifier(domain)
            if not isinstance(item, dict) or set(item) != {'gateway_id', 'secret_env', 'bot_id'}:
                raise IsolationRejected('Invalid worker')
            gid, bot = _identifier(item['gateway_id']), _identifier(item['bot_id'])
            env = _identifier(item['secret_env'])
            if not re.fullmatch(r'[A-Z_][A-Z0-9_]*', env):
                raise IsolationRejected('Invalid credential name')
            scope = current_secret_scope()
            if scope is not None and env not in scope:
                raise IsolationRejected('Missing scoped secret')
            secret = get_secret(env)
            if not isinstance(secret, str) or not secret.strip() or gid in ids or secret in secrets:
                raise IsolationRejected('Invalid credential/binding')
            ids.add(gid)
            secrets.add(secret)
            workers[domain] = (gid, bot, secret)
        owners = []
        for item in cfg.get('local_owners', []):
            if not isinstance(item, dict) or set(item) != {'platform', 'chat_id', 'user_id'}:
                raise IsolationRejected('Invalid owner')
            a = _audience(item['platform'], 'dm', item['chat_id'], None)
            owner = (a[0], a[2], _identifier(item['user_id']))
            if owner in owners:
                raise IsolationRejected('Duplicate owner')
            owners.append(owner)
        routes, audiences, domains = [], set(), set()
        for item in cfg['routes']:
            if not isinstance(item, dict) or set(item) - {
                    'platform', 'chat_id', 'chat_type', 'thread_id', 'scope_id', 'sender_ids', 'domain'}:
                raise IsolationRejected('Invalid route')
            a = _audience(item['platform'], item['chat_type'], item['chat_id'],
                          item.get('thread_id'), item.get('scope_id'))
            # Native recipients must be canonical, not aliases resolved later
            # by an adapter. Only these adapter scope semantics are reviewed.
            if a[4] is not None:
                raise IsolationRejected('Unsupported native scope')
            if a[0] is Platform.TELEGRAM:
                if (not re.fullmatch(r'-?[1-9][0-9]*', a[2])
                        or (a[3] is not None and not re.fullmatch(r'[1-9][0-9]*', a[3]))
                        or (a[1] == 'dm' and a[3] is not None)):
                    raise IsolationRejected('Unsupported Telegram audience')
            elif a[0] is Platform.WHATSAPP:
                suffix = r'(?:s\.whatsapp\.net|lid)' if a[1] == 'dm' else r'g\.us'
                if (a[1] not in {'dm', 'group'} or a[3] is not None
                        or not re.fullmatch(r'[1-9][0-9]*@' + suffix, a[2])):
                    raise IsolationRejected('Unsupported WhatsApp audience')
            else:
                raise IsolationRejected('Unsupported native adapter')
            senders = item['sender_ids']
            if not isinstance(senders, list) or not senders:
                raise IsolationRejected('Missing actors')
            senders = frozenset(_identifier(s) for s in senders)
            domain = item['domain']
            if a in audiences or domain in domains:
                raise IsolationRejected('Shared audience/worker')
            if a[1] == 'dm' and any((a[0], a[2], s) in owners for s in senders):
                raise IsolationRejected('Owner overlap')
            gid, bot, secret = workers[domain]
            binding = RelayBinding(gid, a[0], a[2], a[3], bot, a[4], secret)
            routes.append(Route(a, senders, binding))
            audiences.add(a)
            domains.add(domain)
        if domains != set(workers):
            raise IsolationRejected('Unbound worker')
        return IsolationPolicy(tuple(owners), tuple(routes), host, port, path)
    except (ValueError, TypeError, KeyError, AttributeError):
        return IsolationPolicy(invalid=True)


def load_policy(config_path):
    import yaml
    config_path = Path(config_path)
    required = True
    try:
        config_path.with_name('.isolated-agents-required').lstat()
    except FileNotFoundError:
        required = False
    except OSError:
        pass

    class UniqueKeyLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            keys = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                if key in keys:
                    raise IsolationRejected('Duplicate config key')
                keys.add(key)
            return super().construct_mapping(node, deep=deep)
    try:
        raw = yaml.load(config_path.read_text(encoding='utf-8'), Loader=UniqueKeyLoader)
    except FileNotFoundError:
        return IsolationPolicy(invalid=True) if required else None
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        return IsolationPolicy(invalid=True)
    policy = policy_from_config({} if raw is None else raw)
    return IsolationPolicy(invalid=True) if required and policy is None else policy


class IsolatedRelayRuntime:
    """One loop-owned server, startup lock and shutdown boundary per gateway."""
    def __init__(self, policy, adapters):
        self.policy = policy
        self.adapters = adapters
        self.connector = None
        self.server = None
        self.lock = asyncio.Lock()
        self.closed = False
        self._admissions = {r.binding.gateway_id: asyncio.Lock() for r in policy.routes}
        self._waiters = {r.binding.gateway_id: 0 for r in policy.routes}
        self._stopped = asyncio.Event()

    @property
    def addresses(self):
        return self.server.addresses if self.server else []

    async def start(self):
        async with self.lock:
            if self.closed or self.policy.invalid:
                raise IsolationRejected('Relay unavailable')
            if self.server is not None:
                return
            connector = ManagedRelayConnector([r.binding for r in self.policy.routes],
                                              self.native_action, store_path=self.policy.store_path,
                                              media_store_path=(self.policy.store_path + '.media' if any(
                                                  'send_media' in getattr(type(a), 'isolated_relay_ops', ())
                                                  for a in self.adapters.values()) else None),
                                              media_max_bytes=25 * 1024 * 1024,
                                              supported_ops=lambda b: getattr(
                                                  type(self.adapters.get(b.platform)), 'isolated_relay_ops', ()))
            server = web.AppRunner(connector.app)
            try:
                await server.setup()
                await web.TCPSite(server, self.policy.listen_host, self.policy.listen_port).start()
            except BaseException:
                await server.cleanup()
                raise
            self.connector, self.server = connector, server

    async def enqueue(self, event, authorized=None):
        # Snapshot before the first await: authorize exactly what is queued.
        snapshot = deepcopy(event)
        route = self.policy.resolve(snapshot.source)
        if route is None or snapshot.internal:
            raise IsolationRejected('Not a worker event')
        key = route.binding.gateway_id
        if len(snapshot.text.encode('utf-8')) > 65536:
            raise IsolationRejected('Native event too large')
        # Bound concurrent direct callers too. Native intake itself holds one
        # outstanding event; there is no fire-and-forget retry queue here.
        if self._waiters[key] >= 128:
            raise AdmissionRetry('Admission busy')
        self._waiters[key] += 1
        try:
            async with self._admissions[key]:
                delay = 0.05
                while True:
                    if self.closed:
                        raise AdmissionRetry('Relay stopped before acceptance')
                    if (self.policy.resolve(snapshot.source) != route
                            or (authorized is not None and not authorized(snapshot.source))):
                        raise IsolationRejected('Admission authority revoked')
                    try:
                        await self.start()
                        connector = self.connector
                        if connector is None:
                            raise AdmissionRetry('Relay unavailable')
                        # start() awaits: authority can change while it yields.
                        # Commit only after checking the exact snapshot again.
                        if (self.policy.resolve(snapshot.source) != route
                                or (authorized is not None and not authorized(snapshot.source))):
                            raise IsolationRejected('Admission authority revoked')
                        connector.enqueue(key, snapshot.message_id, snapshot)
                        return
                    except DuplicateDelivery:
                        return
                    except (asyncio.QueueFull, OSError, sqlite3.OperationalError):
                        # Caller retains ownership until the transaction commits.
                        # Polling rather than ACK notifications also recovers a
                        # transient SQLite failure when no worker ACK occurs.
                        try:
                            await asyncio.wait_for(self._stopped.wait(), delay)
                        except asyncio.TimeoutError:
                            pass
                        delay = min(delay * 2, 1.0)
        finally:
            self._waiters[key] -= 1

    async def close(self):
        async with self.lock:
            self.closed = True
            self._stopped.set()
            if self.server is not None:
                server, self.server = self.server, None
                await server.cleanup()

    async def native_action(self, action):
        rejected = SendResult(False, error='native action failed')
        if self.closed:
            return rejected
        binding = next((r.binding for r in self.policy.routes
                        if r.binding.gateway_id == action.gateway_id), None)
        if binding is None or (action.platform, action.chat_id, action.thread_id, action.scope_id) != (
                binding.platform, binding.chat_id, binding.thread_id, binding.scope_id):
            return rejected
        adapter = self.adapters.get(binding.platform)
        if adapter is None or action.op not in getattr(type(adapter), 'isolated_relay_ops', ()):
            return rejected
        metadata = {'thread_id': binding.thread_id, 'scope_id': binding.scope_id,
                    'strict_scope': True, 'disable_link_previews': True, 'notify': action.notify}
        try:
            if action.op == 'send':
                result = await adapter.send(binding.chat_id, action.content,
                                            reply_to=action.reply_to, metadata=metadata)
            elif action.op == 'edit':
                # Base edit has no metadata seam. Until hardened, don't allow
                # edits containing links to trigger native preview fetching.
                if re.search(r'(?i)(?:https?://|www\.)', action.content or ''):
                    return rejected
                result = await adapter.edit_message(binding.chat_id, action.message_id,
                                                    action.content, finalize=True)
            elif action.op == 'react':
                method = adapter.unreact if action.remove else adapter.react
                result = SendResult(await method(binding.chat_id, action.message_id, action.emoji or '') is True)
            elif action.op == 'typing':
                if type(adapter).send_typing is BasePlatformAdapter.send_typing:
                    return rejected
                await adapter.send_typing(binding.chat_id, metadata=metadata)
                result = SendResult(True)
            elif action.op == 'send_media':
                from gateway.native_media import read_owned
                if self.connector is None or self.connector._media is None:
                    return rejected
                owned, data = read_owned(self.connector._media, binding.gateway_id, action.media)
                result = await adapter.send_owned_media(
                    binding.chat_id, data, filename=owned.filename, content_type=owned.content_type,
                    kind=action.media_kind, caption=action.content or '',
                    reply_to=action.reply_to, metadata=metadata)
            else:
                return rejected
            if not isinstance(result, SendResult) or result.success is not True:
                return rejected
            return SendResult(True, message_id=result.message_id)
        except Exception:
            return rejected


def local_source(runner, source):
    """Observers may see legacy/explicit owner only; never isolated or unbound."""
    policy = getattr(runner, '_isolated_policy', None)
    if policy is None:
        return True
    if isolation_unavailable(runner):
        return False
    try:
        return policy.resolve(source) is None
    except IsolationRejected:
        return False


async def dispatch_isolated(runner, event, *, native=False):
    """True means consumed, even when denied/unavailable. Never local fallback."""
    policy = getattr(runner, '_isolated_policy', None)
    if policy is None:
        if native:
            raise AdmissionRetry('Native admission policy unavailable')
        return False
    if isolation_unavailable(runner):
        if native:
            raise AdmissionRetry('Isolation unavailable')
        return True
    try:
        if policy.resolve(event.source) is None:
            return False
        if native and getattr(runner, '_draining', False):
            raise AdmissionRetry('Gateway draining')
        if (event.internal or getattr(event.source, 'profile_route_rejected', False) is True
                or getattr(runner, '_draining', False)
                or not runner._is_user_authorized_for_source(event.source)):
            return True
        if runner._isolated_runtime is None:
            runner._isolated_runtime = IsolatedRelayRuntime(policy, runner.adapters)
        if runner._isolated_runtime.policy is not policy:
            raise AdmissionRetry('Native runtime belongs to an earlier policy')
        def authorized(source):
            if (runner._isolated_policy is not policy or isolation_unavailable(runner)
                    or getattr(runner, '_draining', False)):
                raise AdmissionRetry('Native admission policy changing or draining')
            return runner._is_user_authorized_for_source(source)
        if native:
            await runner._isolated_runtime.enqueue(event, authorized=authorized)
        else:
            await runner._isolated_runtime.enqueue(event)
    except AdmissionRetry:
        raise
    except IsolationRejected:
        return True
    except Exception:
        # Never log worker text, credentials, native errors or local history.
        logger.error('Isolated relay admission failed; no local fallback')
        if native:
            raise AdmissionRetry('Admission not committed') from None
    return True


def isolation_unavailable(runner):
    """No admission during shutdown or unsupported profile multiplexing."""
    policy = getattr(runner, '_isolated_policy', None)
    return policy is not None and (
        policy.invalid or getattr(runner, '_isolated_stopped', False)
        or bool(getattr(getattr(runner, 'config', None), 'multiplex_profiles', False)))
