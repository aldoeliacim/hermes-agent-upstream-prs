"""Native admission before debounce, command hooks or destructive ACK.

Only data projected from the receiving SDK is used here. No event path, SDK
cache filename, quoted local history or routing metadata grants authority.
"""
from __future__ import annotations

import asyncio

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

# The routing stack (aiohttp.web behind it) is imported where an event is actually
# admitted, never on the bind path: adapters bind admission synchronously inside
# connect()/task spawn, and that path must stay import-light. Importing aiohttp.web
# runs mimetypes.init() -> open('/etc/mime.types'); under a mocked builtins.open that
# iterates forever (tests/gateway/test_native_admission_import_weight.py).


def isolated_runner(adapter):
    runner = getattr(adapter, 'gateway_runner', None)
    return runner if getattr(runner, '_isolated_policy', None) is not None else None


def bind_native_admission(adapter) -> bool:
    """Bind native intake to its runner lifetime, before connect/task awaits.

    Policy availability is not lifetime authority. Factory replacements share
    the runner's requirement; detached adapters keep their already-bound fence.
    Only these two native transports implement the managed intake protocol.
    """
    # A half-constructed adapter (supervisor tests build one with __new__) has no
    # platform yet: it cannot be managed, and must not crash its own supervisor.
    if getattr(adapter, 'platform', None) not in {Platform.TELEGRAM, Platform.WHATSAPP}:
        return False
    runner = getattr(adapter, 'gateway_runner', None)
    if runner is not None and isolated_runner(adapter) is not None:
        runner._native_admission_required = True
    adapter._native_admission_required = (
        getattr(runner, '_native_admission_required', False) is True
        or getattr(adapter, '_native_admission_required', False) is True
    )
    return adapter._native_admission_required


def telegram_event(adapter, update):
    from telegram import Update

    if not isinstance(update, Update) or update.callback_query is not None:
        return None
    # effective_message also includes a callback's old bot-authored message.
    # Only native message update kinds may become a fresh managed delivery.
    msg = (update.message or update.edited_message or update.channel_post
           or update.edited_channel_post or update.business_message
           or update.edited_business_message)
    if msg is None:
        return None
    chat = msg.chat
    user = msg.from_user
    kind = 'group' if chat.type in {'group', 'supergroup'} else 'channel' if chat.type == 'channel' else 'dm'
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id=str(chat.id), chat_type=kind,
        user_id=str(user.id) if user else None,
        thread_id=adapter._effective_message_thread_id(msg),
        chat_name=chat.title, user_name=user.full_name if user else None)
    return MessageEvent(
        text=msg.text or msg.caption or '', source=source,
        message_id=str(msg.message_id),
        message_type=MessageType.TEXT if msg.text else MessageType.DOCUMENT)


def whatsapp_event(data):
    return MessageEvent(
        text=data.get('body') or '', message_id=data.get('messageId'),
        message_type=MessageType.DOCUMENT if data.get('hasMedia') else MessageType.TEXT,
        source=SessionSource(platform=Platform.WHATSAPP,
            chat_id=data.get('chatId'), chat_type='group' if data.get('isGroup') else 'dm',
            user_id=data.get('senderId'), user_name=data.get('senderName'),
            chat_name=data.get('chatName')))


class TelegramAdmissionQueue(asyncio.Queue):
    """PTB awaits put() BEFORE advancing its getUpdates offset.

    There is at most one fetched SDK batch plus this bounded legacy queue.
    A blocked binding can block subsequent getUpdates batches (Telegram's
    offset is global); already admitted bindings keep draining independently.
    """
    def __init__(self, adapter):
        super().__init__(maxsize=100)
        self.adapter = adapter

    async def put(self, item):
        from telegram import Message, Update
        from telegram.ext._application import _STOP_SIGNAL
        from gateway.isolated_routing import AdmissionRetry, dispatch_isolated, local_source

        # PTB Application.stop puts this exact singleton, not an Update. It
        # must remain usable with no policy; arbitrary objects are NOT control.
        if item is _STOP_SIGNAL:
            await super().put(item)
            return
        while True:
            runner = isolated_runner(self.adapter)
            # This queue's lifetime is managed even when policy disappears.
            if runner is None:
                await asyncio.sleep(0.25)
                continue
            if not isinstance(item, Update):
                return
            query = item.callback_query
            if query is not None:
                # No managed prompt authority pipeline exists yet. Only an
                # explicit local owner can use legacy callbacks, using the
                # actual clicker and a known, accessible message audience.
                msg = query.message
                if (not isinstance(msg, Message) or msg.chat.type not in {
                        'private', 'group', 'supergroup', 'channel'}):
                    return
                source = SessionSource(
                    platform=Platform.TELEGRAM, chat_id=str(msg.chat.id),
                    chat_type='dm' if msg.chat.type == 'private' else (
                        'channel' if msg.chat.type == 'channel' else 'group'),
                    user_id=str(query.from_user.id),
                    thread_id=self.adapter._effective_message_thread_id(msg))
                # local_source defaults open without policy: the check above
                # is required, and no await separates it from this decision.
                if (not local_source(runner, source)
                        or not runner._is_user_authorized_for_source(source)):
                    return
            else:
                event = telegram_event(self.adapter, item)
                if event is None:
                    return
                try:
                    if await dispatch_isolated(runner, event, native=True):
                        return
                except AdmissionRetry:
                    await asyncio.sleep(1)
                    continue
            # Recheck authorization after any capacity wait, rather than
            # releasing a stale local decision into the private handlers.
            if self.full():
                await asyncio.sleep(0.25)
                continue
            self.put_nowait(item)
            return


async def poll_whatsapp(adapter):
    """One task per queued chat (at most 100), retained by the bridge until ACK.

    Poll returns only the oldest event of each chat. A slow/offline binding
    therefore does not prevent other chats from being admitted or ACKed.
    Cancellation leaves uncommitted events in the durable bridge inbox.
    """
    import aiohttp
    from gateway.isolated_routing import AdmissionRetry, dispatch_isolated
    active = {}
    base = f'http://127.0.0.1:{adapter._bridge_port}'

    async def accept(data):
        try:
            runner = isolated_runner(adapter)
            if runner is None:
                raise AdmissionRetry('Admission policy disappeared')
            event = whatsapp_event(data)
            if not await dispatch_isolated(runner, event, native=True):
                # Explicit local owner only. No isolated event reaches this.
                native = await adapter._build_message_event(data)
                if native:
                    await adapter.handle_message(native)
            async with adapter._http_session.post(
                base + '/admission-ack', json={'id': data['_admission_id']},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response.raise_for_status()
        except (AdmissionRetry, aiohttp.ClientError, asyncio.TimeoutError):
            # No ACK: the next poll retries with current authorization.
            return

    try:
        while adapter._running:
            async with adapter._http_session.get(
                base + '/admissions', timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response.raise_for_status()
                batch = await response.json()
            if not isinstance(batch, list) or len(batch) > 100:
                raise AdmissionRetry('Invalid native admission batch')
            for chat, task in list(active.items()):
                if task.done():
                    del active[chat]
                    task.result()
            for data in batch:
                chat = data['chatId']
                if chat not in active and len(active) < 100:
                    active[chat] = asyncio.create_task(accept(data))
            await asyncio.sleep(0.1)
    finally:
        for task in active.values():
            task.cancel()
        await asyncio.gather(*active.values(), return_exceptions=True)
