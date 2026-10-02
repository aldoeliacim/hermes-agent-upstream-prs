"""Native bytes consumers. Never route artifacts through legacy path helpers."""
from __future__ import annotations

import hashlib
from urllib.parse import quote

from gateway.platforms.base import SendResult

NATIVE_MEDIA_LIMIT = 25 * 1024 * 1024


def read_owned(store, gateway_id, supplied):
    """Resolve again at the ROOT consumer; a dataclass is not a capability."""
    if supplied is None or supplied.gateway_id != gateway_id:
        raise ValueError('Foreign artifact')
    owned = store.resolve(gateway_id, supplied.id, uploaded_only=True)
    if owned != supplied or owned.size > NATIVE_MEDIA_LIMIT:
        raise ValueError('Invalid native artifact')
    with owned.open() as stream:
        data = stream.read(NATIVE_MEDIA_LIMIT + 1)
    if len(data) != owned.size or hashlib.sha256(data).hexdigest() != owned.sha256:
        raise ValueError('Changed native artifact')
    return owned, data


async def send_telegram(adapter, chat_id, data, *, filename, content_type, kind,
                        caption='', reply_to=None, metadata):
    from telegram import InputFile, ReplyParameters
    methods = {'image': 'send_photo', 'audio': 'send_audio', 'voice': 'send_voice',
               'video': 'send_video', 'document': 'send_document'}
    fields = {'image': 'photo', 'audio': 'audio', 'voice': 'voice',
              'video': 'video', 'document': 'document'}
    if kind not in methods or not isinstance(data, bytes) or not 0 < len(data) <= NATIVE_MEDIA_LIMIT:
        return SendResult(False, error='Invalid media')
    # PTB uploads InputFile bytes; no Pillow/FFmpeg/URL/local-path resolution.
    kwargs = {fields[kind]: InputFile(data, filename=filename), 'caption': caption or None,
              'chat_id': chat_id, 'disable_notification': not metadata.get('notify', False)}
    if metadata.get('thread_id') is not None:
        kwargs['message_thread_id'] = int(metadata['thread_id'])
    if reply_to is not None:
        kwargs['reply_parameters'] = ReplyParameters(message_id=int(reply_to), allow_sending_without_reply=False)
    sent = await getattr(adapter._bot, methods[kind])(**kwargs)
    return SendResult(True, message_id=str(sent.message_id))


async def send_whatsapp(adapter, chat_id, data, *, filename, content_type, kind,
                        caption='', reply_to=None, metadata):
    import aiohttp
    if (not isinstance(data, bytes) or not 0 < len(data) <= NATIVE_MEDIA_LIMIT
            or metadata.get('thread_id') is not None or reply_to is not None):
        return SendResult(False, error='Unsupported native media')
    headers = {'X-Media-Chat': chat_id, 'X-Media-Kind': kind,
               'X-Media-Filename': quote(filename, safe=''),
               'X-Media-Caption': quote(caption or '', safe=''), 'Content-Type': content_type}
    async with adapter._http_session.post(
        f'http://127.0.0.1:{adapter._bridge_port}/send-owned-media', data=data, headers=headers,
        timeout=aiohttp.ClientTimeout(total=80),
    ) as response:
        response.raise_for_status()
        result = await response.json()
    return SendResult(result.get('success') is True, message_id=result.get('messageId'))
