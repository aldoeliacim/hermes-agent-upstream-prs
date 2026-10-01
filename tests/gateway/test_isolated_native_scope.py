"""Native Telegram scope: failed topics must not widen isolated recipients."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import BadRequest
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def adapter():
    a = TelegramAdapter(PlatformConfig(enabled=True, token='123:synthetic'))
    a._rich_messages_enabled = False
    a._bot = SimpleNamespace(send_message=AsyncMock(), send_chat_action=AsyncMock())
    return a


@pytest.mark.asyncio
@pytest.mark.parametrize('strict', [False, True])
async def test_topic_failure_never_widens_scoped_text(adapter, strict):
    calls = []
    async def send(**kw):
        calls.append(kw)
        if kw.get('message_thread_id') is not None:
            raise BadRequest('Message thread not found')
        return SimpleNamespace(message_id=9)
    adapter._bot.send_message = send
    result = await adapter.send('-100', 'test', metadata={'thread_id': '7', 'strict_scope': strict})
    assert result.success is (not strict)
    assert calls
    if strict:
        assert all(c.get('message_thread_id') == 7 for c in calls)


@pytest.mark.asyncio
async def test_scoped_success_preserves_topic_and_suppresses_previews(adapter):
    adapter._bot.send_message.return_value = SimpleNamespace(message_id=9)
    result = await adapter.send('-100', 'https://example.invalid', metadata={
        'thread_id': '7', 'strict_scope': True, 'disable_link_previews': True})
    assert result.success
    kw = adapter._bot.send_message.call_args.kwargs
    assert kw['message_thread_id'] == 7
    assert kw.get('disable_web_page_preview') is True or getattr(kw.get('link_preview_options'), 'is_disabled', False)


@pytest.mark.asyncio
async def test_scoped_media_anchor_retry_does_not_drop_scope(adapter):
    calls = []
    async def send(**kw):
        calls.append(kw)
        if kw.get('reply_to_message_id'):
            raise BadRequest('Message to be replied not found')
        return SimpleNamespace(message_id=9)
    with pytest.raises(BadRequest):
        await adapter._send_with_dm_topic_reply_anchor_retry(send,
            {'chat_id': 100, 'message_thread_id': 7, 'reply_to_message_id': 8},
            {'strict_scope': True, 'telegram_dm_topic_reply_fallback': True}, 8, 'document')
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_scoped_typing_failure_cannot_widen_or_report_success(adapter):
    calls = []
    async def send(**kw):
        calls.append(kw)
        if kw.get('message_thread_id'):
            raise BadRequest('Message thread not found')
    adapter._bot.send_chat_action = send
    with pytest.raises(BadRequest):
        await adapter.send_typing('100', {'thread_id': '7', 'strict_scope': True,
                                         'telegram_dm_topic_reply_fallback': True})
    assert len(calls) == 1
