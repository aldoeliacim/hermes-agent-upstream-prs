from types import SimpleNamespace

from gateway.platforms.base import MessageEvent, MessageType, Platform
from gateway.run import GatewayRunner


def _event(message_type, media_urls, media_types):
    return MessageEvent(
        text="Puedes quitarme los granitos de la boca",
        message_type=message_type,
        source=SimpleNamespace(platform=Platform.WHATSAPP, chat_id="800000000000021@lid"),
        media_urls=media_urls,
        media_types=media_types,
    )


def test_video_attachment_is_not_silently_dropped_from_model_context(tmp_path):
    video = tmp_path / "vid_d60132fd5319.mp4"
    video.write_bytes(b"video")

    context = GatewayRunner._prepend_inbound_media_file_notes("edit it", [], [str(video)])

    assert "[The user sent a video attachment: 'vid_d60132fd5319.mp4'" in context
    assert str(video) in context
    assert context.endswith("edit it")


def test_unknown_binary_attachment_is_not_silently_dropped_from_model_context(tmp_path):
    binary = tmp_path / "photo-edit-reference.bin"
    binary.write_bytes(b"binary")
    event = _event(MessageType.TEXT, [str(binary)], ["application/octet-stream"])

    context = GatewayRunner._prepend_inbound_document_notes(event, "inspect it")

    assert "photo-edit-reference.bin" in context
    assert str(binary) in context
    assert context.endswith("inspect it")
