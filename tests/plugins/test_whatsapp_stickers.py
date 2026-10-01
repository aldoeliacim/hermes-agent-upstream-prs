"""WhatsApp sticker handling — inbound classification and outbound delivery.

Three behaviours are pinned here, each of which was broken:

1. An inbound sticker must reach the agent as an IMAGE attachment (pixels),
   not as an opaque "document" note. The adapter previously fell through to
   the generic ``else`` branch, stamping ``media_types=['unknown']``; the
   gateway's ``_event_media_is_image`` then answered False (STICKER is not
   PHOTO), so the sticker was rendered as "[The user sent a document: …]"
   and the model never saw it.

2. ``_bridge_media_type`` must map a sticker send to the bridge's ``sticker``
   mediaType so it arrives as a native sticker bubble rather than an image.

3. The mapping must be *explicit*: an ordinary ``.webp`` attachment stays an
   image. Stickerhood is a caller decision, never inferred from the extension.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gateway.platforms.base import MessageType  # noqa: E402
from gateway.run import (  # noqa: E402
    _event_media_is_audio,
    _event_media_is_image,
    _event_media_is_video,
)
from plugins.platforms.whatsapp.adapter import (  # noqa: E402
    _bridge_media_type,
    _whatsapp_media_type_for_message,
)


class _Event:
    """Minimal stand-in for the gateway MessageEvent media contract."""

    def __init__(self, path, mime, message_type):
        self.media_urls = [path]
        self.media_types = [mime]
        self.message_type = message_type


def test_inbound_sticker_is_classified_as_an_image_attachment(tmp_path):
    """A sticker must carry an image MIME so the model receives pixels."""
    sticker = tmp_path / "sticker_deadbeef.webp"
    sticker.write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 ")

    mime = _whatsapp_media_type_for_message(
        MessageType.STICKER, str(sticker), bridge_mime="image/webp"
    )
    assert mime == "image/webp"

    event = _Event(str(sticker), mime, MessageType.STICKER)
    assert _event_media_is_image(event, 0) is True, (
        "a sticker must route through the image path; otherwise the gateway "
        "emits a document note and the model is blind to it"
    )
    assert _event_media_is_audio(event, 0) is False
    assert _event_media_is_video(event, 0) is False


def test_inbound_sticker_falls_back_to_webp_when_bridge_sends_no_mime(tmp_path):
    """Missing bridge MIME must not collapse to 'unknown' (the old bug)."""
    sticker = tmp_path / "sticker_cafe.webp"
    sticker.write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 ")

    mime = _whatsapp_media_type_for_message(
        MessageType.STICKER, str(sticker), bridge_mime=""
    )
    assert mime == "image/webp"
    assert _event_media_is_image(_Event(str(sticker), mime, MessageType.STICKER), 0)


def test_other_message_types_keep_their_existing_mime_resolution(tmp_path):
    """The sticker fix must not disturb photo/voice/document classification."""
    photo = tmp_path / "img_1.jpg"
    photo.write_bytes(b"\xff\xd8\xff")
    assert (
        _whatsapp_media_type_for_message(MessageType.PHOTO, str(photo), bridge_mime="")
        == "image/jpeg"
    )

    voice = tmp_path / "aud_1.ogg"
    voice.write_bytes(b"OggS")
    assert (
        _whatsapp_media_type_for_message(MessageType.VOICE, str(voice), bridge_mime="")
        == "audio/ogg"
    )

    doc = tmp_path / "doc_1.pdf"
    doc.write_bytes(b"%PDF-")
    assert (
        _whatsapp_media_type_for_message(
            MessageType.DOCUMENT, str(doc), bridge_mime="application/pdf"
        )
        == "application/pdf"
    )


@pytest.mark.parametrize("suffix", [".webp", ".png", ".gif"])
def test_outbound_sticker_flag_maps_to_bridge_sticker_type(suffix):
    """as_sticker routes any source format to the bridge's sticker branch."""
    assert (
        _bridge_media_type(
            f"/tmp/reaction{suffix}", is_voice=False, force_document=False, as_sticker=True
        )
        == "sticker"
    )


def test_plain_webp_attachment_is_still_an_image_not_a_sticker():
    """Stickerhood is explicit — never inferred from the .webp extension."""
    assert (
        _bridge_media_type("/tmp/photo.webp", is_voice=False, force_document=False)
        == "image"
    )


def test_as_document_still_wins_over_as_sticker():
    """[[as_document]] is the escape hatch; it must not be overridden."""
    assert (
        _bridge_media_type(
            "/tmp/reaction.webp",
            is_voice=False,
            force_document=True,
            as_sticker=True,
        )
        == "document"
    )
