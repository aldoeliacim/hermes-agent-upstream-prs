"""WhatsApp sticker directive: ``_bridge_media_type`` honors ``as_sticker``.

The ``[[as_sticker]]`` directive is an explicit caller decision (not an
extension sniff), so an ordinary ``.webp`` a user attaches still renders as an
image bubble. ``force_document`` (the ``[[as_document]]`` escape hatch) wins
over ``as_sticker``, matching the existing precedence.
"""
import importlib


adapter = importlib.import_module("plugins.platforms.whatsapp.adapter")
_bridge_media_type = adapter._bridge_media_type


def test_as_sticker_forces_sticker_type():
    assert _bridge_media_type("cat.webp", False, False, as_sticker=True) == "sticker"
    # Even a non-webp path becomes a sticker when the directive is explicit.
    assert _bridge_media_type("cat.png", False, False, as_sticker=True) == "sticker"


def test_force_document_wins_over_sticker():
    assert _bridge_media_type("cat.webp", False, True, as_sticker=True) == "document"


def test_webp_without_directive_stays_image():
    # No implicit sticker sniff: a plain .webp is an image bubble.
    assert _bridge_media_type("cat.webp", False, False) == "image"


def test_voice_and_extension_paths_unchanged():
    assert _bridge_media_type("note.ogg", True, False) == "audio"
    assert _bridge_media_type("clip.mp4", False, False) == "video"
    assert _bridge_media_type("data.bin", False, False) == "document"


def test_as_sticker_defaults_false():
    # Back-compat: callers that don't pass the kwarg get the legacy behavior.
    assert _bridge_media_type("cat.webp", False, False) == "image"
