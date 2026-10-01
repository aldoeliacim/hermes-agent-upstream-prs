"""Image-attachment marker scrub on transcript replay (upstream #46115 port)."""
from agent.memory_manager import sanitize_context, sanitize_transcript_context


def test_strips_image_attached_marker():
    text = "look at this [Image attached at: /home/u/.hermes/image_cache/img_9.jpg]"
    assert sanitize_transcript_context(text) == "look at this"


def test_strips_marker_without_at_keyword():
    text = "hi [Image attached: /tmp/image_cache/img_2.png] there"
    out = sanitize_transcript_context(text)
    assert "image_cache" not in out
    assert "hi" in out and "there" in out


def test_collapses_placeholder_and_blank_lines_only_when_marker_present():
    text = "before\n[screenshot]\n[Image attached at: /x/image_cache/img_1.jpg]\n\n\n\nafter"
    out = sanitize_transcript_context(text)
    assert "image_cache" not in out
    assert "[screenshot]" not in out
    assert "\n\n\n" not in out  # excess blank lines collapsed


def test_no_marker_leaves_text_untouched_by_scrub():
    # Without an attachment marker, the placeholder/blank-line collapse must NOT fire.
    text = "a\n\n\n\nb [screenshot]"
    assert sanitize_transcript_context(text) == sanitize_context(text)


def test_wraps_base_sanitizer():
    # Whatever sanitize_context strips, the transcript variant strips too.
    text = "keep [Image attached at: /image_cache/img_0.jpg]"
    assert sanitize_transcript_context(text) == sanitize_context(text).replace(
        " [Image attached at: /image_cache/img_0.jpg]", ""
    )
