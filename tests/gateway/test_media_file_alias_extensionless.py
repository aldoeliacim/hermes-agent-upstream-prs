"""``FILE:`` must be a full alias of ``MEDIA:`` on the extensionless path too.

MEDIA_TAG_CLEANUP_RE (known extensions) accepted ``(?:MEDIA|FILE):`` but
MEDIA_EXTENSIONLESS_TAG_RE only accepted ``MEDIA:``. So a file with an
unknown/absent extension tagged ``FILE:`` extracted as nothing and the raw
tag was printed to the user as text — while the same path tagged ``MEDIA:``
delivered fine. Asymmetry between the two patterns, not a bad path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.platforms.base import BasePlatformAdapter  # noqa: E402


def _extract(text):
    media, cleaned = BasePlatformAdapter.extract_media(text)
    return [m[0] for m in media], cleaned


def test_file_and_media_agree_on_unknown_extension(tmp_path):
    """A ``.30`` suffix is not a deliverable extension -> extensionless path."""
    rom = tmp_path / "ROMD82T4.30"
    rom.write_bytes(b"\xff" * 16)

    via_media, cleaned_media = _extract(f"a\n\nMEDIA:{rom}\n\nb")
    via_file, cleaned_file = _extract(f"a\n\nFILE:{rom}\n\nb")

    assert via_media == [str(rom)]
    assert via_file == via_media, "FILE: must alias MEDIA: for unknown extensions"
    assert "FILE:" not in cleaned_file and "MEDIA:" not in cleaned_media


def test_file_alias_on_extensionless_basename(tmp_path):
    """No extension at all (Makefile-style) — same requirement."""
    f = tmp_path / "Dockerfile"
    f.write_text("FROM scratch\n")

    assert _extract(f"x\n\nFILE:{f}\n\ny")[0] == [str(f)]


def test_nonexistent_path_still_stays_visible(tmp_path):
    """Validation remains the oracle: a bogus path is not silently swallowed."""
    missing = tmp_path / "NOPE1.99"
    media, cleaned = _extract(f"x\n\nFILE:{missing}\n\ny")

    assert media == []
    assert f"FILE:{missing}" in cleaned


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        test_file_and_media_agree_on_unknown_extension(p)
        test_file_alias_on_extensionless_basename(p)
        test_nonexistent_path_still_stays_visible(p)
    print("ok")
