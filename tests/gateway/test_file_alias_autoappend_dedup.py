"""Regression test for the FILE:-alias double-send bug (2026-06-11).

Symptom: an image_generate result that the model also referenced in its text
with a ``FILE:<path>`` delivery tag was delivered TWICE on WhatsApp.

Root cause: the auto-append guard in ``gateway/run.py`` that prevents the
gateway from re-appending a media tag the model already emitted checked only
for the literal substring ``"MEDIA:"``. ``FILE:`` is a documented alias for
``MEDIA:`` (cp 3cdc07659), so a response that already carried ``FILE:<path>``
slipped past the guard, the auto-append re-added the same path as ``MEDIA:``,
and the dispatcher sent the artifact twice.

Fix: the guard now uses the shared anchored ``MEDIA_TAG_CLEANUP_RE`` (which
matches both ``MEDIA:`` and ``FILE:`` followed by a real deliverable path) so a
pre-existing delivery directive in EITHER form suppresses the auto-append —
while a bare ``file://`` URL sitting in prose is NOT mistaken for a directive.
"""

from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE


_IMG = "/home/user/.hermes/cache/images/generated_artifact.png"


def _guard_skips_autoappend(final_response: str) -> bool:
    """Mirror the production guard: True => auto-append is SUPPRESSED."""
    return bool(MEDIA_TAG_CLEANUP_RE.search(final_response))


def test_file_alias_tag_suppresses_autoappend():
    # The model already wrote a FILE: tag for the artifact it produced.
    resp = f"Here is your image.\n\nFILE:{_IMG}\n\nEnjoy."
    assert _guard_skips_autoappend(resp) is True, (
        "A response already carrying a FILE: delivery tag must suppress the "
        "auto-append, or the same artifact is sent twice."
    )


def test_media_tag_suppresses_autoappend():
    resp = f"Here is your image.\n\nMEDIA:{_IMG}\n\nEnjoy."
    assert _guard_skips_autoappend(resp) is True


def test_no_directive_allows_autoappend():
    # A plain text reply with no delivery tag must NOT suppress the auto-append
    # (otherwise tool-produced artifacts the model forgot to tag never deliver).
    resp = "Here is the analysis you asked for. No attachments."
    assert _guard_skips_autoappend(resp) is False


def test_bare_file_url_in_prose_does_not_suppress():
    # A bare file:// URL mentioned in prose is NOT a delivery directive and must
    # not be mistaken for one (it has no MEDIA:/FILE: prefix + deliverable path
    # grammar match).
    resp = "The log was at file:///var/log/app.log but that's not an attachment."
    assert _guard_skips_autoappend(resp) is False


def test_old_substring_guard_would_have_missed_file_alias():
    # Documents the exact defect: the OLD guard `"MEDIA:" not in resp` would have
    # let the auto-append run on a FILE:-tagged response (the double-send), while
    # the new regex guard correctly catches it.
    resp = f"Here is your image.\n\nFILE:{_IMG}\n\nEnjoy."
    old_guard_runs_autoappend = "MEDIA:" not in resp          # True == bug
    new_guard_runs_autoappend = not _guard_skips_autoappend(resp)
    assert old_guard_runs_autoappend is True                  # the latent bug
    assert new_guard_runs_autoappend is False                 # fixed