"""MEDIA:/FILE: delivery-tag parity — the bug class, made structurally impossible.

Both spellings are documented to the model, and on the Anthropic OAuth lane the
system-prompt sanitizer rewrites every ``MEDIA:`` to ``FILE:`` — so on that lane
``FILE:`` is the ONLY spelling the model is ever shown. Any code path that
recognises one and not the other is a live bug.

The failure mode is silent and asymmetric: the extractors always matched both,
so a ``FILE:`` tag *delivered* the attachment, while the display-strip guards
short-circuited on a bare ``"MEDIA:" not in text`` substring test and left the
raw ``FILE:/home/...`` string in the user's chat next to the delivered file.
The same drift previously double-sent media via the auto-append guard.

Two layers here, because behavioural tests alone let the class come back:

1. ``TestMarkerParity`` — every public entry point must treat the two
   spellings identically. Parametrised over ``MEDIA_TAG_MARKERS``, so adding a
   third spelling to that tuple immediately demands parity everywhere.
2. ``TestNoBareMarkerLiterals`` — a source scan that FAILS on a newly
   introduced ``"MEDIA:" in text`` literal. This is the part that makes the
   class impossible rather than merely fixed: the next person to write the
   substring check gets a red test naming the helper to use instead.
"""

import re
from pathlib import Path

import pytest

from gateway.platforms.base import (
    MEDIA_TAG_MARKERS,
    BasePlatformAdapter,
    _strip_media_tag_directives,
    has_media_marker,
)

# Files that legitimately contain a delivery-tag marker literal in a
# comparison/method-call shape the token scanner would otherwise flag. Narrow by
# construction: canonical producers (``f"MEDIA:{path}"``) and regex definitions
# (``(?:MEDIA|FILE):``) are never in a flagged shape, so they need no entry.
# Empty today; grow only with a per-entry justification, never a subtree exclude.
_SOURCE_SCAN_ALLOWLIST: frozenset = frozenset()

# Not production code; the parity test itself carries the forbidden literals.
_SOURCE_SCAN_SKIP_DIRS = ("__pycache__", "tests")


def _production_gateway_files():
    """Walk gateway/ for every production ``*.py`` (minus the narrow allowlist).

    Discovery, not a hand-maintained list — a hand-list stops covering the
    module where the next bare guard is written, which is exactly how
    run_turn_runner.py kept its ``"MEDIA:" in final_response`` guard.
    """
    root = Path(__file__).resolve().parents[2]
    gateway = root / "gateway"
    for path in sorted(gateway.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if any(part in _SOURCE_SCAN_SKIP_DIRS
               for part in path.relative_to(gateway).parts):
            continue
        if rel in _SOURCE_SCAN_ALLOWLIST:
            continue
        yield rel


_SCANNED_GATEWAY_FILES = tuple(_production_gateway_files())


@pytest.fixture()
def real_pdf(tmp_path):
    p = tmp_path / "report.pdf"
    p.write_bytes(b"%PDF-1.4\n")
    return str(p)


@pytest.fixture()
def real_extensionless(tmp_path):
    p = tmp_path / "Caddyfile"
    p.write_text("localhost\n")
    return str(p)


@pytest.mark.parametrize("marker", MEDIA_TAG_MARKERS)
class TestMarkerParity:
    """Every entry point must behave identically for MEDIA: and FILE:."""

    def test_extract_media_delivers(self, marker, real_pdf):
        media, _ = BasePlatformAdapter.extract_media(f"Here you go.\n{marker}:{real_pdf}")
        assert [p for p, _voice in media] == [real_pdf]

    def test_extract_media_removes_tag_from_body(self, marker, real_pdf):
        _media, cleaned = BasePlatformAdapter.extract_media(
            f"Here you go.\n{marker}:{real_pdf}"
        )
        assert marker not in cleaned
        assert real_pdf not in cleaned

    def test_display_strip_removes_tag(self, marker, real_pdf):
        """THE REPORTED BUG: the raw tag leaked into the visible message."""
        out = BasePlatformAdapter.strip_media_directives_for_display(
            f"Here you go.\n{marker}:{real_pdf}"
        )
        assert marker not in out, f"{marker}: tag leaked into display text"
        assert real_pdf not in out
        assert out.strip() == "Here you go."

    def test_module_level_strip_removes_tag(self, marker, real_pdf):
        out = _strip_media_tag_directives(f"Here you go.\n{marker}:{real_pdf}")
        assert marker not in out
        assert real_pdf not in out

    def test_extensionless_path_delivers(self, marker, real_extensionless):
        media, _ = BasePlatformAdapter.extract_media(f"{marker}:{real_extensionless}")
        assert [p for p, _voice in media] == [real_extensionless]

    def test_has_delivery_directive(self, marker, real_pdf):
        assert BasePlatformAdapter.has_delivery_directive(f"done {marker}:{real_pdf}")

    def test_inline_code_wrapped_tag_delivers(self, marker, real_pdf):
        media, _ = BasePlatformAdapter.extract_media(f"Result: `{marker}:{real_pdf}`")
        assert [p for p, _voice in media] == [real_pdf]

    def test_marker_detected(self, marker, real_pdf):
        assert has_media_marker(f"{marker}:{real_pdf}")


class TestAdjacentMixedTags:
    """Two tags in DIFFERENT spellings, side by side, must both deliver.

    Found by the source-scan gate below, not by hand: the spaced-path recovery
    in ``_match_extensionless_path`` bounded its forward scan on the literal
    ``"MEDIA:"``, so a following ``FILE:`` tag was absorbed into the previous
    path as a filename token — invalidating the candidate and dropping BOTH
    files. Same class as #68773, reached through the alias.
    """

    def test_extensionless_then_other_marker(self, tmp_path):
        a = tmp_path / "Caddyfile"
        a.write_text("localhost\n")
        b = tmp_path / "Dockerfile"
        b.write_text("FROM scratch\n")

        for first, second in ((MEDIA_TAG_MARKERS[0], MEDIA_TAG_MARKERS[1]),
                              (MEDIA_TAG_MARKERS[1], MEDIA_TAG_MARKERS[0])):
            media, _ = BasePlatformAdapter.extract_media(
                f"{first}:{a} {second}:{b}"
            )
            paths = [p for p, _voice in media]
            assert str(a) in paths, f"{first}/{second}: first path dropped"
            assert str(b) in paths, f"{first}/{second}: second path dropped"


class TestDeliveryDirectiveOracle:
    """The auto-append guard's oracle: suppress iff something WILL be delivered."""

    def test_no_directive_when_nothing_deliverable(self):
        assert not BasePlatformAdapter.has_delivery_directive("just a plain reply")

    @pytest.mark.parametrize(
        "prose",
        [
            "see file:///var/log/syslog for details",
            "download from file://host/share",
        ],
    )
    def test_bare_file_url_is_not_a_directive(self, prose):
        """A ``file://`` URL must NOT suppress the auto-append.

        This is why the guard is the extractor itself and not a marker test:
        ``has_media_marker`` matches ``file://`` (it is deliberately a loose
        superset), so using it here would wrongly suppress delivery of a real
        artifact the model forgot to tag.
        """
        assert has_media_marker(prose), "precondition: marker test is a loose superset"
        assert not BasePlatformAdapter.has_delivery_directive(prose)

    def test_nonexistent_extensionless_path_is_not_a_directive(self):
        assert not BasePlatformAdapter.has_delivery_directive(
            "FILE:/nonexistent/definitely/not/here"
        )


class TestNoBareMarkerLiterals:
    """Source scan: a bare ``"MEDIA:" in text`` check must not come back.

    The behavioural tests above prove the CURRENT paths are correct. This one
    stops the class from returning through a path nobody thought to test —
    which is exactly how it came back the second time.

    The scan is token-level (``tokenize``), not regex-over-lines: a comment or
    docstring that *discusses* the old substring check is a single COMMENT /
    STRING token and can never be mistaken for live code, so documenting the
    bug does not trip its own guard.
    """

    # Scan coverage and the narrow allowlist are defined at module level
    # (_production_gateway_files / _SOURCE_SCAN_ALLOWLIST) so the parametrize
    # decorator below can enumerate the walked files at collection time.

    @staticmethod
    def _bare_marker_checks(source: str):
        """Yield (lineno, snippet) for ``"MEDIA:" in x`` / ``.startswith("MEDIA:")``."""
        import ast
        import io
        import tokenize

        toks = [
            t for t in tokenize.generate_tokens(io.StringIO(source).readline)
            if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                              tokenize.INDENT, tokenize.DEDENT)
        ]

        def is_marker_string(tok):
            if tok.type != tokenize.STRING:
                return False
            try:
                value = ast.literal_eval(tok.string)
            except Exception:
                return False
            # Both spellings of the delivery-tag marker are equally forbidden as
            # a bare substring guard: on the OAuth lane FILE: is the only spelling
            # the model is shown, so a ``"FILE:" in text`` check drifts exactly
            # the way ``"MEDIA:" in text`` did — just in the opposite direction.
            return isinstance(value, str) and value.upper() in ("MEDIA:", "FILE:")

        for i, tok in enumerate(toks):
            if not is_marker_string(tok):
                continue
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            prev = toks[i - 1] if i else None
            prev2 = toks[i - 2] if i > 1 else None
            # ``"MEDIA:" in x`` / ``"MEDIA:" not in x`` / ``== "MEDIA:"``
            if nxt and nxt.string in ("in", "not", "==", "!="):
                yield tok.start[0], tok.line.strip()
            # ``x.startswith("MEDIA:")``, ``x.split("MEDIA:")``, ...
            elif prev and prev.string == "(" and prev2 and prev2.string in (
                "startswith", "endswith", "count", "index", "find",
                "split", "replace", "partition",
            ):
                yield tok.start[0], tok.line.strip()

    def test_scanner_detects_the_original_bug(self):
        """The gate must REJECT the exact code that shipped the bug."""
        offenders = list(self._bare_marker_checks(
            'def f(text):\n'
            '    if "MEDIA:" not in text:\n'
            '        return text\n'
        ))
        assert offenders, "scanner failed to flag the original substring check"

    def test_scanner_detects_the_file_spelling(self):
        """The FILE: alias is the ONLY spelling shown on the OAuth lane, so a
        bare ``"FILE:" in text`` guard is the same class of bug and must be
        flagged too — not just the MEDIA: spelling."""
        offenders = list(self._bare_marker_checks(
            'def f(text):\n'
            '    if "FILE:" in text:\n'
            '        return text\n'
        ))
        assert offenders, "scanner failed to flag the FILE: spelling"

    def test_scanner_ignores_prose_about_the_bug(self):
        """A comment/docstring discussing the check is not a violation."""
        offenders = list(self._bare_marker_checks(
            'def f(text):\n'
            '    """Do not use ``"MEDIA:" not in text`` here."""\n'
            '    # the old ``"MEDIA:" in text`` check missed FILE:\n'
            '    return has_media_marker(text)\n'
        ))
        assert offenders == []

    def test_scanner_allows_the_correct_helpers(self):
        offenders = list(self._bare_marker_checks(
            'def f(text):\n'
            '    if not has_media_marker(text):\n'
            '        return text\n'
            '    return BasePlatformAdapter.has_delivery_directive(text)\n'
        ))
        assert offenders == []

    @pytest.mark.parametrize("relpath", _SCANNED_GATEWAY_FILES)
    def test_no_bare_marker_substring_checks(self, relpath):
        root = Path(__file__).resolve().parents[2]
        path = root / relpath
        assert path.exists(), f"scanned file moved: {relpath}"

        offenders = list(self._bare_marker_checks(path.read_text()))
        assert not offenders, (
            f"{relpath} contains a bare 'MEDIA:'/'FILE:' substring check — it "
            f"silently ignores the aliased spelling. On the Anthropic OAuth lane "
            f"FILE: is the ONLY spelling the model is shown, so a bare MEDIA: "
            f"check leaks the raw tag; a bare FILE: check drifts the same way in "
            f"reverse. Use has_media_marker(text) for a presence guard, "
            f"BasePlatformAdapter.has_delivery_directive(text) when you need "
            f"'will this actually deliver?', or a regex built from "
            f"_MEDIA_MARKER_ALTERNATION. Offending lines: {offenders}"
        )

    def test_migrated_sites_are_actually_scanned(self):
        """Regression guard for the SCAN ITSELF: the four originally-migrated
        sites plus the run_turn_runner auto-append guard must all fall inside the
        walked set. A future refactor that narrowed discovery (e.g. re-introduced
        a hand-list) would drop coverage silently; this fails loudly instead."""
        scanned = set(_production_gateway_files())
        for required in (
            "gateway/platforms/base.py",
            "gateway/platforms/api_server.py",
            "gateway/run.py",
            "gateway/media_repair.py",
            "gateway/run_turn_runner.py",
        ):
            assert required in scanned, (
                f"{required} is no longer covered by the source scan — the walk "
                f"or the allowlist regressed"
            )


class TestMarkerSourceOfTruth:
    def test_python_and_typescript_marker_lists_agree(self):
        """Backend MEDIA_TAG_MARKERS must match the desktop renderer's list.

        The desktop renderer keeps its own copy (it cannot import Python), and
        a tag spelling known to one side but not the other renders as literal
        text with no attachment — the same user-visible bug on a different
        surface.
        """
        root = Path(__file__).resolve().parents[2]
        ts = root / "apps/desktop/src/lib/chat-messages.ts"
        if not ts.exists():
            pytest.skip("desktop app not present in this checkout")

        m = re.search(
            r"export const MEDIA_TAG_MARKERS\s*=\s*\[(?P<body>[^\]]*)\]",
            ts.read_text(),
        )
        assert m, "MEDIA_TAG_MARKERS not found in chat-messages.ts"
        ts_markers = tuple(re.findall(r"['\"]([A-Za-z]+)['\"]", m.group("body")))
        assert ts_markers == MEDIA_TAG_MARKERS
