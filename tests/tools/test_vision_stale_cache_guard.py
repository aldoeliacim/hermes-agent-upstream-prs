"""Direct unit tests for the vision current-turn inbound-image staleness guard.

These exercise the guard's PURE logic — no gateway, no model call. The property
under test: an inbound-image-cache path is refused iff (a) a gateway turn context
is present AND (b) the path's basename is not among the images attached this turn.
Everything else — no turn context, paths outside the cache dir — is left alone.
"""
from pathlib import Path

import pytest

import tools.vision_tools as vt


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point the guard at a temp inbound-cache dir and yield it."""
    d = (tmp_path / "image_cache").resolve()
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(vt, "_inbound_image_cache_dir", lambda: d)
    return d


def _touch(d: Path, name: str) -> Path:
    p = d / name
    p.write_bytes(b"\x00")
    return p.resolve()


def test_guard_is_inert_without_turn_context(cache_dir):
    """No gateway turn context (contextvar None) => nothing is ever stale."""
    img = _touch(cache_dir, "img_abc.jpg")
    # No set_current_turn_image_paths() call — contextvar defaults to None.
    assert vt._is_stale_inbound_cache_path(img) is False


def test_path_outside_cache_dir_is_never_stale(cache_dir, tmp_path):
    outside = (tmp_path / "elsewhere" / "photo.png")
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"\x00")
    token = vt.set_current_turn_image_paths([])  # empty turn: nothing attached
    try:
        # Outside the inbound cache dir => out of scope, not stale.
        assert vt._is_stale_inbound_cache_path(outside.resolve()) is False
    finally:
        vt.reset_current_turn_image_paths(token)


def test_cache_path_attached_this_turn_is_fresh(cache_dir):
    fresh = _touch(cache_dir, "img_fresh.jpg")
    token = vt.set_current_turn_image_paths([str(fresh)])
    try:
        assert vt._is_stale_inbound_cache_path(fresh) is False
    finally:
        vt.reset_current_turn_image_paths(token)


def test_cache_path_not_attached_this_turn_is_stale(cache_dir):
    fresh = _touch(cache_dir, "img_fresh.jpg")
    old = _touch(cache_dir, "img_old_from_yesterday.jpg")
    token = vt.set_current_turn_image_paths([str(fresh)])
    try:
        # A cache image the current turn did NOT attach => stale, must be refused.
        assert vt._is_stale_inbound_cache_path(old) is True
    finally:
        vt.reset_current_turn_image_paths(token)


def test_empty_turn_makes_every_cache_path_stale(cache_dir):
    """An empty attached-list means the turn brought no images => all cache stale."""
    some = _touch(cache_dir, "img_whatever.jpg")
    token = vt.set_current_turn_image_paths([])
    try:
        assert vt._is_stale_inbound_cache_path(some) is True
    finally:
        vt.reset_current_turn_image_paths(token)


def test_set_normalizes_to_basename(cache_dir):
    """Attached paths are matched by basename, so absolute vs relative is irrelevant."""
    fresh = _touch(cache_dir, "img_norm.jpg")
    # Pass a path with redundant components; guard should still treat it as fresh.
    token = vt.set_current_turn_image_paths([f"{cache_dir}/./img_norm.jpg"])
    try:
        assert vt._is_stale_inbound_cache_path(fresh) is False
    finally:
        vt.reset_current_turn_image_paths(token)


def test_reset_restores_inert_state(cache_dir):
    old = _touch(cache_dir, "img_reset.jpg")
    token = vt.set_current_turn_image_paths([])
    assert vt._is_stale_inbound_cache_path(old) is True
    vt.reset_current_turn_image_paths(token)
    # After reset the contextvar is None again => inert.
    assert vt._is_stale_inbound_cache_path(old) is False
