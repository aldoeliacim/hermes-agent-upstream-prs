"""A scratch entry that was just created or changed is not idle, whatever mtime it carries (#149841).

``cp -p``/``cp -a``, ``rsync -a``, ``tar x``, ``unzip`` and ``mv`` give what they create the
source's old mtime, and ``rsync -a``/``tar x`` reset an existing target directory's mtime too.
These tests use the wall clock, not ``idle_clock``: under a clock ahead of it no ctime is ever
recent, and the change time is exactly what they check.
"""

import os
import shutil
import time

import pytest

from hermes_constants import get_scratch_dir, prune_scratch_dir
from hermes_constants_scratch import subtree_touch_state
from tests.waiting import wait_until

OLD = time.time() - 30 * 3600


def _old(*paths):
    for path in paths:
        os.utime(path, (OLD, OLD))


def _cutoff_after(probe, *paths):
    """A cutoff later than every timestamp of *paths*, returned once the filesystem's own clock
    has passed it: file timestamps come from a coarse kernel clock that can lag ``time.time()``,
    so a change made right after a bare ``time.time()`` cutoff could still be stamped before it."""
    newest = max(max(st.st_mtime, st.st_ctime) for st in (os.lstat(p) for p in paths))
    cutoff = newest + 0.001

    def passed():
        probe.write_bytes(b"")
        return os.lstat(probe).st_ctime > cutoff

    wait_until(passed, timeout=10, interval=0.005, what="the filesystem clock passing the cutoff")
    return cutoff


def test_a_copy_that_kept_an_old_mtime_survives_the_prune(tmp_path):
    """A file copied with ``cp -p`` and a tree copied with ``cp -a`` arrive with a 30 h old
    mtime; they were created a moment ago and stay. Red when the prune reads mtime alone."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    src = tmp_path / "report.txt"
    src.write_text("work", encoding="utf-8")
    tree = tmp_path / "project"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "notes.md").write_text("notes", encoding="utf-8")
    _old(src, tree / "sub" / "notes.md", tree / "sub", tree)

    shutil.copy2(src, scratch / "report-backup.txt")  # cp -p
    shutil.copytree(tree, scratch / "project-copy")  # cp -a: copy2 per file, copystat per dir
    copies = [scratch / "report-backup.txt", scratch / "project-copy", scratch / "project-copy" / "sub",
              scratch / "project-copy" / "sub" / "notes.md"]
    assert all(os.lstat(p).st_mtime < time.time() - 24 * 3600 for p in copies), "the copies kept old mtimes"

    assert prune_scratch_dir(scratch) == 0
    assert all(p.exists() for p in copies)


@pytest.mark.platforms("linux")  # rename sets the moved inode's ctime on Linux filesystems
def test_a_file_moved_in_with_its_old_mtime_survives_the_prune(tmp_path):
    """``mv`` within one filesystem keeps the file's mtime; the rename itself is the change."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    moved = tmp_path / "moved.txt"
    moved.write_text("moved", encoding="utf-8")
    _old(moved)
    os.rename(moved, scratch / "moved.txt")
    assert os.lstat(scratch / "moved.txt").st_mtime < time.time() - 24 * 3600

    assert prune_scratch_dir(scratch) == 0
    assert (scratch / "moved.txt").exists()


@pytest.mark.platforms("posix")  # st_ctime is the change time only on POSIX
def test_a_copy_into_a_subdirectory_of_an_old_entry_keeps_it(tmp_path):
    """``rsync -a src/ entry/sub/`` (or ``tar x -C entry/sub`` of an archive holding ``./``)
    writes the copies with old mtimes and then resets ``sub``'s own mtime to the source's, and
    ``entry`` is untouched because ``sub`` already existed: every mtime in the tree predates the
    copy. Red when the subtree walk judges nested entries by mtime alone."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "lane"
    sub = entry / "sub"
    sub.mkdir(parents=True)
    (sub / "earlier.txt").write_text("earlier", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "data.csv").write_text("a,b", encoding="utf-8")
    _old(src / "data.csv", src)
    cutoff = _cutoff_after(tmp_path / "probe", entry, sub, sub / "earlier.txt")

    shutil.copy2(src / "data.csv", sub / "data.csv")
    shutil.copystat(src, sub)  # the target directory takes the source's times
    assert all(os.lstat(p).st_mtime < cutoff for p in (entry, sub, sub / "earlier.txt", sub / "data.csv"))

    assert subtree_touch_state(entry, cutoff) == "touched"


@pytest.mark.platforms("posix")  # DirEntry.stat() reports st_nlink, and st_ctime is the change time, only on POSIX
def test_a_hardlinked_files_change_time_does_not_keep_an_idle_clone(tmp_path):
    """A pack hardlinked into a scratch clone changes its ctime whenever the source repo links,
    unlinks or freshens it; like its mtime, that is the live repo's activity, not the clone's.
    The same metadata-only change to a file the clone alone holds is the clone's own."""
    live_pack = tmp_path / "live-repo" / "pack-1.pack"
    live_pack.parent.mkdir()
    live_pack.write_bytes(b"P" * 64)
    clone = get_scratch_dir(tmp_path, prune=False) / "clone"
    (clone / "pack").mkdir(parents=True)
    os.link(live_pack, clone / "pack" / "pack-1.pack")
    own = clone / "README"
    own.write_text("clone", encoding="utf-8")
    cutoff = _cutoff_after(tmp_path / "probe", clone, clone / "pack", clone / "pack" / "pack-1.pack", own)

    os.link(live_pack, tmp_path / "live-repo" / "pack-1.keep")  # the live repo adds a link
    assert os.lstat(live_pack).st_ctime >= cutoff
    assert subtree_touch_state(clone, cutoff) == "idle"

    os.chmod(own, 0o600)
    assert os.lstat(own).st_mtime < cutoff <= os.lstat(own).st_ctime
    assert subtree_touch_state(clone, cutoff) == "touched"
