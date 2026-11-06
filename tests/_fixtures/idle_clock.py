"""Idle without backdating: a clock ahead of the wall clock for the prunes' idle cutoffs.

The scratch prune (``hermes_constants_scratch``) and the terminal temp cleanup
(``tools.environments.local``) count an entry as changed at the later of its mtime and its
ctime. Backdating with ``os.utime`` therefore no longer ages anything: utime sets the ctime
to now, exactly as ``cp -p`` does for the copy it makes (#149841). Instead the cutoff's clock
moves forward: everything a test creates is then past the 24 h window, and a write meant to
count as recent is stamped at the advanced time with :meth:`IdleClock.touch`.

Only the ``time`` binding of the module whose cutoff is driven changes; process ancestry,
the spawn ledger and the stuck-entry records keep the wall clock. Tests that need the wall
clock itself to move (stuck-record expiry) patch ``time.time`` globally on top: this clock
follows it.
"""

from __future__ import annotations

import os
import time
import types

import pytest


class IdleClock:
    AHEAD_S = 30 * 3600

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch

    def now(self) -> float:
        return time.time() + self.AHEAD_S

    def touch(self, *paths: os.PathLike[str] | str) -> None:
        """Stamp *paths* as written at the advanced now (mtime; the ctime stays the wall
        clock's, which the advanced cutoff counts as old)."""
        stamp = self.now()
        for path in paths:
            os.utime(path, (stamp, stamp))

    def drive(self, module: types.ModuleType) -> None:
        """Make *module*'s ``time.time()`` this clock; every other ``time`` function is the real one."""
        ahead = types.SimpleNamespace(**{n: getattr(time, n) for n in dir(time) if not n.startswith("_")})
        ahead.time = self.now
        self._monkeypatch.setattr(module, "time", ahead)

    @classmethod
    def child_preamble(cls) -> str:
        """Python statements that do the same for ``hermes_constants_scratch`` in another process."""
        return ("import time as _t, types as _ty, hermes_constants_scratch as _r; "
                "_a = _ty.SimpleNamespace(**{n: getattr(_t, n) for n in dir(_t) if not n.startswith('_')}); "
                f"_a.time = lambda: _t.time() + {cls.AHEAD_S}; _r.time = _a; ")


@pytest.fixture
def idle_clock(monkeypatch: pytest.MonkeyPatch) -> IdleClock:
    """The scratch prune's cutoff clock, 30 h ahead (see the module docstring)."""
    import hermes_constants_scratch

    clock = IdleClock(monkeypatch)
    clock.drive(hermes_constants_scratch)
    return clock
