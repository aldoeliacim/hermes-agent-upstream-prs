"""#77xxx: a REMOTE-backend terminal command that invokes an interpreter by
absolute path must not crash the guard with ValueError: embedded null byte.

The local reader already skips binaries (NUL byte in the first chunk), but the
remote reader had no such filter: the fetched machine code was decoded as text,
re-tokenized as shell, and a junk "path" containing a NUL reached
Path.expanduser(). That raised ValueError out of the guard and took the entire
terminal tool down -- on an ordinary command whose only sin was naming an
interpreter by absolute path.

Observed as: `venv/bin/python script.py args` failing with
"Failed to execute command: embedded null byte" on an ssh-backed session.
"""
import pytest

from cron.lifecycle_guard import (
    contains_gateway_lifecycle_command_or_referenced_script,
)


def _binary_reader(_path: str) -> str:
    """Stand in for a remote backend handing back a BINARY's decoded bytes."""
    return "\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00/lib/ld.so\x00"


def test_remote_binary_contents_do_not_crash_guard():
    result = contains_gateway_lifecycle_command_or_referenced_script(
        "/opt/venv/bin/python run.py /tmp/in.jsonl host",
        cwd="/tmp",
        read_remote_script=_binary_reader,
    )
    assert result is False


def test_nul_in_token_does_not_crash_path_resolution():
    """A NUL byte in a token must be swallowed by the resolver, not raised.

    The resolver is now ``_resolve_script_path(script_path)`` (the ``(path, cwd)`` form
    was retired when relative script names moved under ``<HERMES_HOME>/scripts``). A NUL
    can never be part of a real path, so the contract is "no path to scan" — None —
    rather than a ValueError escaping into the terminal tool.
    """
    from cron.lifecycle_guard import _resolve_script_path

    assert _resolve_script_path("/usr/bin/py\x00thon") is None


def test_nul_token_does_not_crash_the_public_guard():
    """The end-to-end path: a NUL-bearing command returns a verdict, never raises."""
    assert contains_gateway_lifecycle_command_or_referenced_script(
        "/usr/bin/py\x00thon run.py",
        cwd="/tmp",
    ) is False


def test_remote_shell_script_is_still_scanned():
    """The binary filter must not blind the guard to genuine remote scripts."""

    def _script_reader(_path: str) -> str:
        return "#!/bin/bash\nhermes gateway restart\n"

    result = contains_gateway_lifecycle_command_or_referenced_script(
        "/opt/deploy/wrapper.sh",
        cwd="/tmp",
        read_remote_script=_script_reader,
    )
    assert result is True
