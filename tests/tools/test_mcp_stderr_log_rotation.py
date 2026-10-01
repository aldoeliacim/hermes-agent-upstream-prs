"""Tests for the shared MCP stderr log's open-time rotation.

The handle stays open for the life of the process (asyncio wires child fds
to it directly), so rotation can only happen when the handle is first
opened. Without it the file grows unboundedly — no logrotate config knows
about ~/.hermes/logs/mcp-stderr.log.

The state lives in ``tools.mcp_tool_config``: that is where this checkout keeps the
shared stderr log (one handle PER PROFILE HOME, so a multiplexed gateway's secondary
profile logs under its own ``logs/``). ``tools.mcp_tool`` imports the accessor from
there rather than owning the handle.
"""

import importlib

import tools.mcp_tool_config as mcp_config


def _reset_handle():
    """Clear the cached per-profile handles so each test re-opens them."""
    for fh in list(mcp_config._mcp_stderr_log_fh.values()):
        try:
            fh.close()
        except Exception:
            pass
    mcp_config._mcp_stderr_log_fh.clear()


class TestMcpStderrLogRotation:
    def _hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # hermes_constants caches nothing relevant, but reload defensively in
        # case a prior import froze the home path at module level.
        import hermes_constants
        importlib.reload(hermes_constants)
        return tmp_path

    def test_oversized_log_is_rotated_on_open(self, tmp_path, monkeypatch):
        home = self._hermes_home(tmp_path, monkeypatch)
        log_dir = home / "logs"
        log_dir.mkdir(parents=True)
        log_path = log_dir / "mcp-stderr.log"
        log_path.write_bytes(b"x" * (mcp_config._MCP_STDERR_LOG_MAX_BYTES + 1))

        _reset_handle()
        try:
            fh = mcp_config._get_mcp_stderr_log()
            fh.write("fresh line\n")
            fh.flush()

            rotated = log_dir / "mcp-stderr.log.1"
            assert rotated.exists(), "oversized log must move to .log.1"
            assert rotated.stat().st_size > mcp_config._MCP_STDERR_LOG_MAX_BYTES
            # The live file restarted from (near) empty.
            assert log_path.stat().st_size < 1024
        finally:
            _reset_handle()

    def test_small_log_is_left_in_place(self, tmp_path, monkeypatch):
        home = self._hermes_home(tmp_path, monkeypatch)
        log_dir = home / "logs"
        log_dir.mkdir(parents=True)
        log_path = log_dir / "mcp-stderr.log"
        log_path.write_text("existing content\n")

        _reset_handle()
        try:
            fh = mcp_config._get_mcp_stderr_log()
            fh.write("appended\n")
            fh.flush()

            assert not (log_dir / "mcp-stderr.log.1").exists()
            text = log_path.read_text()
            assert "existing content" in text and "appended" in text
        finally:
            _reset_handle()

    def test_missing_log_dir_still_opens(self, tmp_path, monkeypatch):
        self._hermes_home(tmp_path, monkeypatch)

        _reset_handle()
        try:
            fh = mcp_config._get_mcp_stderr_log()
            assert fh is not None
            fh.write("first line\n")
            fh.flush()
        finally:
            _reset_handle()
