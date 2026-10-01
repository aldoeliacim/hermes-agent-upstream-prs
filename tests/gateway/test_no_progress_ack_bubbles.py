"""#921 — the edited/rewritten in-chat progress-acknowledgement bubble is DELETED.

This file proves the *class* of bug is impossible, not just that one code path is
quiet. Two independent guards:

1. Behavioural: a multi-tool turn against a fake adapter must perform ZERO
   ``send_message`` and ZERO ``edit_message`` calls *before* the single final
   reply. Parametrised across group + DM, across every legacy
   ``display.tool_progress`` value (off/new/all/verbose/plain/log), and across
   streaming + non-streaming. The only ``send`` allowed is the final reply
   itself; ``edit_message`` must never be called at all on a plain adapter.

2. Source-scan: no progress/heartbeat code path in ``gateway/**/*.py`` may call
   ``.edit_message`` again. A narrow, justified allowlist covers the Slack
   native task-card fallback (which legitimately edits a *card*, not the ack
   bubble). The scan is proven RED by the accompanying meta-test that shows a
   reintroduced progress-path edit would trip it.

If someone reintroduces the ack bubble, at least one of these fails.
"""

from __future__ import annotations

import ast
import importlib
import sys
import time
import types
from pathlib import Path

import pytest

from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Behavioural guard
# ---------------------------------------------------------------------------


class _CaptureAdapter(BasePlatformAdapter):
    """Records every send / edit / typing call without touching a platform API."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self.typing: list[dict] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {"chat_id": chat_id, "content": content, "reply_to": reply_to, "metadata": metadata}
        )
        return SendResult(success=True, message_id=f"msg-{len(self.sent)}")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        # A plain (non-native-card) adapter must NEVER receive an edit: that was
        # the deleted progress-ack-bubble behaviour.
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": {"stopped": True}})

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class _MultiToolAgent:
    """Fires several tool-progress callbacks mid-turn, then returns a real final
    reply. Under #921 none of those callbacks may produce a chat send/edit."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        cb = self.tool_progress_callback
        if cb is not None:
            cb("tool.started", "terminal", "pwd", {})
            time.sleep(0.02)
            cb("tool.started", "browser_navigate", "https://example.com", {})
            time.sleep(0.02)
            cb("tool.started", "read_file", "/etc/hosts", {})
            time.sleep(0.02)
        # Deliver the final reply through streaming if the gateway wired a
        # delta callback for this turn; otherwise the runner sends it directly.
        if self.stream_delta_callback is not None:
            self.stream_delta_callback("done")
        return {"final_response": "done", "messages": [], "api_calls": 1}


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.session_store = types.SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = types.SimpleNamespace(loaded_hooks=False)
    runner.config = types.SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


# Every legacy display.tool_progress value, including the retired text modes.
_LEGACY_MODES = ["off", "new", "all", "verbose", "plain", "log"]


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["group", "dm"])
@pytest.mark.parametrize("mode", _LEGACY_MODES)
@pytest.mark.parametrize("streaming", [False, True])
async def test_multi_tool_turn_emits_no_progress_bubble(
    monkeypatch, tmp_path, chat_type, mode, streaming
):
    """A multi-tool turn posts EXACTLY the final reply and edits nothing —
    no progress ack bubble in any legacy mode, group or DM, streaming or not."""
    import yaml

    config_data = {
        "display": {"tool_progress": mode},
        "streaming": {"enabled": streaming},
    }
    (tmp_path / "config.yaml").write_text(yaml.dump(config_data), encoding="utf-8")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _MultiToolAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = _CaptureAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    runner.config.streaming = StreamingConfig.from_dict({"enabled": streaming})
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    chat_id = "-1001" if chat_type == "group" else "42"
    thread_id = "topic-9" if chat_type == "group" else None
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=chat_type,
        thread_id=thread_id,
    )
    session_key = f"agent:main:telegram:{chat_type}:{chat_id}"
    if thread_id:
        session_key = f"{session_key}:{thread_id}"

    result = await runner._run_agent(
        message="do three things",
        context_prompt="",
        history=[],
        source=source,
        session_id=f"sess-{chat_type}-{mode}-{streaming}",
        session_key=session_key,
    )

    assert result["final_response"] == "done"
    # Depending on the delivery lane the runner either sends the final reply
    # itself (streaming/interim finalisation) or hands ``final_response`` back to
    # the caller to send. EITHER way, the runner must emit NO progress bubble:
    # every send it made (if any) is the final reply verbatim — never an interim
    # "🔍 running…"-style ack, and never more than the single final message.
    sent_contents = [c["content"] for c in adapter.sent]
    assert sent_contents in ([], ["done"]), sent_contents
    # No bubble was ever edited/rewritten (that whole machinery is gone) — an
    # edit on a plain adapter would only ever have been the ack-bubble rewrite.
    assert adapter.edits == [], adapter.edits


# ---------------------------------------------------------------------------
# Source-scan guard
# ---------------------------------------------------------------------------

_GATEWAY_DIR = Path(importlib.import_module("gateway").__file__).parent

# Functions whose bodies MAY call ``.edit_message`` for a legitimate,
# non-progress-ack reason. Narrow and justified:
#   * Slack/Buzz native task-card fallback edits a *card* message, not the ack
#     bubble (``_send_progress_text`` carries CARD content — the KEEP fallback).
#   * Streaming finalisation edits the streamed FINAL reply in place (e.g. a
#     transform_llm_output hook rewrote it after streaming) — also not the ack.
_EDIT_ALLOWLIST = {
    "_send_progress_text",  # native task-card text fallback (card, not ack bubble)
}

# Name fragments that mark a function as a progress / heartbeat ACK path. If a
# function whose name matches one of these calls ``.edit_message`` (and it is not
# on the allowlist), the deleted bubble has crept back.
_PROGRESS_NAME_MARKERS = (
    "progress",
    "heartbeat",
    "_edit_progress",
    "_flush_progress",
    "_progress_emit",
    "_progress_build",
)


def _iter_gateway_py_files():
    for path in sorted(_GATEWAY_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _functions_calling_edit_message(tree: ast.AST):
    """Yield (func_name, lineno) for every function/method whose body calls
    something.edit_message(...)."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "edit_message"
            ):
                yield node.name, sub.lineno
                break


def test_no_progress_path_calls_edit_message():
    """Static guard: no progress/heartbeat function in gateway/**/*.py may call
    ``.edit_message`` (except the justified native-card fallback allowlist)."""
    offenders: list[str] = []
    for path in _iter_gateway_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for func_name, lineno in _functions_calling_edit_message(tree):
            if func_name in _EDIT_ALLOWLIST:
                continue
            low = func_name.lower()
            if any(marker in low for marker in _PROGRESS_NAME_MARKERS):
                offenders.append(f"{path.relative_to(_GATEWAY_DIR.parent)}:{lineno} {func_name}")
    assert not offenders, (
        "A progress/heartbeat path calls edit_message again — the deleted "
        "ack bubble is back:\n  " + "\n  ".join(offenders)
    )


def test_source_scan_would_catch_a_reintroduced_progress_edit():
    """Meta-test: prove the scan is RED, not vacuously green. A synthetic
    progress function that calls ``.edit_message`` must be flagged by the same
    predicate the real scan uses."""
    reintroduced = (
        "class X:\n"
        "    async def _emit_progress_bubble(self, st):\n"
        "        await st.adapter.edit_message(st.chat_id, st.msg_id, 'working...')\n"
    )
    tree = ast.parse(reintroduced)
    flagged = []
    for func_name, lineno in _functions_calling_edit_message(tree):
        if func_name in _EDIT_ALLOWLIST:
            continue
        low = func_name.lower()
        if any(marker in low for marker in _PROGRESS_NAME_MARKERS):
            flagged.append(func_name)
    assert flagged == ["_emit_progress_bubble"], flagged
