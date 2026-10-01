"""Regression coverage for per-turn setup and native-image attachment seams.

Both cover commit 40bd0cc41e's invariant on the reused gateway worker thread:

* Per-turn ContextVar binds and the approval notify registration live inside ONE
  fail-closed cleanup scope, so a setup exception cannot leak session-key state or
  a notify callback into the pooled worker thread that serves the next turn.
* The native-image attach site logs the *materialized* image-part count, so
  "image received" is distinguishable from "image reached the model".

NOTE ON TOPOLOGY: on this branch the inbound message-source is bound one level up
in ``run_turn.py`` (the reply-gate scope) and propagates into the executor worker
via ``copy_context``; ``_run_conversation_with_approval`` no longer binds it. The
two setup seams it owns here are the session-key bind and the notify registration.
"""

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from tools import approval as approval_mod
from tools import approval_context


@pytest.mark.parametrize("failure_site", ["session_key", "notify_registration"])
def test_turn_setup_failure_restores_all_earlier_bindings(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    """A setup failure must not leak turn state onto the reused worker thread.

    ``session_key``: the session-key bind itself raises → nothing was applied, so
    the outer session key is untouched and no notify callback is registered.

    ``notify_registration``: registration mutates its registry and THEN raises →
    treated as attempted, so the finally unregisters it fail-closed and restores
    the outer session key that was already bound.
    """
    session_key = f"turn-{failure_site}"
    turn_runner = TurnRunner(
        cast(Any, SimpleNamespace()),
        TurnContext(
            source=SimpleNamespace(user_id="turn-user", user_name="turn", is_bot=False),
            message="hello",
            session_id="turn-session",
            session_key=session_key,
        ),
    )

    outer_session_token = approval_context.set_current_session_key("outer-session")
    approval_mod._gateway_notify_cbs.pop(session_key, None)
    try:
        if failure_site == "session_key":

            def fail_session_key(_session_key):
                raise RuntimeError("session-key bind failed")

            monkeypatch.setattr(
                approval_context,
                "set_current_session_key",
                fail_session_key,
            )
        else:
            real_register = approval_mod.register_gateway_notify

            def register_then_fail(key, callback):
                # Mutate the registry FIRST, then raise: a fail-open finally would
                # leave this callback bound on the reused worker thread.
                real_register(key, callback)
                raise RuntimeError("notify registration failed")

            monkeypatch.setattr(
                approval_mod,
                "register_gateway_notify",
                register_then_fail,
            )

        with pytest.raises(RuntimeError):
            turn_runner._run_conversation_with_approval(
                agent=object(),
                agent_history=[],
                observed_group_context=None,
                persist_user_message_override=None,
                persist_user_timestamp_override=None,
            )

        # Outer session key is intact and no per-turn notify callback leaked through.
        assert approval_context.get_current_session_key() == "outer-session"
        assert session_key not in approval_mod._gateway_notify_cbs
    finally:
        approval_mod._gateway_notify_cbs.pop(session_key, None)
        approval_context.reset_current_session_key(outer_session_token)


@pytest.mark.parametrize(
    ("readable_images", "expected_level", "expected_log"),
    [
        (
            2,
            logging.INFO,
            "Native image attachment: attached 2 image part(s) to the user turn "
            "(buffered=3, skipped=1).",
        ),
        (
            0,
            logging.WARNING,
            "Native image attachment: 0 image parts attached despite 3 buffered "
            "path(s) — falling back to text-only turn (the model will NOT see the image).",
        ),
    ],
    ids=["attached-parts-info", "zero-parts-warning"],
)
def test_native_image_attach_site_logs_materialized_part_count(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
    readable_images: int,
    expected_level: int,
    expected_log: str,
) -> None:
    """Attach-site logs report what reached the turn, not only what was buffered."""
    image_paths = []
    for index in range(readable_images):
        image_path = tmp_path / f"image-{index}.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        image_paths.append(str(image_path))
    image_paths.extend(
        str(tmp_path / f"missing-{index}.png") for index in range(3 - readable_images)
    )

    gateway_runner = SimpleNamespace(
        _consume_pending_native_image_paths=lambda _session_key: image_paths,
    )
    turn_runner = TurnRunner(
        cast(Any, gateway_runner),
        TurnContext(
            message="inspect these",
            session_key="native-image-turn",
        ),
    )
    caplog.set_level(logging.INFO, logger="gateway.run")

    run_message = turn_runner._native_image_run_message()

    image_parts = (
        [part for part in run_message if part.get("type") == "image_url"]
        if isinstance(run_message, list)
        else []
    )
    assert len(image_parts) == readable_images
    if readable_images == 0:
        assert run_message == "inspect these"
    else:
        assert run_message != "inspect these"
    assert any(
        record.levelno == expected_level and record.getMessage() == expected_log
        for record in caplog.records
    )
