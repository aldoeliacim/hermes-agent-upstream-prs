"""Recovery notes preserve unfinished work, not blind action replay."""
import pytest
from gateway.run import _build_interruption_system_note, _build_tool_tail_system_note, build_resume_recovery_note

@pytest.mark.parametrize("note", [
    _build_tool_tail_system_note(),
    _build_interruption_system_note("restart_timeout", True),
    _build_interruption_system_note("restart_timeout", True, group_mode=True),
    build_resume_recovery_note("restart_timeout", "Also check the worker model."),
])
def test_new_input_does_not_cancel_unfinished_authorized_work(note):
    assert "IGNORE those pending results" not in note
    assert "skip any unfinished" not in note
    assert "unfinished authorized work" in note
    assert "cancels, pauses, or replaces" in note
    assert "Do NOT re-execute" in note

@pytest.mark.parametrize("note", [
    _build_interruption_system_note("restart_timeout", False),
    _build_interruption_system_note("restart_timeout", False, group_mode=True),
    build_resume_recovery_note("restart_timeout", ""),
])
def test_synthetic_startup_keeps_pending_work_without_authorizing_resumption(note):
    assert "continue the remaining authorized work" not in note
    assert "unfinished work remains pending" in note
    assert "wait for real user direction" in note


def test_recovery_keeps_actual_user_input_verbatim():
    message = "Stop the deployment. Only inspect the logs."
    note = build_resume_recovery_note("restart_timeout", message)
    assert note.endswith("\n\n" + message)
    assert "cancels, pauses, or replaces" in note

def test_group_recovery_still_forbids_public_lifecycle_narration():
    note = _build_interruption_system_note("restart_timeout", True, group_mode=True)
    assert "do NOT narrate" in note
    assert "into the group" in note
