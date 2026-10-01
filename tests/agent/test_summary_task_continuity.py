"""A summary is not an order, but compaction is not task completion."""
from agent.context_compressor import SUMMARY_PREFIX


def test_compaction_does_not_declare_unfinished_work_completed():
    assert "they were already addressed" not in SUMMARY_PREFIX
    assert "unfinished authorized work" in SUMMARY_PREFIX
    assert "additional request" in SUMMARY_PREFIX


def test_compaction_preserves_cancellation_and_no_standalone_replay():
    assert "cancels, pauses, or replaces" in SUMMARY_PREFIX
    assert "if no user message appears after this summary, do nothing" in SUMMARY_PREFIX.lower()
    assert "not permission to start or resume work" in SUMMARY_PREFIX
