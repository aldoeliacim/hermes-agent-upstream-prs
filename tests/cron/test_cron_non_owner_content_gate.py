"""Cron delivery to a NON-owner target honours the same gates as the reply path.

Incident: a recurring status job with ``deliver: origin`` whose origin was a contact's DM
ran on a route that returned the internal compaction handoff (``<summary>`` plus a tool
call written as text, containing internal service commands). Interactive replies and
``send_message(target="current")`` refuse that content; the cron delivery door did not,
so the contact received it and it was mirrored into the contact's transcript.

Driven through the REAL ``run_one_job`` -> ``_deliver_result`` path down to the platform
sender seam (same harness as test_cron_failure_deliver.py).
"""
from __future__ import annotations

import pytest

import cron.scheduler as s
from cron import scheduler_delivery as sd

# Reuse the real-delivery harness (fake platform sender, stubbed bookkeeping).
from tests.cron.test_cron_failure_deliver import run_env  # noqa: F401  (pytest fixture)

CRON_ARTIFACT = """<summary>
## Overview
The cron job is designed to monitor the progress of the series and its audio track.
## Next Steps
Check season completion.
</summary>

<run_subagent>
{"CommandLine":"curl -s http://127.0.0.1:8989/api/v3/series | jq .","explanation":"Checking season completion on Sonarr."}
</run_subagent>"""

CLEAN_STATUS = "¡Listo, Robin! Ya quedó la temporada 2 en latino. Ya la puedes ver."


def _run(monkeypatch, final, deliver="slack:D0CONTACT"):
    monkeypatch.setattr(s, "run_job", lambda job, **_kw: (True, "raw", final, None))
    return s.run_one_job({"id": "jc", "name": "status", "deliver": deliver})


def test_internal_artifact_is_withheld_from_non_owner_cron_target(run_env, monkeypatch):
    _run(monkeypatch, CRON_ARTIFACT)

    sent = [c["message"] for c in run_env["send"] if c["chat_id"] == "D0CONTACT"]
    assert sent, "the contact still gets a (neutral) notice"
    assert all("<summary>" not in m and "run_subagent" not in m and "Sonarr" not in m for m in sent)
    assert sent == [sd._NON_OWNER_CRON_FAILURE]
    args, _kw = run_env["marked"][0]
    assert "withheld" in str(args) or "withheld" in str(_kw)


def test_clean_output_still_reaches_non_owner_cron_target(run_env, monkeypatch):
    _run(monkeypatch, CLEAN_STATUS)

    sent = [c["message"] for c in run_env["send"] if c["chat_id"] == "D0CONTACT"]
    assert sent == [CLEAN_STATUS]


def test_owner_home_still_receives_the_raw_output(run_env, monkeypatch):
    """The gate is for non-owner targets only: the owner's home channel is unchanged."""
    monkeypatch.setattr(
        sd, "_get_home_target_chat_id", lambda platform: "D0MAIN" if platform == "slack" else ""
    )
    monkeypatch.setattr(sd, "_get_home_target_thread_id", lambda _platform: "")
    _run(monkeypatch, CRON_ARTIFACT, deliver="slack:D0MAIN")

    sent = [c["message"] for c in run_env["send"] if c["chat_id"] == "D0MAIN"]
    assert sent and all(m != sd._NON_OWNER_CRON_FAILURE for m in sent)


@pytest.mark.parametrize(
    "content,expected",
    [
        (CRON_ARTIFACT, "summary_envelope"),
        ('Revisando.\n<execute_bash>\n{"CommandLine":"ls /srv"}\n</execute_bash>', "text_tool_markup"),
        ("Ya estaba registrada (Radarr) pero sin archivo en el servidor.", "backend_substance_disclosure"),
        (CLEAN_STATUS, None),
    ],
)
def test_withhold_reason_uses_the_reply_path_classifiers(content, expected):
    assert sd._non_owner_withhold_reason(content) == expected
