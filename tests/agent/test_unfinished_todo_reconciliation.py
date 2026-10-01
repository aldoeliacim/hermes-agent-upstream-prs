"""Current-turn todo reconciliation through real loop, dispatch and SessionDB.

Only the provider is scripted: no model/network calls and no fake todo results.
"""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent
from tools.todo_tool import TODO_SCHEMA, TodoStore


PLAN = [
    {"id": "diagnose", "content": "Diagnose the bug", "status": "in_progress"},
    {"id": "fix", "content": "Fix the bug", "status": "pending"},
    {"id": "verify", "content": "Verify the fix", "status": "pending"},
]
SIDE_ANSWER = "The setting is in config.yaml."
FINAL = "The fix is implemented and the regression test passes."


def response(content=SIDE_ANSWER, *, todos=None, merge=False, call_id="plan"):
    calls = None
    if todos is not None:
        calls = [SimpleNamespace(
            id=call_id, type="function",
            function=SimpleNamespace(name="todo_list", arguments=json.dumps({
                "todos": todos, "merge": merge,
            })),
        )]
        content = ""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=calls, phase="final_answer"),
            finish_reason="tool_calls" if calls else "stop",
        )], model="test/model", usage=None,
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    # Patch the DEFINING modules, never the run_agent compat pointers: those are kept only
    # for external plugins, and patching them leaves the real construction site untouched —
    # the client then isn't a Mock, the loop picks the streaming seam instead of the one this
    # test scripts, and every case dials the live base_url until it times out.
    with (
        patch("model_tools.get_tool_definitions", return_value=[{"type": "function", "function": TODO_SCHEMA}]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
        patch("agent.model_metadata.get_model_context_length", return_value=256000),
        patch("agent.title_generator.maybe_auto_title"),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
        patch("hermes_cli.plugins.has_hook", return_value=False),
    ):
        instance = AIAgent(
            session_id="todo-reconcile-test", api_key="test-key",
            base_url="https://example.invalid/v1", provider="openai-compat",
            model="test/model", max_iterations=8, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
        instance._cached_system_prompt = "Stable test prompt."
        instance._intent_ack_continuation = False
        instance._stall_guards = True
        instance._session_json_enabled = False
        instance.save_trajectories = False
        instance.compression_enabled = False
        instance._cleanup_task_resources = lambda *_a, **_kw: None
        instance._save_trajectory = lambda *_a, **_kw: None
        instance._handle_max_iterations = MagicMock(return_value="UNEXPECTED SUMMARY")
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id=instance.session_id, source="cli", model="test/model")
        instance._session_db = db
        instance._session_db_created = True
        assert isinstance(instance._todo_store, TodoStore)
        yield instance
        db.close()


def run_script(agent, answers, *, side_request=None, history=None, user="Fix and verify the bug"):
    requests = []
    live = []
    durable = []
    answers = iter(answers)
    if side_request:
        execute = agent._execute_tool_calls

        def execute_then_redirect(*args, **kwargs):
            value = execute(*args, **kwargs)
            if len(requests) == 1:
                agent.redirect(side_request)
            return value

        agent._execute_tool_calls = execute_then_redirect

    def call(kwargs):
        requests.append(deepcopy(kwargs))
        live.append(deepcopy(agent._session_messages))
        durable.append(agent._session_db.get_messages_as_conversation(agent.session_id))
        return next(answers)

    agent._interruptible_api_call = call
    result = agent.run_conversation(user, conversation_history=history)
    return result, requests, live, durable


def test_side_request_final_reconciles_current_plan_then_finishes(agent):
    done = [{**item, "status": "completed"} for item in PLAN]
    result, requests, live, durable = run_script(agent, [
        response(todos=PLAN), response(), response(todos=done, call_id="done"), response(FINAL),
    ], side_request="Also tell me where the setting lives.")
    assert len(requests) == 4
    assert result["final_response"] == FINAL
    assert agent._todo_store.read() == done
    # Request prefix, system prompt and schemas are unchanged by reconciliation.
    assert requests[2]["messages"][:len(requests[1]["messages"])] == requests[1]["messages"]
    assert requests[2].get("tools") == requests[1].get("tools")
    assert requests[2]["messages"][-2]["content"] == SIDE_ANSWER
    nudge = requests[2]["messages"][-1]["content"]
    assert "unfinished" in nudge.lower()
    assert "diagnose" in nudge and "verify" in nudge
    # No candidate/nudge in any in-flight flush, result or final DB history.
    for rows in [*live, *durable, result["messages"],
                 agent._session_db.get_messages_as_conversation(agent.session_id)]:
        assert not any(row.get("content") in (SIDE_ANSWER, nudge) for row in rows)
    assert all(row.get("content") != nudge for row in requests[3]["messages"])
    agent._handle_max_iterations.assert_not_called()


@pytest.mark.parametrize("status", ["completed", "cancelled"])
def test_closed_current_plan_does_not_nudge(agent, status):
    closed = [{**item, "status": status} for item in PLAN]
    result, requests, _, _ = run_script(agent, [response(todos=closed), response()])
    assert len(requests) == 2
    assert result["final_response"] == SIDE_ANSWER
    assert agent._todo_store.read() == closed


@pytest.mark.parametrize("old", [False, True])
def test_no_new_plan_does_not_nudge(agent, old):
    if old:
        agent._todo_store.write(PLAN)
    before = agent._todo_store.snapshot()
    result, requests, _, _ = run_script(agent, [response()])
    assert len(requests) == 1
    assert result["final_response"] == SIDE_ANSWER
    assert agent._todo_store.snapshot() == before


def test_hydrated_old_todos_are_baseline_not_current_turn(agent):
    history = [
        {"role": "user", "content": "Old task"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "old", "type": "function", "function": {
                "name": "todo_list", "arguments": json.dumps({"todos": PLAN}),
            },
        }]},
        {"role": "tool", "name": "todo_list", "tool_call_id": "old",
         "content": json.dumps({"todos": PLAN, "revision": 7})},
        {"role": "assistant", "content": "Paused."},
    ]
    result, requests, _, _ = run_script(agent, [response()], history=history)
    assert len(requests) == 1
    assert agent._todo_store.read() == PLAN
    assert result["final_response"] == SIDE_ANSWER


def test_opt_out_does_not_nudge(agent):
    agent._stall_guards = False
    result, requests, _, _ = run_script(agent, [response(todos=PLAN), response()])
    assert len(requests) == 2
    assert result["final_response"] == SIDE_ANSWER


def test_editing_old_item_reconciles_only_that_item(agent):
    agent._todo_store.write(PLAN)
    changed = [{**PLAN[0], "content": "Diagnose the newly requested bug"}]
    result, requests, _, _ = run_script(agent, [
        response(todos=changed, merge=True), response(), response("Blocked on access."),
    ])
    assert len(requests) == 3
    nudge = requests[-1]["messages"][-1]["content"]
    assert changed[0]["content"] in nudge
    assert PLAN[1]["content"] not in nudge
    assert PLAN[2]["content"] not in nudge
    assert result["final_response"] == "Blocked on access."


def test_completing_one_old_item_does_not_resurrect_untouched_pending_items(agent):
    agent._todo_store.write(PLAN)
    result, requests, _, _ = run_script(agent, [
        response(todos=[{"id": "diagnose", "status": "completed"}], merge=True), response(),
    ])
    assert len(requests) == 2
    assert agent._todo_store.read()[1:] == PLAN[1:]
    assert result["final_response"] == SIDE_ANSWER


def test_idempotent_old_plan_write_does_not_nudge(agent):
    agent._todo_store.write(PLAN)
    result, requests, _, _ = run_script(agent, [response(todos=PLAN), response()])
    assert len(requests) == 2
    assert result["final_response"] == SIDE_ANSWER


def test_later_turn_has_its_own_baseline_and_one_check(agent):
    first, requests, _, _ = run_script(agent, [
        response(todos=PLAN), response(), response("Work remains blocked."),
    ])
    assert len(requests) == 3
    second, requests, _, _ = run_script(agent, [response()], history=first["messages"])
    assert len(requests) == 1
    changed = [{**PLAN[0], "content": "Newly authorized investigation"}]
    third, requests, _, _ = run_script(agent, [
        response(todos=changed, merge=True, call_id="new-turn"), response(),
        response("The new investigation is blocked."),
    ], history=second["messages"])
    assert len(requests) == 3
    assert third["final_response"] == "The new investigation is blocked."


def test_one_nudge_per_turn_even_after_model_revises_plan(agent):
    revised = [{**PLAN[0], "content": "Diagnose another component"}]
    refusal = "I cannot finish this task with the available access."
    result, requests, _, _ = run_script(agent, [
        response(todos=PLAN), response(),
        response(todos=revised, merge=True, call_id="revise"), response(refusal),
    ])
    assert len(requests) == 4
    assert result["final_response"] == refusal
    assert agent._todo_store.read()[0] == revised[0]
    assert agent._todo_store.read()[1:] == PLAN[1:]
    agent._handle_max_iterations.assert_not_called()


@pytest.mark.parametrize("budget", [2, 3])
def test_budget_exhaustion_preserves_actual_candidate_without_completing(agent, budget):
    agent.max_iterations = budget
    agent.iteration_budget.max_total = budget
    answers = [response(todos=PLAN), response()]
    if budget == 3:
        answers.append(response(todos=[{**PLAN[0], "content": "Diagnosis still pending"}],
                                merge=True, call_id="revise"))
    result, requests, _, _ = run_script(agent, answers)
    assert len(requests) == budget
    assert result["final_response"] == SIDE_ANSWER
    assert result["completed"] is False
    assert all(item["status"] in {"pending", "in_progress"} for item in agent._todo_store.read())
    assert result["messages"][-1]["content"] == SIDE_ANSWER
    agent._handle_max_iterations.assert_not_called()


@pytest.mark.parametrize("reason", [
    "Stop the original task.", "Pause the original task.", "Cancel the original task.",
    "Replace the original task with this question only.",
    "You do not have authorization to make changes.",
    "The background job will notify me; do not wait or start another one.",
    "Only provide a plan; do not execute it.",
])
def test_reconciliation_is_advisory_not_authorization(agent, reason):
    final = "The work remains unfinished; no further action was taken."
    result, requests, _, _ = run_script(agent, [
        response(todos=PLAN), response(final), response(final),
    ], side_request=reason)
    assert len(requests) == 3
    nudge = requests[-1]["messages"][-1]["content"].lower()
    # Semantic stop/scope decisions stay with the model, not a phrase classifier.
    for constraint in ("stop", "pause", "cancel", "supersed", "authorization", "background", "planning"):
        assert constraint in nudge
    assert result["final_response"] == final
    assert agent._todo_store.snapshot() == {"todos": PLAN, "revision": 1}
    assert sum(len(row.get("tool_calls") or []) for row in result["messages"]) == 1
