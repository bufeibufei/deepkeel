from __future__ import annotations

import pytest

from harness_core.contracts import AgentMessage, RunContext, RunStatus
from harness_core.control import InMemoryRunControl
from harness_core.failures import RunCanceledError
from harness_core.graph_nodes import GraphNodes
from harness_core.graph_state import _state_from_context
from harness_core.graph_workflow import (
    _route_after_model,
    _route_after_tools,
    _route_from_start,
)
from harness_core.runtime_api import RuntimeResultStatus
from harness_core.runtime_policy import _budget_limits, _resolved_model_policy
from harness_core.runtime_results import project_harness_result
from harness_core.subagents.contracts import DelegationTask
from harness_core.subagents.execution_support import _child_run_id
from harness_core.subagents.execution_types import _DelegationQuota
from harness_core.subagents.output_validation import (
    _confidence,
    _json_object,
    _validate_input,
)


class _Provider:
    model_role = "fast"


def test_runtime_policy_resolves_provider_roles_and_budget_limits() -> None:
    policy = _resolved_model_policy(
        {"budget": {"max_model_calls": 3, "max_parallel_tools": 2}},
        provider=_Provider(),
        providers=None,
        max_steps=12,
    )

    assert policy["mode"] == "adaptive"
    assert policy["primary_role"] == "reasoning"
    assert policy["available_roles"] == ["fast", "reasoning"]
    assert _budget_limits(policy)["model_calls"] == 3
    assert _budget_limits(policy)["tool_concurrency"] == 2


def test_runtime_results_projects_a_completed_typed_result() -> None:
    state = {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "status": "completed",
        "step_count": 1,
        "messages": [],
        "observations": [],
        "artifacts": [],
        "tool_results": [],
        "pending_tool_calls": [],
        "final_answer": {
            "markdown": "The task is complete.",
            "summary": "Complete",
            "status": "completed",
        },
        "metadata": {},
    }

    result = project_harness_result(
        state,
        question="Complete the task",
        context_bundle={},
        short_context={},
        skill_activation={},
        streamed_events=[],
    )

    assert result.status is RuntimeResultStatus.COMPLETED
    assert result.final_answer.markdown == "The task is complete."
    assert result.checkpoint["schema_version"] == "harness-checkpoint-v2"


def test_graph_state_serializes_typed_context_without_losing_identity() -> None:
    context = RunContext(
        run_id="run-1",
        thread_id="thread-1",
        turn_id="turn-1",
        user_id="user-1",
        status=RunStatus.PREPARING,
        messages=[AgentMessage(id="message-1", role="user", content="hello")],
        skill_activation={"kind": "workflow"},
    )

    state = _state_from_context(context)

    assert state["run_id"] == "run-1"
    assert state["messages"][0]["content"] == "hello"
    assert state["policy_phase"] == "pending"
    assert state["tool_results"] == []


@pytest.mark.parametrize(
    ("router", "state", "expected"),
    [
        (_route_from_start, {"pending_tool_calls": [{"id": "call-1"}]}, "tools"),
        (_route_from_start, {"pending_tool_calls": []}, "model"),
        (_route_after_tools, {"status": "waiting_user"}, "await_user"),
        (_route_after_tools, {"status": "waiting_async"}, "await_async"),
        (_route_after_model, {"pending_tool_calls": [{"id": "call-1"}]}, "tools"),
        (_route_after_model, {"status": "completed"}, "end"),
    ],
)
def test_graph_workflow_routes_each_runtime_phase(router, state, expected) -> None:
    assert router(state) == expected


def test_graph_nodes_honors_cooperative_cancellation() -> None:
    control = InMemoryRunControl()
    control.cancel("run-1")
    nodes = object.__new__(GraphNodes)
    nodes.control = control
    nodes.deadline_monotonic = None

    with pytest.raises(RunCanceledError):
        nodes.ensure_active({"run_id": "run-1"})


def test_subagent_quota_reserves_calls_atomically() -> None:
    quota = _DelegationQuota(max_model_calls=1, max_tool_calls=2)

    quota.reserve_model_call()
    quota.reserve_tool_calls(2)

    with pytest.raises(RuntimeError, match="model call budget"):
        quota.reserve_model_call()
    with pytest.raises(RuntimeError, match="tool call budget"):
        quota.reserve_tool_calls(1)


def test_subagent_output_validation_normalizes_structured_output() -> None:
    assert _json_object('```json\n{"conclusion":"ready"}\n```') == {
        "conclusion": "ready"
    }
    assert _confidence("0.75") == 0.75
    with pytest.raises(RuntimeError, match="missing required field"):
        _validate_input({}, {"required": ["record_id"]})


def test_subagent_child_run_id_is_stable_and_task_scoped() -> None:
    first = DelegationTask(id="task-1", agent_id="reviewer", objective="Review")
    second = DelegationTask(id="task-2", agent_id="reviewer", objective="Review")

    assert _child_run_id("parent", "delegation", first) == _child_run_id(
        "parent", "delegation", first
    )
    assert _child_run_id("parent", "delegation", first) != _child_run_id(
        "parent", "delegation", second
    )
