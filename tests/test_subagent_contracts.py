from __future__ import annotations

import json
from typing import Any, cast

import pytest

from harness_core.events import event_runtime_status
from harness_core.orchestration_sdk import (
    SUBAGENT_EVENT_SCHEMA_VERSION,
    DelegationBatchResult,
    DelegationRequest,
    DelegationTask,
    DelegationToolHandler,
    SubAgentArtifactRef,
    SubAgentBudget,
    SubAgentContextRef,
    SubAgentExecutor,
    SubAgentInputRequest,
    SubAgentRegistry,
    SubAgentResult,
    SubAgentSpec,
    TaskBrief,
    delegation_tool_parameters_schema,
)
from harness_core.runtime_sdk import ToolCall
from harness_core.extension_sdk import ToolExecutionContext
from harness_core.subagents.execution_support import _child_run_id


class RecordingStore:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.settled: list[SubAgentResult] = []
        self.checkpoints: dict[str, dict[str, Any]] = {}

    def parent_accepts_results(self, _parent_run_id: str) -> bool:
        return True

    def create_child(self, **payload: Any) -> None:
        self.created.append(payload)

    def settle_child(self, result: SubAgentResult) -> None:
        self.settled.append(result)

    def load_child_result(self, _child_run_id: str) -> SubAgentResult | None:
        return None

    def load_child_checkpoint(self, child_run_id: str) -> dict[str, Any] | None:
        return self.checkpoints.get(child_run_id)

    def checkpoint_child(
        self,
        child_run_id: str,
        *,
        phase: str,
        state: dict[str, Any],
    ) -> None:
        self.checkpoints[child_run_id] = {**state, "phase": phase}

    def cancel_requested(self, _child_run_id: str, _parent_run_id: str) -> bool:
        return False


class JsonProvider:
    model = "test-reasoner"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.request_timeouts: list[float] = []

    def complete(
        self,
        _system_prompt: str,
        _user_prompt: str,
        *,
        request_timeout: float,
        **_kwargs: Any,
    ) -> str:
        self.request_timeouts.append(request_timeout)
        return json.dumps(self.payload)


def _completed_payload() -> dict[str, Any]:
    return {
        "conclusion": "Checked.",
        "evidence": ["fact"],
        "risks": [],
        "recommendations": ["continue"],
        "artifact_refs": [{"id": "report-1", "artifact_type": "report"}],
    }


def test_task_brief_is_a_backward_compatible_delegation_task() -> None:
    assert TaskBrief is DelegationTask
    task = TaskBrief(
        id="task-1",
        agent_id="review.facts",
        objective=" Review the facts ",
        normalized_question=" What happened? ",
        context_refs=[SubAgentContextRef(id="context-1")],
        artifact_refs=[SubAgentArtifactRef(id="artifact-1", artifact_type="report")],
        idempotency_key="stable-review",
        budget=SubAgentBudget(max_model_calls=2, max_tool_calls=1),
    )

    assert task.objective == "Review the facts"
    assert task.normalized_question == "What happened?"
    assert task.context_refs == [SubAgentContextRef(id="context-1")]
    assert task.artifact_refs == [
        SubAgentArtifactRef(id="artifact-1", artifact_type="report")
    ]
    assert task.effective_idempotency_key == "stable-review"


def test_explicit_idempotency_key_is_stable_across_retry_batches() -> None:
    first = TaskBrief(
        id="attempt-1",
        agent_id="review.facts",
        objective="Review",
        idempotency_key="review-once",
    )
    retry = first.model_copy(update={"id": "attempt-2"})
    legacy = DelegationTask(id="legacy", agent_id="review.facts", objective="Review")

    assert _child_run_id("parent", "batch-1", first) == _child_run_id(
        "parent", "batch-2", retry
    )
    assert _child_run_id("parent", "batch-1", legacy) != _child_run_id(
        "parent", "batch-2", legacy
    )


def test_result_promotes_legacy_artifact_refs_and_validates_needs_input() -> None:
    legacy = SubAgentResult(
        task_id="task-1",
        agent_id="review.facts",
        child_run_id="child-1",
        status="completed",
        metadata={"artifact_refs": ["artifact-1"]},
    )
    pending = SubAgentResult(
        task_id="task-2",
        agent_id="review.facts",
        child_run_id="child-2",
        status="needs_input",
        input_request=SubAgentInputRequest(
            prompt="Which period should I review?",
            requirements=["date range"],
            resume_token="resume-1",
        ),
    )

    assert legacy.parent_projection()["artifact_refs"][0]["id"] == "artifact-1"
    assert pending.outcome == "needs_input"
    assert pending.parent_projection()["input_request"]["resume_token"] == "resume-1"
    with pytest.raises(ValueError, match="input_request"):
        SubAgentResult(
            task_id="invalid",
            agent_id="review.facts",
            child_run_id="child-invalid",
            status="needs_input",
        )


def test_executor_persists_bound_lineage_and_emits_standard_event_fields() -> None:
    store = RecordingStore()
    provider = JsonProvider(_completed_payload())
    executor = SubAgentExecutor(
        SubAgentRegistry(
            [
                SubAgentSpec(
                    id="review.facts",
                    label="Fact reviewer",
                    max_model_calls=2,
                )
            ]
        ),
        run_store=store,
    )
    events: list[dict[str, Any]] = []
    batch = executor.execute_many(
        DelegationRequest(
            delegation_id="delegation-1",
            tasks=[
                TaskBrief(
                    id="task-1",
                    agent_id="review.facts",
                    objective="Review",
                    idempotency_key="review-once",
                    timeout_seconds=15,
                    context_refs=[SubAgentContextRef(id="context-1", kind="memory")],
                )
            ],
        ),
        context=ToolExecutionContext(
            run_id="run-1",
            user_id="user-1",
            thread_id="thread-1",
        ),
        providers={"reasoning": provider},
        event_sink=events.append,
    )

    created_task = store.created[0]["task"]
    result = batch.results[0]
    completed_event = next(item for item in events if item["event_type"] == "subagent.completed")
    assert created_task.lineage.root_run_id == "run-1"
    assert created_task.lineage.parent_run_id == "run-1"
    assert created_task.lineage.child_run_id == result.child_run_id
    assert result.idempotency_key == "review-once"
    assert result.lineage == created_task.lineage
    assert result.artifact_refs[0].id == "report-1"
    assert completed_event["payload"]["schema_version"] == SUBAGENT_EVENT_SCHEMA_VERSION
    assert completed_event["payload"]["idempotency_key"] == "review-once"
    assert completed_event["payload"]["spec_version"] == "1.0"
    assert provider.request_timeouts[0] <= 15


def test_executor_returns_typed_needs_input_and_enforces_task_model_budget() -> None:
    registry = SubAgentRegistry(
        [SubAgentSpec(id="review.facts", label="Fact reviewer")]
    )
    pending_provider = JsonProvider(
        {
            "status": "needs_input",
            "conclusion": "I need a date range.",
            "evidence": [],
            "risks": [],
            "recommendations": [],
            "input_request": {
                "prompt": "Which date range should I review?",
                "requirements": ["start date", "end date"],
                "resume_token": "resume-dates",
            },
        }
    )
    pending_store = RecordingStore()
    pending = SubAgentExecutor(registry, run_store=pending_store).execute_many(
        DelegationRequest(
            delegation_id="pending-1",
            tasks=[
                TaskBrief(
                    id="task-pending",
                    agent_id="review.facts",
                    objective="Review",
                )
            ],
        ),
        context=ToolExecutionContext(run_id="run-1", user_id="user-1"),
        providers={"reasoning": pending_provider},
    )

    failing_provider = JsonProvider({})
    failed = SubAgentExecutor(registry).execute_many(
        DelegationRequest(
            delegation_id="budget-1",
            tasks=[
                TaskBrief(
                    id="task-budget",
                    agent_id="review.facts",
                    objective="Review",
                    budget=SubAgentBudget(max_model_calls=1),
                )
            ],
        ),
        context=ToolExecutionContext(run_id="run-2", user_id="user-1"),
        providers={"reasoning": failing_provider},
    )

    assert pending.status == "needs_input"
    assert pending.results[0].input_request is not None
    assert pending.results[0].input_request.resume_token == "resume-dates"
    assert pending_store.settled == []
    assert next(iter(pending_store.checkpoints.values()))["phase"] == "needs_input"
    assert failed.status == "failed"
    assert "model call budget exceeded" in failed.results[0].error
    assert len(failing_provider.request_timeouts) == 1


class NeedsInputExecutor:
    class Registry:
        @staticmethod
        def get(agent_id: str) -> SubAgentSpec:
            return SubAgentSpec(id=agent_id, label="Fact reviewer")

    registry = Registry()

    @staticmethod
    def execute_many(request: DelegationRequest, **_kwargs: Any) -> DelegationBatchResult:
        task = request.tasks[0]
        return DelegationBatchResult(
            delegation_id=request.delegation_id,
            root_run_id=request.root_run_id,
            parent_run_id=request.parent_run_id,
            status="needs_input",
            results=[
                SubAgentResult(
                    task_id=task.id,
                    agent_id=task.agent_id,
                    child_run_id="child-input",
                    status="needs_input",
                    input_request=SubAgentInputRequest(
                        prompt="Please provide the missing date range.",
                        requirements=["start date", "end date"],
                        input_schema={"type": "object"},
                        resume_token="resume-input",
                    ),
                )
            ],
        )


def test_needs_input_result_suspends_the_parent_tool() -> None:
    result = DelegationToolHandler(cast(SubAgentExecutor, NeedsInputExecutor()))(
        ToolCall(
            id="delegate-1",
            name="agent.delegate",
            arguments={
                "execution_mode": "foreground",
                "tasks": [
                    {
                        "id": "task-1",
                        "agent_id": "review.facts",
                        "objective": "Review",
                    }
                ],
            },
        ),
        ToolExecutionContext(
            run_id="run-1",
            user_id="user-1",
            metadata={"model_providers": {"reasoning": object()}},
        ),
    )

    assert result.status == "requires_user_action"
    assert result.pending_action is not None
    assert result.pending_action.action_type == "subagent_input"
    assert result.pending_action.payload["resume_token"] == "resume-input"
    assert result.observation is not None
    assert result.observation.status == "requires_user_action"


def test_delegation_schema_and_runtime_status_include_new_contracts() -> None:
    task_properties = delegation_tool_parameters_schema()["properties"]["tasks"]["items"][
        "properties"
    ]

    assert {"context_refs", "artifact_refs", "idempotency_key", "timeout_seconds"} <= set(
        task_properties
    )
    assert event_runtime_status({"event_type": "subagent.started"}) == "executing_tools"
    assert event_runtime_status({"event_type": "subagent.needs_input"}) == "waiting_user"
