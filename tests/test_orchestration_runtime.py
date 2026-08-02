from __future__ import annotations

from typing import Any

from harness_core.orchestration_sdk import (
    DelegationBatchResult,
    DeliberationCoordinator,
    DeliberationParticipant,
    DeliberationSpec,
    DelegationToolHandler,
    SubAgentResult,
    SubAgentSpec,
)
from harness_core.runtime_sdk import ToolCall
from harness_core.extension_sdk import ToolExecutionContext


class ScriptedSubAgents:
    def __init__(self, *, first_decision: str = "continue") -> None:
        self.first_decision = first_decision
        self.requests = []

    def execute_many(self, request, **_kwargs) -> DelegationBatchResult:
        self.requests.append(request)
        results = []
        for task in request.tasks:
            moderator = task.agent_id == "review.moderator"
            phase = str(task.input_data.get("phase") or "")
            decision = (
                self.first_decision
                if moderator and phase == "moderate" and task.input_data.get("round_index") == 1
                else "synthesize"
            )
            output = {
                "decision": decision,
                "target_agent_ids": ["review.risk"] if decision != "synthesize" else [],
                "unresolved_questions": ["remaining risk"] if decision != "synthesize" else [],
                "convergence_score": 0.4 if decision != "synthesize" else 0.9,
                "conditions": ["verify the source"],
                "action_recommendations": ["proceed carefully"],
                "judgment_boundary": "facts only",
            }
            results.append(
                SubAgentResult(
                    task_id=task.id,
                    agent_id=task.agent_id,
                    child_run_id=f"child:{task.id}",
                    status="completed",
                    conclusion="Synthesis complete." if moderator else f"View from {task.agent_id}.",
                    evidence=["shared fact"],
                    risks=["remaining risk"],
                    recommendations=["verify"],
                    confidence=0.8,
                    output=output if moderator else {},
                )
            )
        return DelegationBatchResult(
            delegation_id=request.delegation_id,
            root_run_id=request.root_run_id,
            parent_run_id=request.parent_run_id,
            status="completed",
            results=results,
        )


def _participants() -> list[DeliberationParticipant]:
    return [
        DeliberationParticipant(
            agent_id="review.facts",
            participant_instance_id="facts-1",
            label="Facts",
            display_name="Fact reviewer",
        ),
        DeliberationParticipant(
            agent_id="review.risk",
            participant_instance_id="risk-1",
            label="Risk",
            display_name="Risk reviewer",
        ),
    ]


def _context(**metadata: Any) -> ToolExecutionContext:
    return ToolExecutionContext(
        run_id="run-1",
        thread_id="thread-1",
        user_id="user-1",
        metadata=metadata,
    )


def test_deliberation_runs_rebuttal_synthesis_events_and_checkpoints() -> None:
    executor = ScriptedSubAgents(first_decision="continue")
    events = []
    checkpoints: dict[str, dict[str, Any]] = {}
    result = DeliberationCoordinator(executor).run(
        DeliberationSpec(
            deliberation_id="review-1",
            question="Should the plan proceed?",
            facts={"quantity": 12},
            participants=_participants(),
            moderator_agent_id="review.moderator",
        ),
        context=_context(),
        providers={"reasoning": object()},
        event_sink=events.append,
        checkpoint_sink=lambda phase, state: checkpoints.__setitem__(phase, state),
    )

    assert result.status == "completed"
    assert result.stop_reason == "moderator_converged"
    assert [item.phase for item in result.arguments] == ["opening", "opening", "rebuttal"]
    assert result.synthesis["judgment_boundary"] == "facts only"
    assert {"opening", "moderating", "rebuttal", "synthesizing", "completed"} <= set(checkpoints)
    assert events[0]["event_type"] == "deliberation.started"
    assert events[-1]["event_type"] == "deliberation.completed"


def test_deliberation_resumes_without_replaying_opening() -> None:
    first = ScriptedSubAgents(first_decision="synthesize")
    checkpoints: dict[str, dict[str, Any]] = {}
    spec = DeliberationSpec(
        deliberation_id="review-resume",
        question="Review the evidence.",
        facts={"record": "stable"},
        participants=_participants(),
        moderator_agent_id="review.moderator",
    )
    DeliberationCoordinator(first).run(
        spec,
        context=_context(),
        providers={"reasoning": object()},
        checkpoint_sink=lambda phase, state: checkpoints.__setitem__(phase, state),
    )
    resumed_executor = ScriptedSubAgents(first_decision="synthesize")
    resumed = DeliberationCoordinator(resumed_executor).run(
        spec,
        context=_context(),
        providers={"reasoning": object()},
        resume_state=checkpoints["moderating"],
    )

    assert len(resumed.arguments) == 2
    assert len(resumed_executor.requests) == 1
    assert resumed_executor.requests[0].tasks[0].id.startswith("synthesize-")


def test_deliberation_scopes_participant_facts_and_reports_diagnostics() -> None:
    executor = ScriptedSubAgents(first_decision="synthesize")
    participants = _participants()
    participants[0] = participants[0].model_copy(
        update={
            "fact_keys": ["record"],
            "instructions": ["Inspect only the supplied record."],
        }
    )
    result = DeliberationCoordinator(executor).run(
        DeliberationSpec(
            deliberation_id="review-scoped",
            question="Review the record.",
            facts={
                "record": {"status": "stable"},
                "timing": {"year": 2026},
                "subject": {"id": "subject-1"},
                "provenance": [{"source": "fixture"}],
            },
            participants=participants,
            moderator_agent_id="review.moderator",
        ),
        context=_context(),
        providers={"reasoning": object()},
    )

    opening = executor.requests[0].tasks
    assert set(opening[0].input_data["facts"]) == {"record", "subject", "provenance"}
    assert "Inspect only the supplied record." in opening[0].constraints
    assert "timing" in opening[1].input_data["facts"]
    assert result.diagnostics["completed_argument_count"] == 2
    assert result.diagnostics["failed_argument_count"] == 0
    assert result.diagnostics["budget"]["maximum"] == 12


def test_deliberation_emits_explicit_stage_transitions() -> None:
    executor = ScriptedSubAgents(first_decision="continue")
    events: list[dict[str, Any]] = []
    DeliberationCoordinator(executor).run(
        DeliberationSpec(
            deliberation_id="review-stages",
            question="Should the plan proceed?",
            facts={"quantity": 12},
            participants=_participants(),
            moderator_agent_id="review.moderator",
        ),
        context=_context(),
        providers={"reasoning": object()},
        event_sink=events.append,
    )

    stages = [
        event["payload"]["phase"]
        for event in events
        if event["event_type"] == "deliberation.stage.started"
    ]
    assert stages == ["opening", "moderate", "rebuttal", "moderate", "synthesize"]
    rebuttal = next(
        event for event in events
        if event["event_type"] == "deliberation.stage.started"
        and event["payload"]["phase"] == "rebuttal"
    )
    assert rebuttal["payload"]["participant_ids"] == ["review.risk"]


class Registry:
    def get(self, agent_id: str) -> SubAgentSpec:
        if agent_id != "review.facts":
            raise KeyError(agent_id)
        return SubAgentSpec(id=agent_id, label="Fact reviewer")


class InlineExecutor:
    registry = Registry()

    def execute_many(self, request, **_kwargs) -> DelegationBatchResult:
        task = request.tasks[0]
        return DelegationBatchResult(
            delegation_id=request.delegation_id,
            root_run_id=request.root_run_id,
            parent_run_id=request.parent_run_id,
            status="completed",
            results=[
                SubAgentResult(
                    task_id=task.id,
                    agent_id=task.agent_id,
                    child_run_id="child-inline",
                    status="completed",
                    conclusion="Checked.",
                )
            ],
        )


def _delegation_call(agent_id: str = "review.facts") -> ToolCall:
    return ToolCall(
        id="delegate-1",
        name="agent.delegate",
        arguments={
            "tasks": [
                {
                    "id": "facts",
                    "agent_id": agent_id,
                    "objective": "Review facts",
                }
            ]
        },
    )


def test_delegation_tool_handles_success_invalid_request_and_missing_provider() -> None:
    handler = DelegationToolHandler(InlineExecutor())
    success = handler(
        _delegation_call(),
        _context(model_providers={"fast": object()}),
    )
    invalid = handler(
        _delegation_call("review.unknown"),
        _context(model_providers={"fast": object()}),
    )
    unavailable = handler(_delegation_call(), _context())

    assert success.status == "succeeded"
    assert success.outcome == "completed"
    assert invalid.status == "succeeded"
    assert invalid.outcome == "skipped"
    assert invalid.data["fallback"] == "continue_with_parent_agent"
    assert unavailable.status == "failed"
    assert unavailable.retryable is True


class Dispatcher:
    def dispatch(self, request, **_kwargs) -> dict[str, Any]:
        return {"batch_id": request.delegation_id, "status": "running"}


def test_delegation_tool_projects_async_dispatch() -> None:
    context = _context(model_providers={"fast": object()})
    context.session_factory = lambda: object()
    result = DelegationToolHandler(InlineExecutor(), dispatcher=Dispatcher())(
        _delegation_call(),
        context,
    )

    assert result.status == "waiting_async"
    assert result.observation is not None
    assert result.observation.status == "pending"
