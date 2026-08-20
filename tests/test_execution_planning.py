from __future__ import annotations

import json

import pytest

from deepkeel.adapter_sdk import RuntimePorts
from deepkeel.extension_sdk import ToolExecutionContext, ToolExecutor, ToolRegistry, ToolSpec
from deepkeel.orchestration_sdk import (
    ExecutionPlan,
    ExecutionPlanValidator,
    PlanStep,
    PlanValidationError,
    PlanningPolicy,
    merge_plan_revision,
    select_ready_plan_steps,
)
from deepkeel.runtime_sdk import (
    HarnessRuntimeBuilder,
    PendingAction,
    RuntimeRequest,
    ToolCall,
    ToolResult,
)


def _tool(
    name: str,
    *,
    read_only: bool = True,
    parallel_safe: bool = True,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Execute {name}.",
        parameters_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "additionalProperties": False,
        },
        read_only=read_only,
        parallel_safe=parallel_safe,
    )


def test_plan_validator_rejects_cycles_and_unauthorized_capabilities() -> None:
    registry = ToolRegistry([_tool("facts.profile"), _tool("facts.report")])
    validator = ExecutionPlanValidator(registry)
    cyclic = ExecutionPlan(
        plan_id="plan-cycle",
        run_id="run-1",
        objective="Build a grounded answer.",
        steps=[
            PlanStep(
                id="profile",
                title="Read profile",
                objective="Read the profile.",
                capability_ref="facts.profile",
                depends_on=["report"],
            ),
            PlanStep(
                id="report",
                title="Read report",
                objective="Read the report.",
                capability_ref="facts.report",
                depends_on=["profile"],
            ),
        ],
    )

    with pytest.raises(PlanValidationError, match="cycle"):
        validator.validate(cyclic, policy=PlanningPolicy())

    unauthorized = cyclic.model_copy(
        update={
            "plan_id": "plan-permission",
            "steps": [
                PlanStep(
                    id="profile",
                    title="Read profile",
                    objective="Read the profile.",
                    capability_ref="facts.profile",
                )
            ],
        }
    )
    with pytest.raises(PlanValidationError, match="not allowed"):
        validator.validate(
            unauthorized,
            policy=PlanningPolicy(),
            allowed_names={"facts.report"},
        )


def test_ready_step_scheduler_parallelizes_reads_but_serializes_writes() -> None:
    plan = ExecutionPlan(
        plan_id="plan-ready",
        run_id="run-1",
        objective="Prepare and persist a result.",
        status="running",
        steps=[
            PlanStep(
                id="read-a",
                title="Read A",
                objective="Read A.",
                capability_ref="facts.a",
                read_only=True,
                parallel_safe=True,
            ),
            PlanStep(
                id="read-b",
                title="Read B",
                objective="Read B.",
                capability_ref="facts.b",
                read_only=True,
                parallel_safe=True,
            ),
            PlanStep(
                id="write",
                title="Persist",
                objective="Persist the result.",
                capability_ref="records.write",
                depends_on=["read-a", "read-b"],
                read_only=False,
                parallel_safe=False,
            ),
        ],
    )

    assert [step.id for step in select_ready_plan_steps(plan, max_parallel_steps=4)] == [
        "read-a",
        "read-b",
    ]

    completed_reads = plan.model_copy(
        update={
            "steps": [
                step.model_copy(update={"status": "completed"})
                if step.id.startswith("read-")
                else step
                for step in plan.steps
            ]
        }
    )
    assert [step.id for step in select_ready_plan_steps(completed_reads)] == ["write"]


class _PlanningProvider:
    model = "planning-provider"
    model_role = "reasoning"

    def __init__(self) -> None:
        self.calls = 0
        self.tool_views: list[list[str]] = []
        self.tool_choices: list[object] = []

    def complete_chat(self, _messages, *, tools=None, **kwargs):
        self.calls += 1
        self.tool_views.append(
            [str(item.get("function", {}).get("name") or "") for item in tools or []]
        )
        self.tool_choices.append(kwargs.get("tool_choice"))
        if self.calls == 1:
            arguments = {
                "objective": "Combine two independent evidence sources.",
                "steps": [
                    {
                        "id": "source-a",
                        "title": "Read source A",
                        "objective": "Collect source A.",
                        "capability_ref": "facts.a",
                        "arguments": {"value": "A"},
                    },
                    {
                        "id": "source-b",
                        "title": "Read source B",
                        "objective": "Collect source B.",
                        "capability_ref": "facts.b",
                        "arguments": {"value": "B"},
                    },
                ],
            }
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "create-plan",
                            "type": "function",
                            "function": {
                                "name": "runtime.create_plan",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
                "model": self.model,
            }
        return {
            "message": {
                "role": "assistant",
                "content": "Both evidence sources were collected and synthesized.",
            },
            "finish_reason": "stop",
            "model": self.model,
        }


def test_runtime_executes_a_plan_inside_the_existing_react_graph() -> None:
    registry = ToolRegistry([_tool("facts.a"), _tool("facts.b")])
    executor = ToolExecutor(registry)
    executed: list[str] = []

    def handler(call: ToolCall, _context: ToolExecutionContext) -> ToolResult:
        executed.append(call.name)
        return ToolResult(
            call=call,
            status="succeeded",
            summary=f"Collected {call.arguments['value']}.",
            data={"value": call.arguments["value"]},
        )

    executor.register("facts.a", handler)
    executor.register("facts.b", handler)
    runtime = (
        HarnessRuntimeBuilder(registry, executor)
        .with_ports(RuntimePorts(planning_enabled=True))
        .build()
    )
    provider = _PlanningProvider()

    result = runtime.run(
        RuntimeRequest(question="Compare both sources.", run_id="planned-run"),
        provider=provider,
    )

    assert result.status == "completed", result.model_dump(mode="json")
    assert sorted(executed) == ["facts.a", "facts.b"]
    assert "runtime.create_plan" in provider.tool_views[0]
    assert result.mode == "plan_execute"
    assert result.execution_plan["status"] == "completed"
    assert result.checkpoint["execution_plan"]["status"] == "completed"
    event_types = [event.event_type for event in result.events]
    assert "plan.validated" in event_types
    assert "plan.started" in event_types
    assert event_types.count("plan.step.completed") == 2
    assert "plan.completed" in event_types


def test_disabled_planning_policy_hides_the_control_tool() -> None:
    registry = ToolRegistry([_tool("facts.a")])
    executor = ToolExecutor(registry)
    runtime = (
        HarnessRuntimeBuilder(registry, executor)
        .with_ports(RuntimePorts(planning_enabled=True))
        .build()
    )
    provider = _PlanningProvider()

    result = runtime.run(
        RuntimeRequest(
            question="Answer directly.",
            run_id="planning-disabled",
            skill_activation={
                "skill_id": "direct-answer",
                "planning_policy": {"mode": "disabled"},
            },
        ),
        provider=provider,
    )

    assert result.status == "completed"
    assert "runtime.create_plan" not in provider.tool_views[0]


def test_required_planning_policy_forces_the_plan_control_tool() -> None:
    registry = ToolRegistry([_tool("facts.a"), _tool("facts.b")])
    executor = ToolExecutor(registry)

    def handler(call: ToolCall, _context: ToolExecutionContext) -> ToolResult:
        return ToolResult(call=call, status="succeeded", summary="Collected.")

    executor.register("facts.a", handler)
    executor.register("facts.b", handler)
    runtime = (
        HarnessRuntimeBuilder(registry, executor)
        .with_ports(RuntimePorts(planning_enabled=True))
        .build()
    )
    provider = _PlanningProvider()

    result = runtime.run(
        RuntimeRequest(
            question="Compare both sources.",
            run_id="planning-required",
            skill_activation={
                "skill_id": "required-plan",
                "planning_policy": {"mode": "required"},
            },
        ),
        provider=provider,
    )

    assert result.status == "completed", result.model_dump(mode="json")
    assert provider.tool_choices[0] == {
        "type": "function",
        "function": {"name": "runtime.create_plan"},
    }


def test_plan_revision_preserves_completed_steps_and_rejects_mutation() -> None:
    registry = ToolRegistry([_tool("facts.a"), _tool("facts.b")])
    validator = ExecutionPlanValidator(registry)
    previous = ExecutionPlan(
        plan_id="plan-revision",
        run_id="run-1",
        objective="Collect evidence.",
        revision=1,
        status="replanning",
        steps=[
            PlanStep(
                id="source-a",
                title="Read source A",
                objective="Read A.",
                capability_ref="facts.a",
                status="completed",
                attempt_count=1,
                result_summary="A collected.",
            )
        ],
    )
    proposed = ExecutionPlan(
        plan_id="model-generated-id",
        run_id="ignored-run",
        objective="Collect revised evidence.",
        steps=[
            PlanStep(
                id="source-a",
                title="Read a different source",
                objective="Mutate completed work.",
                capability_ref="facts.a",
            ),
            PlanStep(
                id="source-b",
                title="Read source B",
                objective="Read B.",
                capability_ref="facts.b",
                depends_on=["source-a"],
            ),
        ],
    )
    merged = merge_plan_revision(previous, proposed, reason="Need another source.")

    with pytest.raises(PlanValidationError, match="immutable"):
        validator.validate(
            merged,
            policy=PlanningPolicy(),
            previous=previous,
        )


class _ResumablePlanningProvider:
    model = "resumable-planning-provider"
    model_role = "reasoning"

    def __init__(self) -> None:
        self.calls = 0

    def complete_chat(self, _messages, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "create-resumable-plan",
                            "type": "function",
                            "function": {
                                "name": "runtime.create_plan",
                                "arguments": json.dumps(
                                    {
                                        "objective": "Confirm before collecting evidence.",
                                        "steps": [
                                            {
                                                "id": "approval",
                                                "title": "Confirm access",
                                                "objective": "Obtain user approval.",
                                                "capability_ref": "actions.confirm",
                                            },
                                            {
                                                "id": "evidence",
                                                "title": "Collect evidence",
                                                "objective": "Collect evidence after approval.",
                                                "capability_ref": "facts.after",
                                                "depends_on": ["approval"],
                                            },
                                        ],
                                    }
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
                "model": self.model,
            }
        return {
            "message": {
                "role": "assistant",
                "content": "Approval and evidence collection are complete.",
            },
            "finish_reason": "stop",
            "model": self.model,
        }


def test_plan_resumes_after_user_action_without_replaying_completed_work() -> None:
    approval_spec = _tool(
        "actions.confirm",
        read_only=False,
        parallel_safe=False,
    ).model_copy(update={"requires_user_action": True})
    registry = ToolRegistry([approval_spec, _tool("facts.after")])
    executor = ToolExecutor(registry)
    executed: list[str] = []

    def confirm(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        executed.append(call.name)
        return ToolResult(
            call=call,
            status="requires_user_action",
            summary="Approval is required.",
            pending_action=PendingAction(
                id="approval-action",
                run_id=context.run_id,
                tool_call_id=call.id,
                action_type="confirmation",
                title="Confirm access",
                prompt="Continue?",
            ),
        )

    def collect(call: ToolCall, _context: ToolExecutionContext) -> ToolResult:
        executed.append(call.name)
        return ToolResult(
            call=call,
            status="succeeded",
            summary="Evidence collected.",
        )

    executor.register("actions.confirm", confirm)
    executor.register("facts.after", collect)
    runtime = (
        HarnessRuntimeBuilder(registry, executor)
        .with_ports(RuntimePorts(planning_enabled=True))
        .build()
    )
    provider = _ResumablePlanningProvider()
    base = {
        "question": "Run the resumable workflow.",
        "run_id": "resumable-plan-run",
    }

    waiting = runtime.run(RuntimeRequest(**base), provider=provider)
    resumed = runtime.run(
        RuntimeRequest(
            **base,
            short_context={
                "resume": True,
                "resume_observation": {
                    "status": "succeeded",
                    "summary": "Access approved.",
                },
                "previous_runtime": waiting.model_dump(mode="json"),
            },
        ),
        provider=provider,
    )

    assert waiting.status == "waiting_user_action"
    assert waiting.execution_plan["status"] == "waiting"
    assert resumed.status == "completed", resumed.model_dump(mode="json")
    assert executed == ["actions.confirm", "facts.after"]
    assert [step["status"] for step in resumed.execution_plan["steps"]] == [
        "completed",
        "completed",
    ]
    assert "plan.step.completed" in [event.event_type for event in resumed.events]


def test_plan_resumes_from_portable_checkpoint_on_a_fresh_runtime() -> None:
    approval_spec = _tool(
        "actions.confirm",
        read_only=False,
        parallel_safe=False,
    ).model_copy(update={"requires_user_action": True})
    registry = ToolRegistry([approval_spec, _tool("facts.after")])
    executor = ToolExecutor(registry)
    executed: list[str] = []

    def confirm(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        executed.append(call.name)
        return ToolResult(
            call=call,
            status="requires_user_action",
            summary="Approval is required.",
            pending_action=PendingAction(
                id="portable-approval",
                run_id=context.run_id,
                tool_call_id=call.id,
                action_type="confirmation",
                prompt="Continue?",
            ),
        )

    def collect(call: ToolCall, _context: ToolExecutionContext) -> ToolResult:
        executed.append(call.name)
        return ToolResult(call=call, status="succeeded", summary="Evidence collected.")

    executor.register("actions.confirm", confirm)
    executor.register("facts.after", collect)
    first_runtime = (
        HarnessRuntimeBuilder(registry, executor)
        .with_ports(RuntimePorts(planning_enabled=True))
        .build()
    )
    provider = _ResumablePlanningProvider()
    base = {
        "question": "Run the portable workflow.",
        "run_id": "portable-plan-run",
    }
    waiting = first_runtime.run(RuntimeRequest(**base), provider=provider)

    replacement_runtime = (
        HarnessRuntimeBuilder(registry, executor)
        .with_ports(RuntimePorts(planning_enabled=True))
        .build()
    )
    resumed = replacement_runtime.run(
        RuntimeRequest(
            **base,
            short_context={
                "resume": True,
                "resume_observation": {
                    "status": "succeeded",
                    "summary": "Access approved on another worker.",
                },
                "previous_runtime": waiting.model_dump(mode="json"),
            },
        ),
        provider=provider,
    )

    assert resumed.status == "completed", resumed.model_dump(mode="json")
    assert resumed.diagnostics["recovery"]["checkpoint_source"] == "session_projection"
    assert executed == ["actions.confirm", "facts.after"]
    assert [step["status"] for step in resumed.execution_plan["steps"]] == [
        "completed",
        "completed",
    ]
