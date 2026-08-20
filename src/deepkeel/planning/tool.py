from __future__ import annotations

from typing import Any

from deepkeel.contracts import Observation, ToolCall, ToolResult
from deepkeel.planning.constants import PLAN_TOOL_NAME
from deepkeel.planning.contracts import ExecutionPlan, PlanStep, PlanningPolicy
from deepkeel.planning.validator import ExecutionPlanValidator, PlanValidationError
from deepkeel.skills import SkillPolicy
from deepkeel.tool_registry import ToolRegistry, ToolSpec
from deepkeel.tools import ToolExecutionContext, ToolExecutor


def execution_planning_prompt(skill_snapshot: dict[str, Any] | None) -> str:
    policy = PlanningPolicy.from_snapshot(skill_snapshot)
    if not policy.enabled:
        return ""
    preference = {
        "allowed": "You may",
        "preferred": "Prefer to",
        "required": "You must",
    }[policy.mode]
    return (
        "Execution planning:\n"
        f"- {preference} call runtime.create_plan before business tools when the request "
        "requires multiple capabilities, dependent stages, or parallel evidence collection.\n"
        "- Do not create a plan for greetings, one-step lookups, or direct answers.\n"
        "- Submit the complete bounded DAG in one standalone control-tool call. Each executable "
        "step must reference a currently available tool; use a synthesis step only when useful.\n"
        "- The runtime executes ready steps, preserves completed work, and returns control for "
        "synthesis or bounded replanning. Never expose hidden reasoning or the raw plan JSON."
    )


def execution_plan_tool_spec() -> ToolSpec:
    step_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "minLength": 1, "maxLength": 96},
            "title": {"type": "string", "minLength": 1, "maxLength": 160},
            "objective": {"type": "string", "minLength": 1, "maxLength": 1200},
            "executor_kind": {
                "type": "string",
                "enum": ["tool", "workflow", "subagent", "synthesis"],
                "default": "tool",
            },
            "capability_ref": {
                "type": "string",
                "description": "Registered tool name; omit only for synthesis steps.",
            },
            "arguments": {"type": "object", "additionalProperties": True},
            "depends_on": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "success_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "max_attempts": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": ["id", "title", "objective"],
        "additionalProperties": False,
    }
    return ToolSpec(
        name=PLAN_TOOL_NAME,
        description=(
            "Create or revise a bounded execution DAG for a genuinely multi-step request. "
            "Use only when several tools, dependencies, parallel evidence, or durable workflows "
            "must be coordinated; never use for a simple one-step response."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "objective": {"type": "string", "minLength": 1, "maxLength": 2000},
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 16,
                    "items": step_schema,
                },
                "reason": {
                    "type": "string",
                    "maxLength": 1000,
                    "description": "Why a prior plan must be revised; blank for a new plan.",
                },
            },
            "required": ["objective", "steps"],
            "additionalProperties": False,
        },
        required_args=["objective", "steps"],
        read_only=True,
        parallel_safe=False,
        visible_label="Preparing an execution plan",
        exposure_mode="baseline",
        discovery_tags=["plan", "workflow", "orchestration", "multi-step"],
        usage_policy={
            "when_to_use": "The request needs multiple tools, dependencies, or parallel evidence.",
            "when_not_to_use": "The request is conversational, direct, or needs only one tool.",
        },
        runtime_policy={
            "internal_runtime_tool": True,
            "plan_control": True,
            "start_event_visible": False,
        },
    )


def install_execution_planning(registry: ToolRegistry, executor: ToolExecutor) -> None:
    if PLAN_TOOL_NAME not in {spec.name for spec in registry.list_tools()}:
        registry.register(execution_plan_tool_spec())
    validator = ExecutionPlanValidator(registry)

    def create_plan(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        skill_snapshot = context.metadata.get("skill_activation")
        skill = SkillPolicy.from_snapshot(
            skill_snapshot if isinstance(skill_snapshot, dict) else {}
        )
        policy = PlanningPolicy.from_snapshot(
            skill_snapshot if isinstance(skill_snapshot, dict) else {}
        )
        allowed_names = (
            set(skill.allowed_tools)
            if skill.active and skill.tool_scope_mode == "allowlist"
            else None
        )
        try:
            plan = ExecutionPlan(
                plan_id=f"plan:{context.run_id}:{call.id}",
                run_id=context.run_id,
                objective=str(call.arguments.get("objective") or "").strip(),
                revision_reason=str(call.arguments.get("reason") or "").strip(),
                steps=[
                    PlanStep.model_validate(item)
                    for item in call.arguments.get("steps", [])
                    if isinstance(item, dict)
                ],
            )
            plan = validator.validate(
                plan,
                policy=policy,
                allowed_names=allowed_names,
            )
        except (PlanValidationError, ValueError, TypeError) as exc:
            return ToolResult(
                call=call,
                status="failed",
                summary="The proposed execution plan is invalid.",
                error=str(exc),
                retryable=True,
                observation=Observation(
                    id=f"{call.id}:plan-invalid",
                    run_id=context.run_id,
                    tool_call_id=call.id,
                    source=PLAN_TOOL_NAME,
                    status="failed",
                    summary="The proposed execution plan is invalid.",
                    error=str(exc),
                    metadata={"visible": False, "runtime_internal": True},
                ),
                metadata={"visible": False, "runtime_internal": True},
            )
        payload = plan.model_dump(mode="json")
        return ToolResult(
            call=call,
            status="succeeded",
            summary="Execution plan prepared.",
            data={"execution_plan": payload},
            observation=Observation(
                id=f"{call.id}:plan",
                run_id=context.run_id,
                tool_call_id=call.id,
                source=PLAN_TOOL_NAME,
                status="succeeded",
                summary="Execution plan prepared.",
                data={"execution_plan": payload},
                metadata={"visible": False, "runtime_internal": True},
            ),
            metadata={"visible": False, "runtime_internal": True},
        )

    executor.register(PLAN_TOOL_NAME, create_plan)


__all__ = [
    "PLAN_TOOL_NAME",
    "execution_plan_tool_spec",
    "execution_planning_prompt",
    "install_execution_planning",
]
