from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from deepkeel.contracts import utc_now


PlanningMode = Literal["disabled", "allowed", "preferred", "required"]
PlanStatus = Literal[
    "proposed",
    "running",
    "waiting",
    "replanning",
    "synthesizing",
    "completed",
    "partially_completed",
    "failed",
    "canceled",
]
PlanStepStatus = Literal[
    "pending",
    "running",
    "waiting",
    "completed",
    "failed",
    "skipped",
    "canceled",
]
PlanExecutorKind = Literal["tool", "workflow", "subagent", "synthesis"]


class PlanningPolicy(BaseModel):
    """Bounded policy controlling when and how a model may create a plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: PlanningMode = "allowed"
    max_steps: int = Field(default=8, ge=1, le=16)
    max_revisions: int = Field(default=2, ge=0, le=4)
    max_parallel_steps: int = Field(default=4, ge=1, le=8)
    max_attempts_per_step: int = Field(default=2, ge=1, le=3)

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any] | None) -> "PlanningPolicy":
        raw = snapshot if isinstance(snapshot, dict) else {}
        configured = raw.get("planning_policy")
        policy = configured if isinstance(configured, dict) else {}
        return cls.model_validate(policy)


class PlanStep(BaseModel):
    """One bounded, replayable unit in an execution plan."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=96)
    title: str = Field(min_length=1, max_length=160)
    objective: str = Field(min_length=1, max_length=1200)
    executor_kind: PlanExecutorKind = "tool"
    capability_ref: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    status: PlanStepStatus = "pending"
    attempt_count: int = Field(default=0, ge=0, le=3)
    max_attempts: int = Field(default=1, ge=1, le=3)
    tool_call_id: str = ""
    result_summary: str = ""
    error: str = ""
    artifact_ids: list[str] = Field(default_factory=list)
    read_only: bool | None = None
    parallel_safe: bool | None = None
    resource_key: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("depends_on", "success_criteria", "artifact_ids")
    @classmethod
    def unique_non_blank_values(cls, values: list[str]) -> list[str]:
        normalized = [str(value or "").strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("plan step lists must not contain blank values")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_step_shape(self) -> "PlanStep":
        if self.id in self.depends_on:
            raise ValueError("plan step cannot depend on itself")
        if self.executor_kind == "synthesis":
            if self.capability_ref:
                raise ValueError("synthesis step must not declare capability_ref")
        elif not self.capability_ref.strip():
            raise ValueError("executable plan step requires capability_ref")
        return self

    def immutable_signature(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "objective": self.objective,
            "executor_kind": self.executor_kind,
            "capability_ref": self.capability_ref,
            "arguments": self.arguments,
            "depends_on": self.depends_on,
            "success_criteria": self.success_criteria,
        }


class ExecutionPlan(BaseModel):
    """Durable DAG selected by the model and executed by the existing runtime."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["deepkeel-execution-plan-v1"] = "deepkeel-execution-plan-v1"
    plan_id: str = Field(min_length=1, max_length=160)
    run_id: str = Field(min_length=1)
    objective: str = Field(min_length=1, max_length=2000)
    revision: int = Field(default=1, ge=1)
    status: PlanStatus = "proposed"
    steps: list[PlanStep] = Field(min_length=1, max_length=16)
    revision_reason: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_step_identities(self) -> "ExecutionPlan":
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("execution plan step ids must be unique")
        return self

    @property
    def completed_step_count(self) -> int:
        return sum(step.status in {"completed", "skipped"} for step in self.steps)

    @property
    def progress(self) -> dict[str, int]:
        return {
            "completed": self.completed_step_count,
            "total": len(self.steps),
        }


class PlanPatch(BaseModel):
    """Optimistic full-plan replacement used for bounded replanning."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1)
    expected_revision: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=1000)
    objective: str = ""
    steps: list[PlanStep] = Field(min_length=1, max_length=16)


__all__ = [
    "ExecutionPlan",
    "PlanExecutorKind",
    "PlanPatch",
    "PlanStatus",
    "PlanStep",
    "PlanStepStatus",
    "PlanningMode",
    "PlanningPolicy",
]
