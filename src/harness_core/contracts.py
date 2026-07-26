from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ObservationStatus = Literal["pending", "succeeded", "failed", "requires_user_action"]
ResultOutcome = Literal["completed", "partial", "degraded", "skipped", "canceled"]
ToolResultStatus = Literal["succeeded", "failed", "requires_user_action", "waiting_async"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunStatus(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    REASONING = "reasoning"
    EXECUTING_TOOLS = "executing_tools"
    WAITING_USER = "waiting_user"
    WAITING_ASYNC = "waiting_async"
    STREAMING_ANSWER = "streaming_answer"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskLifecycle(StrEnum):
    """Product-neutral lifecycle exposed to hosts and user interfaces."""

    COLLECTING_INPUT = "collecting_input"
    WAITING_USER_ACTION = "waiting_user_action"
    QUEUED = "queued"
    RUNNING = "running"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class AgentMessage(ContractModel):
    id: str = Field(min_length=1)
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str = ""
    tool_call_id: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(ContractModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = ""
    resource_key: str = ""
    read_only: bool = False
    parallel_safe: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PendingAction(ContractModel):
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    tool_call_id: str = ""
    action_type: str = Field(min_length=1)
    status: Literal["pending", "resolved", "canceled"] = "pending"
    title: str = ""
    prompt: str = ""
    handoff_view: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    resolution: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None


class Artifact(ContractModel):
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    title: str = ""
    summary: str = ""
    source_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(ContractModel):
    id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    tool_call_id: str = ""
    source: str = Field(min_length=1)
    status: ObservationStatus
    outcome: ResultOutcome | None = None
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    artifact_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(ContractModel):
    tool_call_id: str = ""
    name: str = ""
    status: ToolResultStatus
    outcome: ResultOutcome | None = None
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    retryable: bool = False
    call: ToolCall | None = Field(default=None, exclude=True)
    observation: Observation | None = None
    pending_action: PendingAction | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_call_identity(self) -> ToolResult:
        if self.call is not None:
            if self.tool_call_id and self.tool_call_id != self.call.id:
                raise ValueError("tool_call_id must match call.id")
            if self.name and self.name != self.call.name:
                raise ValueError("name must match call.name")
            if not self.tool_call_id:
                self.tool_call_id = self.call.id
            if not self.name:
                self.name = self.call.name
        if not self.tool_call_id:
            raise ValueError("tool_call_id is required")
        if not self.name:
            raise ValueError("name is required")
        correlated = [
            item
            for item in (self.observation, self.pending_action)
            if item is not None
        ]
        for item in correlated:
            if item.tool_call_id and item.tool_call_id != self.tool_call_id:
                raise ValueError(
                    f"{type(item).__name__}.tool_call_id must match tool_call_id"
                )
        run_ids = {
            item.run_id
            for item in [*correlated, *self.artifacts]
        }
        if len(run_ids) > 1:
            raise ValueError("tool result projections must belong to one run")
        _require_unique_ids(self.artifacts, field_name="artifacts")
        return self


class FinalAnswer(ContractModel):
    markdown: str = ""
    summary: str = ""
    status: Literal["completed", "failed", "interrupted"] = "completed"
    artifact_ids: list[str] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)
    model_role: str = ""
    model_id: str = ""
    stop_reason: str = "completed"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeEvent(ContractModel):
    run_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    sequence: int | None = None
    title: str = ""
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    ephemeral: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_sequence(self) -> RuntimeEvent:
        if not self.ephemeral and (self.sequence is None or self.sequence < 1):
            raise ValueError("persistent runtime events require a positive sequence")
        if self.sequence is not None and self.sequence < 1:
            raise ValueError("sequence must be positive")
        return self


class RunContext(ContractModel):
    run_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    status: RunStatus = RunStatus.QUEUED
    messages: list[AgentMessage] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    pending_tool_calls: list[ToolCall] = Field(default_factory=list)
    pending_action: PendingAction | None = None
    pending_async: dict[str, Any] | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    skill_activation: dict[str, Any] = Field(default_factory=dict)
    model_policy: dict[str, Any] = Field(default_factory=dict)
    budget_state: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    step_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_correlations(self) -> RunContext:
        for observation in self.observations:
            if observation.run_id != self.run_id:
                raise ValueError("observation.run_id must match run_id")
        for artifact in self.artifacts:
            if artifact.run_id != self.run_id:
                raise ValueError("artifact.run_id must match run_id")
        if self.pending_action is not None and self.pending_action.run_id != self.run_id:
            raise ValueError("pending_action.run_id must match run_id")

        _require_unique_ids(self.messages, field_name="messages")
        _require_unique_ids(self.observations, field_name="observations")
        _require_unique_ids(self.pending_tool_calls, field_name="pending_tool_calls")
        _require_unique_ids(self.artifacts, field_name="artifacts")
        return self


def _require_unique_ids(items: list[Any], *, field_name: str) -> None:
    identifiers = [str(item.id) for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{field_name} must have unique ids")
