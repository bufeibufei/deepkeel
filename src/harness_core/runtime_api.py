from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from harness_core.contracts import Artifact, FinalAnswer, Observation, PendingAction, ToolResult


class RuntimeRequest(BaseModel):
    """Serializable input owned by Core rather than a product host."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    user_id: str = "local-device"
    short_context: dict[str, Any] = Field(default_factory=dict)
    context_bundle: dict[str, Any] = Field(default_factory=dict)
    skill_activation: dict[str, Any] = Field(default_factory=dict)
    model_policy: dict[str, Any] = Field(default_factory=dict)


class RuntimeResultStatus(StrEnum):
    RUNNING = "running"
    WAITING_USER_ACTION = "waiting_user_action"
    WAITING_USER_INPUT = "waiting_user_input"
    TASK_RUNNING = "task_running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class RuntimeStreamEvent(BaseModel):
    """Event emitted while a run is executing, before host persistence metadata."""

    model_config = ConfigDict(extra="allow")

    event_type: str = Field(min_length=1)
    source_event_type: str = ""
    title: str = ""
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    ephemeral: bool = False
    created_at: datetime | None = None


class RuntimeResult(BaseModel):
    """Canonical typed result returned by the public runtime API."""

    model_config = ConfigDict(extra="forbid")

    question: str
    run_id: str
    thread_id: str
    graph_thread_id: str
    turn_id: str
    status: RuntimeResultStatus
    stop_reason: str
    schema_version: str = "harness-runtime-v2"
    core_contract_version: str = ""
    core_version: str = ""
    loop_engine: str = ""
    mode: str = ""
    step_count: int = 0
    final_answer: FinalAnswer
    observations: list[Observation] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    pending_action: PendingAction | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    events: list[RuntimeStreamEvent] = Field(default_factory=list)
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    skill_activation: dict[str, Any] = Field(default_factory=dict)
    active_task: dict[str, Any] | None = None
    ui_state: dict[str, Any] = Field(default_factory=dict)
    references: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    needs_user_input: bool = False
    answer_delta_streamed: bool = False
    error: dict[str, Any] | None = None
