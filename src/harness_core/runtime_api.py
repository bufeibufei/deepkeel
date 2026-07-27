from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from harness_core.contracts import (
    Artifact,
    FinalAnswer,
    Observation,
    PendingAction,
    RunContext,
    TaskLifecycle,
    ToolResult,
)
from harness_core.scope import RuntimeScope, resolve_runtime_scope
from harness_core.artifact_views import ArtifactView
from harness_core.references import EvidenceBundle
from harness_core.version import EVENT_SCHEMA_VERSION


class RuntimeActiveTask(TypedDict):
    kind: str
    tool_name: str
    artifact_type: str
    source_id: str
    summary: str


class RuntimeUIState(TypedDict):
    schema_version: str
    lifecycle: TaskLifecycle
    execution_status: str
    composer_mode: str
    can_send: bool
    input_strategy: str
    requires_user_action: bool
    is_resumable: bool
    show_progress: bool
    can_cancel: bool
    active_task: RuntimeActiveTask | None
    reason: str


class RuntimeReference(TypedDict):
    reference_id: str
    kind: str
    title: str
    snippet: str
    source_tool: str
    query: str
    is_evidence: bool
    unit_id: NotRequired[str]
    url: NotRequired[str]
    site_name: NotRequired[str]
    publish_time: NotRequired[str]
    flow: NotRequired[str]
    quality_grade: NotRequired[str]
    page: NotRequired[str]
    chapter: NotRequired[str]
    source_file: NotRequired[str]


class RuntimeErrorPayload(TypedDict):
    type: str
    code: str
    category: str
    retryable: bool
    message: str


class RuntimeRequest(BaseModel):
    """Serializable input owned by Core rather than a product host."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    user_id: str = "local-device"
    tenant_id: str = ""
    namespace: str = "default"
    scope: RuntimeScope | None = None
    run_id: str = ""
    thread_id: str = ""
    turn_id: str = ""
    short_context: dict[str, Any] = Field(default_factory=dict)
    context_bundle: dict[str, Any] = Field(default_factory=dict)
    skill_activation: dict[str, Any] = Field(default_factory=dict)
    model_policy: dict[str, Any] = Field(default_factory=dict)

    @property
    def runtime_scope(self) -> RuntimeScope:
        return resolve_runtime_scope(
            self.scope,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            namespace=self.namespace,
        )


class RuntimeResultStatus(StrEnum):
    RUNNING = "running"
    WAITING_USER_ACTION = "waiting_user_action"
    WAITING_USER_INPUT = "waiting_user_input"
    TASK_RUNNING = "task_running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class RuntimeEventEnvelope(BaseModel):
    """Cursor-addressable event emitted by the canonical runtime."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = EVENT_SCHEMA_VERSION
    event_id: str = ""
    sequence: int = Field(default=0, ge=0)
    run_version: int = Field(default=0, ge=0)
    run_id: str = ""
    thread_id: str = ""
    turn_id: str = ""
    tenant_id: str = ""
    user_id: str = ""
    namespace: str = "default"
    visibility: Literal["public", "internal"] = "internal"
    event_type: str = Field(min_length=1)
    source_event_type: str = ""
    title: str = ""
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    ephemeral: bool = False
    created_at: datetime | None = None

    @property
    def cursor(self) -> str:
        return f"{self.run_id}:{self.sequence}" if self.run_id and self.sequence else ""


class RuntimeStreamEvent(RuntimeEventEnvelope):
    """Backward-compatible name for streaming API consumers."""


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
    run_context: RunContext
    observations: list[Observation] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    pending_action: PendingAction | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    artifact_views: list[ArtifactView] = Field(default_factory=list)
    events: list[RuntimeStreamEvent] = Field(default_factory=list)
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    skill_activation: dict[str, Any] = Field(default_factory=dict)
    active_task: RuntimeActiveTask | None = None
    ui_state: RuntimeUIState
    references: list[RuntimeReference] = Field(default_factory=list)
    evidence: list[RuntimeReference] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)
    needs_user_input: bool = False
    answer_delta_streamed: bool = False
    error: RuntimeErrorPayload | None = None
