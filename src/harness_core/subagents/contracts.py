from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


SubAgentModelRole = Literal["auto", "fast", "reasoning"]
SubAgentStatus = Literal["completed", "failed", "canceled", "needs_input"]
SubAgentExecutionMode = Literal["auto", "foreground", "background"]
SubAgentBatchStatus = Literal["completed", "partial", "failed", "canceled", "needs_input"]
SUBAGENT_EVENT_SCHEMA_VERSION = "harness-subagent-event-v1"


class SubAgentContextRef(BaseModel):
    """Opaque reference to context owned and resolved by the host."""

    id: str = Field(min_length=1, max_length=256)
    kind: str = Field(default="context", min_length=1, max_length=80)
    uri: str = ""
    summary: str = ""
    content_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_identifier(cls, value: Any) -> Any:
        return {"id": value} if isinstance(value, str) else value


class SubAgentArtifactRef(BaseModel):
    """Portable reference to an artifact without embedding its payload."""

    id: str = Field(min_length=1, max_length=256)
    artifact_type: str = Field(default="artifact", min_length=1, max_length=160)
    uri: str = ""
    version: str = ""
    content_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_identifier(cls, value: Any) -> Any:
        return {"id": value} if isinstance(value, str) else value


class SubAgentLineage(BaseModel):
    """Stable parent/child identity persisted with a delegated task or result."""

    root_run_id: str = ""
    parent_run_id: str = ""
    parent_task_id: str = ""
    delegation_id: str = ""
    child_run_id: str = ""
    depth: int = Field(default=1, ge=1, le=1)


class SubAgentBudget(BaseModel):
    """Optional task-level limits narrowed against the specialist and parent budgets."""

    max_model_calls: int | None = Field(default=None, ge=1, le=32)
    max_tool_calls: int | None = Field(default=None, ge=0, le=64)
    max_output_tokens: int | None = Field(default=None, ge=128, le=128_000)
    max_elapsed_seconds: float | None = Field(default=None, gt=0, le=3600)
    limits: dict[str, float] = Field(default_factory=dict)


class SubAgentCancellationPolicy(BaseModel):
    """Cooperative cancellation behavior for one bounded child task."""

    propagate_parent: bool = True
    discard_late_result: bool = True
    cancellation_key: str = ""


class SubAgentInputRequest(BaseModel):
    """Typed suspension payload returned when a specialist needs more user input."""

    prompt: str = Field(min_length=1, max_length=4000)
    requirements: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    resume_token: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubAgentSpec(BaseModel):
    """Stable, product-neutral contract for one bounded specialist."""

    id: str = Field(min_length=1, max_length=80)
    version: str = "1.0"
    label: str = Field(min_length=1, max_length=80)
    description: str = ""
    domain: str = ""
    system_prompt: str = ""
    model_role: SubAgentModelRole = "reasoning"
    capabilities: list[str] = Field(default_factory=list)
    tool_allowlist: list[str] = Field(default_factory=list)
    input_contract: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)
    context_policy: dict[str, Any] = Field(default_factory=dict)
    evidence_policy: dict[str, Any] = Field(default_factory=dict)
    failure_policy: Literal["partial", "fail_batch"] = "partial"
    read_only: bool = True
    allow_delegation: bool = False
    max_tool_rounds: int = Field(default=2, ge=0, le=3)
    max_tool_calls: int = Field(default=4, ge=0, le=8)
    max_model_calls: int | None = Field(default=None, ge=1, le=16)
    timeout_seconds: int = Field(default=90, ge=5, le=300)
    max_tokens: int = Field(default=1200, ge=128, le=8000)
    permission_scopes: list[str] = Field(default_factory=list)
    budget_limits: dict[str, float] = Field(default_factory=dict)
    cancellation_policy: SubAgentCancellationPolicy = Field(
        default_factory=SubAgentCancellationPolicy
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_bounded_execution(self) -> "SubAgentSpec":
        if self.allow_delegation:
            raise ValueError("subagents cannot delegate recursively")
        return self


class DelegationTask(BaseModel):
    """Durable task brief passed from the lead agent to one specialist.

    ``TaskBrief`` is the preferred public name. ``DelegationTask`` remains the
    concrete class so existing imports, persisted payloads, and type checks keep
    working unchanged.
    """

    id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=128)
    agent_id: str = Field(min_length=1, max_length=80)
    objective: str = Field(min_length=1, max_length=2000)
    normalized_question: str = ""
    input_data: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    context_refs: list[SubAgentContextRef] = Field(default_factory=list)
    artifact_refs: list[SubAgentArtifactRef] = Field(default_factory=list)
    expected_output: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = ""
    lineage: SubAgentLineage = Field(default_factory=SubAgentLineage)
    budget: SubAgentBudget = Field(default_factory=SubAgentBudget)
    timeout_seconds: float | None = Field(default=None, gt=0, le=3600)
    cancellation: SubAgentCancellationPolicy = Field(
        default_factory=SubAgentCancellationPolicy
    )
    model_role: SubAgentModelRole = "auto"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_id", "objective", "normalized_question", "idempotency_key")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return str(value or "").strip()

    @property
    def effective_idempotency_key(self) -> str:
        """Return the explicit idempotency key or the legacy task identifier."""

        return self.idempotency_key or self.id

    def bind_lineage(
        self,
        *,
        root_run_id: str,
        parent_run_id: str,
        delegation_id: str,
        depth: int,
        child_run_id: str = "",
    ) -> "DelegationTask":
        current = self.lineage
        return self.model_copy(
            update={
                "lineage": current.model_copy(
                    update={
                        "root_run_id": current.root_run_id or root_run_id,
                        "parent_run_id": current.parent_run_id or parent_run_id,
                        "delegation_id": current.delegation_id or delegation_id,
                        "child_run_id": current.child_run_id or child_run_id,
                        "depth": depth,
                    }
                )
            }
        )


# Preferred product-neutral name; retained as an alias to preserve concrete type identity.
TaskBrief = DelegationTask


class DelegationRequest(BaseModel):
    delegation_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    root_run_id: str = ""
    parent_run_id: str = ""
    depth: int = Field(default=1, ge=1, le=1)
    max_concurrency: int = Field(default=3, ge=1, le=3)
    execution_mode: SubAgentExecutionMode = "auto"
    tasks: list[DelegationTask] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_unique_task_ids_and_keys(self) -> "DelegationRequest":
        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("delegation task ids must be unique")
        idempotency_keys = [task.effective_idempotency_key for task in self.tasks]
        if len(idempotency_keys) != len(set(idempotency_keys)):
            raise ValueError("delegation task idempotency keys must be unique")
        return self


class SubAgentResult(BaseModel):
    task_id: str
    agent_id: str
    child_run_id: str
    status: SubAgentStatus
    outcome: Literal["completed", "degraded", "needs_input"] = "completed"
    conclusion: str = ""
    evidence: list[str] = Field(default_factory=list)
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    context_refs: list[SubAgentContextRef] = Field(default_factory=list)
    artifact_refs: list[SubAgentArtifactRef] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    abstained: bool = False
    input_request: SubAgentInputRequest | None = None
    idempotency_key: str = ""
    lineage: SubAgentLineage = Field(default_factory=SubAgentLineage)
    output: dict[str, Any] = Field(default_factory=dict)
    model_role: str = ""
    model_id: str = ""
    duration_ms: int = 0
    error: str = ""
    raw_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def promote_legacy_metadata_refs(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        promoted = dict(value)
        metadata = promoted.get("metadata")
        if isinstance(metadata, dict) and not promoted.get("artifact_refs"):
            promoted["artifact_refs"] = metadata.get("artifact_refs") or []
        return promoted

    @model_validator(mode="after")
    def validate_input_suspension(self) -> "SubAgentResult":
        if self.status == "needs_input":
            if self.input_request is None:
                raise ValueError("needs_input results require input_request")
            if self.outcome != "needs_input":
                self.outcome = "needs_input"
        return self

    def parent_projection(self) -> dict[str, Any]:
        """Bounded result injected into the lead agent context."""

        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "child_run_id": self.child_run_id,
            "status": self.status,
            "outcome": self.outcome,
            "conclusion": self.conclusion,
            "evidence": list(self.evidence),
            "evidence_refs": list(self.evidence_refs),
            "context_refs": [item.model_dump(mode="json") for item in self.context_refs],
            "risks": list(self.risks),
            "recommendations": list(self.recommendations),
            "warnings": list(self.warnings),
            "confidence": self.confidence,
            "abstained": self.abstained,
            "artifact_refs": [item.model_dump(mode="json") for item in self.artifact_refs],
            "input_request": (
                self.input_request.model_dump(mode="json") if self.input_request else None
            ),
            "idempotency_key": self.idempotency_key,
            "lineage": self.lineage.model_dump(mode="json"),
            "error": self.error,
        }


class DelegationBatchResult(BaseModel):
    delegation_id: str
    root_run_id: str = ""
    parent_run_id: str = ""
    status: SubAgentBatchStatus
    results: list[SubAgentResult] = Field(default_factory=list)
    duration_ms: int = 0

    @property
    def successful_results(self) -> list[SubAgentResult]:
        return [result for result in self.results if result.status == "completed"]

    @property
    def pending_input_results(self) -> list[SubAgentResult]:
        return [result for result in self.results if result.status == "needs_input"]

    def parent_payload(self) -> dict[str, Any]:
        return {
            "delegation_id": self.delegation_id,
            "root_run_id": self.root_run_id,
            "parent_run_id": self.parent_run_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "results": [result.parent_projection() for result in self.results],
        }


class SubAgentEventFields(BaseModel):
    """Canonical payload fields shared by all SubAgent runtime events."""

    schema_version: Literal["harness-subagent-event-v1"] = "harness-subagent-event-v1"
    delegation_id: str = ""
    task_id: str = ""
    agent_id: str = ""
    agent_label: str = ""
    child_run_id: str = ""
    root_run_id: str = ""
    parent_run_id: str = ""
    parent_task_id: str = ""
    idempotency_key: str = ""
    cancellation_key: str = ""
    spec_version: str = ""
    status: str = ""
    attempt: int = Field(default=1, ge=1)
    visible: bool = False
    model_role: str = ""
    model_id: str = ""
    duration_ms: int = Field(default=0, ge=0)
    timeout_seconds: float | None = Field(default=None, gt=0)
    budget: SubAgentBudget | None = None
    artifact_refs: list[SubAgentArtifactRef] = Field(default_factory=list)
    error: str = ""
    reason_code: str = ""


def delegation_tool_parameters_schema(
    agents: list[SubAgentSpec] | None = None,
) -> dict[str, Any]:
    """Return the provider-facing contract for bounded first-level delegation."""

    available = list(agents or [])
    agent_ids = [agent.id for agent in available]
    catalog = "\n".join(
        f"- {agent.id}: {agent.label}. {agent.description}".rstrip(". ")
        for agent in available
    )
    agent_id_schema: dict[str, Any] = {
        "type": "string",
        "minLength": 1,
        "description": (
            "Select a registered specialist agent that matches the task domain."
            + (f"\nAvailable agents:\n{catalog}" if catalog else "")
        ),
    }
    if agent_ids:
        agent_id_schema["enum"] = agent_ids

    return {
        "type": "object",
        "properties": {
            "delegation_id": {
                "type": "string",
                "minLength": 1,
                "description": "Stable identifier for this specialist batch.",
            },
            "max_concurrency": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
                "default": 3,
            },
            "execution_mode": {
                "type": "string",
                "enum": ["auto", "foreground", "background"],
                "default": "auto",
                "description": (
                    "Use background for long independent tasks and foreground for "
                    "short tasks whose result is needed immediately."
                ),
            },
            "tasks": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "description": "Include 1-3 bounded and independent specialist tasks.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1, "maxLength": 128},
                        "agent_id": agent_id_schema,
                        "objective": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 2000,
                            "description": "A precise objective for the specialist agent.",
                        },
                        "normalized_question": {"type": "string"},
                        "input_data": {
                            "type": "object",
                            "description": (
                                "Facts required for the task; exclude prompts and model "
                                "configuration."
                            ),
                            "additionalProperties": True,
                        },
                        "constraints": {"type": "array", "items": {"type": "string"}},
                        "context_refs": {
                            "type": "array",
                            "items": {"type": "object", "additionalProperties": True},
                        },
                        "artifact_refs": {
                            "type": "array",
                            "items": {"type": "object", "additionalProperties": True},
                        },
                        "expected_output": {"type": "object", "additionalProperties": True},
                        "idempotency_key": {"type": "string"},
                        "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
                        "budget": {
                            "type": "object",
                            "properties": {
                                "max_model_calls": {"type": "integer", "minimum": 1},
                                "max_tool_calls": {"type": "integer", "minimum": 0},
                                "max_output_tokens": {"type": "integer", "minimum": 128},
                                "max_elapsed_seconds": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                },
                                "limits": {"type": "object"},
                            },
                            "additionalProperties": False,
                        },
                        "cancellation": {
                            "type": "object",
                            "properties": {
                                "propagate_parent": {"type": "boolean"},
                                "discard_late_result": {"type": "boolean"},
                                "cancellation_key": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                        "model_role": {
                            "type": "string",
                            "enum": ["auto", "fast", "reasoning"],
                            "default": "auto",
                        },
                    },
                    "required": ["id", "agent_id", "objective"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["tasks"],
        "additionalProperties": False,
    }
