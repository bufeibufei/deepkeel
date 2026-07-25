from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


SubAgentModelRole = Literal["auto", "fast", "reasoning"]
SubAgentStatus = Literal["completed", "failed", "canceled"]
SubAgentExecutionMode = Literal["auto", "foreground", "background"]


class SubAgentSpec(BaseModel):
    """Stable, product-neutral contract for one bounded specialist."""

    id: str = Field(min_length=1, max_length=80)
    version: str = "1.0"
    label: str = Field(min_length=1, max_length=80)
    description: str = ""
    system_prompt: str = ""
    model_role: SubAgentModelRole = "reasoning"
    capabilities: list[str] = Field(default_factory=list)
    tool_allowlist: list[str] = Field(default_factory=list)
    input_contract: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)
    evidence_policy: dict[str, Any] = Field(default_factory=dict)
    failure_policy: Literal["partial", "fail_batch"] = "partial"
    read_only: bool = True
    allow_delegation: bool = False
    max_tool_rounds: int = Field(default=2, ge=0, le=3)
    max_tool_calls: int = Field(default=4, ge=0, le=8)
    timeout_seconds: int = Field(default=90, ge=5, le=300)
    max_tokens: int = Field(default=1200, ge=128, le=8000)
    permission_scopes: list[str] = Field(default_factory=list)
    budget_limits: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_bounded_execution(self) -> "SubAgentSpec":
        if self.allow_delegation:
            raise ValueError("subagents cannot delegate recursively")
        return self


class DelegationTask(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=128)
    agent_id: str = Field(min_length=1, max_length=80)
    objective: str = Field(min_length=1, max_length=2000)
    input_data: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    model_role: SubAgentModelRole = "auto"

    @field_validator("agent_id", "objective")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return str(value or "").strip()


class DelegationRequest(BaseModel):
    delegation_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    root_run_id: str = ""
    parent_run_id: str = ""
    depth: int = Field(default=1, ge=1, le=1)
    max_concurrency: int = Field(default=3, ge=1, le=3)
    execution_mode: SubAgentExecutionMode = "auto"
    tasks: list[DelegationTask] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_unique_task_ids(self) -> "DelegationRequest":
        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("delegation task ids must be unique")
        return self


class SubAgentResult(BaseModel):
    task_id: str
    agent_id: str
    child_run_id: str
    status: SubAgentStatus
    outcome: Literal["completed", "degraded"] = "completed"
    conclusion: str = ""
    evidence: list[str] = Field(default_factory=list)
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    abstained: bool = False
    output: dict[str, Any] = Field(default_factory=dict)
    model_role: str = ""
    model_id: str = ""
    duration_ms: int = 0
    error: str = ""
    raw_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

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
            "risks": list(self.risks),
            "recommendations": list(self.recommendations),
            "warnings": list(self.warnings),
            "confidence": self.confidence,
            "abstained": self.abstained,
            "artifact_refs": list(self.metadata.get("artifact_refs") or []),
            "error": self.error,
        }


class DelegationBatchResult(BaseModel):
    delegation_id: str
    root_run_id: str = ""
    parent_run_id: str = ""
    status: Literal["completed", "partial", "failed", "canceled"]
    results: list[SubAgentResult] = Field(default_factory=list)
    duration_ms: int = 0

    @property
    def successful_results(self) -> list[SubAgentResult]:
        return [result for result in self.results if result.status == "completed"]

    def parent_payload(self) -> dict[str, Any]:
        return {
            "delegation_id": self.delegation_id,
            "root_run_id": self.root_run_id,
            "parent_run_id": self.parent_run_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "results": [result.parent_projection() for result in self.results],
        }


def delegation_tool_parameters_schema(
    agents: list[SubAgentSpec] | None = None,
) -> dict[str, Any]:
    """Return the provider-facing contract for bounded first-level delegation."""

    available = list(agents or [])
    agent_ids = [agent.id for agent in available]
    catalog = "\n".join(
        f"- {agent.id}: {agent.label}。{agent.description}".rstrip("。")
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
                        "id": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 128,
                        },
                        "agent_id": agent_id_schema,
                        "objective": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 2000,
                            "description": "A precise objective for the specialist agent.",
                        },
                        "input_data": {
                            "type": "object",
                            "description": "Facts required for the task; exclude prompts and model configuration.",
                            "additionalProperties": True,
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
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
