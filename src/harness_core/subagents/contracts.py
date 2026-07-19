from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


SubAgentModelRole = Literal["auto", "fast", "reasoning"]
SubAgentStatus = Literal["completed", "failed", "canceled"]


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


def delegation_tool_input_schema(
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
            "必须选择已注册且与任务领域匹配的专业 Agent。"
            + (f"\n当前可用 Agent：\n{catalog}" if catalog else "")
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
                "description": "本批专业协作的稳定标识。",
            },
            "max_concurrency": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
                "default": 3,
            },
            "tasks": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "description": "仅放入 1-3 个边界清晰、互不依赖的专业判断任务。",
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
                            "description": "该专业 Agent 要独立完成的明确判断目标。",
                        },
                        "input_data": {
                            "type": "object",
                            "description": "完成判断所需的事实输入，不要放提示词或模型配置。",
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
