from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DeliberationParticipant(BaseModel):
    agent_id: str = Field(min_length=1, max_length=80)
    participant_instance_id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=80)
    display_name: str = Field(min_length=1, max_length=80)
    role: str = "participant"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeliberationSpec(BaseModel):
    deliberation_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=2000)
    fact_packet: dict[str, Any]
    participants: list[DeliberationParticipant] = Field(min_length=2, max_length=3)
    moderator_agent_id: str = Field(min_length=1, max_length=80)
    max_rounds: int = Field(default=2, ge=1, le=3)
    max_model_calls: int = Field(default=12, ge=3, le=12)

    @model_validator(mode="after")
    def validate_participants(self) -> "DeliberationSpec":
        agent_ids = [item.agent_id for item in self.participants]
        instance_ids = [item.participant_instance_id for item in self.participants]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("deliberation participants must use unique agents")
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("participant instance ids must be unique")
        return self


class DeliberationArgument(BaseModel):
    argument_id: str
    round_index: int
    phase: Literal["opening", "rebuttal", "synthesis"]
    participant_instance_id: str
    agent_id: str
    display_name: str
    label: str
    status: str
    conclusion: str = ""
    evidence: list[str] = Field(default_factory=list)
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float | None = None
    duration_ms: int = 0
    child_run_id: str = ""
    error: str = ""


class DeliberationResult(BaseModel):
    schema_version: str = "harness-deliberation-v2"
    deliberation_id: str
    status: Literal["completed", "partial", "failed"]
    stop_reason: str
    question: str
    participants: list[DeliberationParticipant]
    arguments: list[DeliberationArgument] = Field(default_factory=list)
    moderator: dict[str, Any] = Field(default_factory=dict)
    moderation_history: list[dict[str, Any]] = Field(default_factory=list)
    synthesis: dict[str, Any] = Field(default_factory=dict)
    model_calls: int = 0
    retry_count: int = 0
