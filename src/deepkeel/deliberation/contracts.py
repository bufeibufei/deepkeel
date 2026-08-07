from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


DeliberationPhase = Literal["opening", "rebuttal", "synthesis"]
DeliberationStatus = Literal["completed", "partial", "failed"]


class DeliberationParticipant(BaseModel):
    agent_id: str = Field(min_length=1, max_length=80)
    participant_instance_id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=80)
    display_name: str = Field(min_length=1, max_length=80)
    role: str = "participant"
    fact_keys: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeliberationSpec(BaseModel):
    deliberation_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=2000)
    facts: dict[str, Any]
    participants: list[DeliberationParticipant] = Field(min_length=2, max_length=3)
    moderator_agent_id: str = Field(min_length=1, max_length=80)
    max_rounds: int = Field(default=2, ge=1, le=3)
    max_model_calls: int = Field(default=12, ge=3, le=12)
    min_completed_participants: int = Field(default=1, ge=1, le=3)
    synthesis_reserve_calls: int = Field(default=1, ge=1, le=2)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_participants(self) -> "DeliberationSpec":
        agent_ids = [item.agent_id for item in self.participants]
        instance_ids = [item.participant_instance_id for item in self.participants]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("deliberation participants must use unique agents")
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("participant instance ids must be unique")
        if self.min_completed_participants > len(self.participants):
            raise ValueError("minimum completed participants exceeds participant count")
        return self


class DeliberationArgument(BaseModel):
    argument_id: str
    round_index: int
    phase: DeliberationPhase
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
    model_role: str = ""
    model_id: str = ""
    outcome: str = ""
    error: str = ""


class DeliberationResult(BaseModel):
    schema_version: str = "harness-deliberation-v2"
    deliberation_id: str
    status: DeliberationStatus
    stop_reason: str
    question: str
    participants: list[DeliberationParticipant]
    arguments: list[DeliberationArgument] = Field(default_factory=list)
    moderator: dict[str, Any] = Field(default_factory=dict)
    moderation_history: list[dict[str, Any]] = Field(default_factory=list)
    synthesis: dict[str, Any] = Field(default_factory=dict)
    model_calls: int = 0
    retry_count: int = 0
    diagnostics: dict[str, Any] = Field(default_factory=dict)
