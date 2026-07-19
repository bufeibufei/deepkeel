from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from harness_core.contracts import Artifact, FinalAnswer, Observation, PendingAction


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
    final_answer: FinalAnswer
    observations: list[Observation] = Field(default_factory=list)
    pending_action: PendingAction | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    events: list[RuntimeStreamEvent] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    skill_activation: dict[str, Any] = Field(default_factory=dict)
    answer_delta_streamed: bool = False
    error: dict[str, Any] | None = None
    compatibility_payload: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    @classmethod
    def from_compatibility_payload(cls, payload: dict[str, Any]) -> RuntimeResult:
        runtime = payload.get("agent_runtime") if isinstance(payload.get("agent_runtime"), dict) else {}
        runtime_status = RuntimeResultStatus(str(runtime.get("status") or "failed"))
        checkpoint = runtime.get("checkpoint") if isinstance(runtime.get("checkpoint"), dict) else {}
        identity = runtime.get("identity") if isinstance(runtime.get("identity"), dict) else {}
        answer_payload = payload.get("final_answer") if isinstance(payload.get("final_answer"), dict) else {}
        answer_metadata = (
            dict(answer_payload.get("metadata"))
            if isinstance(answer_payload.get("metadata"), dict)
            else {}
        )
        answer_fields = set(FinalAnswer.model_fields)
        answer_metadata.update(
            {
                key: value
                for key, value in answer_payload.items()
                if key not in answer_fields
            }
        )
        answer_values = {
                key: value
                for key, value in answer_payload.items()
                if key in answer_fields and key != "metadata"
            }
        answer_values.setdefault(
            "status",
            "completed"
            if runtime_status is RuntimeResultStatus.COMPLETED
            else "failed"
            if runtime_status is RuntimeResultStatus.FAILED
            else "interrupted",
        )
        answer_values.setdefault("stop_reason", str(runtime.get("stop_reason") or ""))
        answer = FinalAnswer(
            **answer_values,
            metadata=answer_metadata,
        )
        observations = [
            Observation.model_validate(item)
            for item in checkpoint.get("observations", [])
            if isinstance(item, dict)
        ]
        artifacts = [
            Artifact.model_validate(item)
            for item in checkpoint.get("artifacts", [])
            if isinstance(item, dict)
        ]
        pending_payload = checkpoint.get("pending_action")
        pending_action = (
            PendingAction.model_validate(pending_payload)
            if isinstance(pending_payload, dict)
            else None
        )
        return cls(
            question=str(payload.get("question") or ""),
            run_id=str(identity.get("run_id") or checkpoint.get("run_id") or ""),
            thread_id=str(
                identity.get("thread_id")
                or identity.get("conversation_thread_id")
                or checkpoint.get("graph_thread_id")
                or ""
            ),
            graph_thread_id=str(
                identity.get("graph_thread_id") or checkpoint.get("graph_thread_id") or ""
            ),
            turn_id=str(identity.get("turn_id") or checkpoint.get("turn_id") or ""),
            status=runtime_status,
            stop_reason=str(runtime.get("stop_reason") or answer.stop_reason or ""),
            final_answer=answer,
            observations=observations,
            pending_action=pending_action,
            artifacts=artifacts,
            events=[
                RuntimeStreamEvent.model_validate(item)
                for item in payload.get("events", [])
                if isinstance(item, dict)
            ],
            diagnostics=dict(runtime.get("diagnostics") or {}),
            context_snapshot=dict(payload.get("context_snapshot") or {}),
            skill_activation=dict(payload.get("skill_activation") or {}),
            answer_delta_streamed=bool(payload.get("answer_delta_streamed")),
            error=dict(payload["error"]) if isinstance(payload.get("error"), dict) else None,
            compatibility_payload=deepcopy(payload),
        )

    def to_compatibility_payload(self) -> dict[str, Any]:
        """Return the v1 mapping while hosts migrate to the typed result."""

        return deepcopy(self.compatibility_payload)
