from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from deepkeel.contracts import ToolCall
from deepkeel.model_capabilities import ModelCapabilities, ResponseContract
from deepkeel.scope import RuntimeScope


class ModelTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = ""
    model_id: str = ""
    model_role: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelProviderInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str
    model_id: str = ""
    model_role: str = "reasoning"
    supports_streaming: bool = True
    supports_native_tools: bool = True
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] = "auto"
    request_timeout: int = 300
    max_output_tokens: int | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    response_contract: ResponseContract | None = None


class ModelInvocationEnvelope(BaseModel):
    """Exact, access-controlled replay input for one governed model attempt."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "model-invocation-envelope-v1"
    invocation_id: str
    run_id: str
    tenant_id: str = ""
    user_id: str = "local-device"
    namespace: str = "default"
    thread_id: str
    turn_id: str
    step_index: int = 0
    attempt_index: int = 1
    retry_kind: str = "primary"
    provider_id: str = ""
    model_id: str = ""
    model_role: str = ""
    router_id: str = ""
    route_reason: str = ""
    estimated_input_tokens: int = 0
    request: ModelInvocation
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def runtime_scope(self) -> RuntimeScope:
        return RuntimeScope(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            namespace=self.namespace,
        )

    @property
    def request_fingerprint(self) -> str:
        encoded = json.dumps(
            self.request.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "invocation_id": self.invocation_id,
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "namespace": self.namespace,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "step_index": self.step_index,
            "attempt_index": self.attempt_index,
            "retry_kind": self.retry_kind,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_role": self.model_role,
            "router_id": self.router_id,
            "route_reason": self.route_reason,
            "estimated_input_tokens": self.estimated_input_tokens,
            "message_count": len(self.request.messages),
            "tool_count": len(self.request.tools),
            "tool_names": [
                str((tool.get("function") or {}).get("name") or tool.get("name") or "")
                for tool in self.request.tools
                if isinstance(tool, dict)
            ],
            "request_fingerprint": self.request_fingerprint,
            "created_at": self.created_at.isoformat(),
        }


class ModelInvocationRecorder(Protocol):
    def record(self, envelope: ModelInvocationEnvelope) -> None: ...

    def get(self, invocation_id: str) -> ModelInvocationEnvelope | None: ...


class ModelInvocationConflict(RuntimeError):
    """Raised when an invocation identity is reused with different input."""


class ModelInvocationUnavailable(RuntimeError):
    """Raised when a prior invocation cannot be safely repeated or replayed."""


class ModelInvocationClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "model-invocation-claim-v1"
    invocation_id: str
    outcome: str
    claim_token: str = ""
    result: ModelTurn | None = None
    failure_type: str = ""
    failure_message: str = ""


class ModelInvocationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "model-invocation-record-v1"
    envelope: ModelInvocationEnvelope
    status: str = "running"
    claim_token: str = ""
    claim_expires_at: datetime | None = None
    result: ModelTurn | None = None
    failure_type: str = ""
    failure_message: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelInvocationStore(Protocol):
    """Atomic ownership and settlement boundary for provider invocations."""

    def claim(
        self,
        envelope: ModelInvocationEnvelope,
        *,
        lease_seconds: float = 300.0,
    ) -> ModelInvocationClaim: ...

    def complete(
        self,
        invocation_id: str,
        *,
        claim_token: str,
        result: ModelTurn,
    ) -> ModelInvocationRecord: ...

    def fail(
        self,
        invocation_id: str,
        *,
        claim_token: str,
        failure_type: str,
        failure_message: str,
    ) -> ModelInvocationRecord: ...

    def get_record(self, invocation_id: str) -> ModelInvocationRecord | None: ...


class InMemoryModelInvocationStore:
    """Thread-safe reference store with fail-closed ambiguous recovery."""

    def __init__(self) -> None:
        self._records: dict[str, ModelInvocationRecord] = {}
        self._lock = Lock()

    def claim(
        self,
        envelope: ModelInvocationEnvelope,
        *,
        lease_seconds: float = 300.0,
    ) -> ModelInvocationClaim:
        now = datetime.now(UTC)
        with self._lock:
            record = self._records.get(envelope.invocation_id)
            if record is not None:
                if record.envelope.request_fingerprint != envelope.request_fingerprint:
                    raise ModelInvocationConflict(
                        "invocation_id cannot be reused with a different request"
                    )
                if record.status == "completed" and record.result is not None:
                    return ModelInvocationClaim(
                        invocation_id=envelope.invocation_id,
                        outcome="replay",
                        result=record.result.model_copy(deep=True),
                    )
                if record.status == "failed":
                    return ModelInvocationClaim(
                        invocation_id=envelope.invocation_id,
                        outcome="failed",
                        failure_type=record.failure_type,
                        failure_message=record.failure_message,
                    )
                if record.claim_expires_at is None or record.claim_expires_at > now:
                    return ModelInvocationClaim(
                        invocation_id=envelope.invocation_id,
                        outcome="in_progress",
                    )
                return ModelInvocationClaim(
                    invocation_id=envelope.invocation_id,
                    outcome="uncertain",
                    failure_type="claim_expired",
                    failure_message=(
                        "the previous provider invocation expired without a durable result"
                    ),
                )

            claim_token = uuid4().hex
            self._records[envelope.invocation_id] = ModelInvocationRecord(
                envelope=envelope.model_copy(deep=True),
                status="running",
                claim_token=claim_token,
                claim_expires_at=now + timedelta(seconds=max(1.0, float(lease_seconds))),
                updated_at=now,
            )
            return ModelInvocationClaim(
                invocation_id=envelope.invocation_id,
                outcome="acquired",
                claim_token=claim_token,
            )

    def complete(
        self,
        invocation_id: str,
        *,
        claim_token: str,
        result: ModelTurn,
    ) -> ModelInvocationRecord:
        return self._settle(
            invocation_id,
            claim_token=claim_token,
            status="completed",
            result=result,
        )

    def fail(
        self,
        invocation_id: str,
        *,
        claim_token: str,
        failure_type: str,
        failure_message: str,
    ) -> ModelInvocationRecord:
        return self._settle(
            invocation_id,
            claim_token=claim_token,
            status="failed",
            failure_type=failure_type,
            failure_message=failure_message,
        )

    def get_record(self, invocation_id: str) -> ModelInvocationRecord | None:
        with self._lock:
            record = self._records.get(str(invocation_id or ""))
            return record.model_copy(deep=True) if record is not None else None

    def _settle(
        self,
        invocation_id: str,
        *,
        claim_token: str,
        status: str,
        result: ModelTurn | None = None,
        failure_type: str = "",
        failure_message: str = "",
    ) -> ModelInvocationRecord:
        with self._lock:
            record = self._records.get(str(invocation_id or ""))
            if record is None:
                raise ModelInvocationConflict("cannot settle an unknown invocation")
            if record.status in {"completed", "failed"}:
                same_result = status == record.status and (
                    status == "failed" or (record.result is not None and record.result == result)
                )
                if same_result:
                    return record.model_copy(deep=True)
                raise ModelInvocationConflict("invocation is already settled")
            if not claim_token or claim_token != record.claim_token:
                raise ModelInvocationConflict("model invocation claim token changed")
            settled = record.model_copy(
                update={
                    "status": status,
                    "claim_token": "",
                    "claim_expires_at": None,
                    "result": result.model_copy(deep=True) if result is not None else None,
                    "failure_type": str(failure_type or ""),
                    "failure_message": str(failure_message or "")[:500],
                    "updated_at": datetime.now(UTC),
                },
                deep=True,
            )
            self._records[invocation_id] = settled
            return settled.model_copy(deep=True)


class InMemoryModelInvocationRecorder:
    """Reference recorder that keeps exact prompts outside ordinary event payloads."""

    def __init__(self) -> None:
        self._records: dict[str, ModelInvocationEnvelope] = {}
        self._lock = Lock()

    def record(self, envelope: ModelInvocationEnvelope) -> None:
        with self._lock:
            existing = self._records.get(envelope.invocation_id)
            if (
                existing is not None
                and existing.request_fingerprint != envelope.request_fingerprint
            ):
                raise ValueError("invocation_id cannot be reused with a different request")
            self._records[envelope.invocation_id] = envelope.model_copy(deep=True)

    def get(self, invocation_id: str) -> ModelInvocationEnvelope | None:
        with self._lock:
            envelope = self._records.get(str(invocation_id or ""))
            return envelope.model_copy(deep=True) if envelope is not None else None
