from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any, Literal, Protocol
from uuid import uuid4

from harness_core.contracts import ToolCall, ToolResult
from harness_core.control import NoopRunControl, RunControl
from harness_core.leases import ExecutionFence
from harness_core.ports import RuntimeSession, SessionFactory


@dataclass(slots=True)
class ToolExecutionContext:
    run_id: str
    user_id: str
    thread_id: str = ""
    turn_id: str = ""
    session: RuntimeSession | None = None
    session_factory: SessionFactory | None = None
    context_bundle: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    budget_limits: dict[str, float] = field(default_factory=dict)
    deadline_monotonic: float | None = None
    run_control: RunControl = field(default_factory=NoopRunControl)
    execution_fence: ExecutionFence | None = None

    @property
    def fence_token(self) -> str:
        return self.execution_fence.token if self.execution_fence is not None else ""

    @property
    def fence_generation(self) -> int:
        return self.execution_fence.generation if self.execution_fence is not None else 0

    def raise_if_fence_lost(self) -> None:
        if self.execution_fence is not None:
            self.execution_fence.raise_if_lost()

    def fork(self, *, session: RuntimeSession | None = None) -> "ToolExecutionContext":
        return ToolExecutionContext(
            run_id=self.run_id,
            user_id=self.user_id,
            thread_id=self.thread_id,
            turn_id=self.turn_id,
            session=session,
            session_factory=self.session_factory,
            context_bundle=_copy_mapping(self.context_bundle),
            metadata=_copy_mapping(self.metadata),
            budget_limits=dict(self.budget_limits),
            deadline_monotonic=self.deadline_monotonic,
            run_control=self.run_control,
            execution_fence=self.execution_fence,
        )


@dataclass(frozen=True, slots=True)
class ToolExecutionClaim:
    status: Literal[
        "claimed",
        "replay",
        "busy",
        "blocked",
        "corrupt",
        "uncertain",
        "exhausted",
    ]
    run_id: str
    idempotency_key: str
    record_id: str = ""
    claim_owner: str = ""
    attempt_count: int = 0
    lease_expires_at: datetime | None = None
    retry_after_seconds: float = 0.0
    terminal_status: str = ""
    detail: str = ""


class ToolExecutionStore(Protocol):
    """Durable claim/replay/settle port for idempotent tool execution."""

    def claim(
        self,
        *,
        run_id: str,
        call: ToolCall,
        lease_seconds: float,
        max_attempts: int,
        reexecution_safe: bool = True,
    ) -> ToolExecutionClaim: ...

    def replay(self, claim: ToolExecutionClaim) -> ToolResult: ...

    def settle(self, claim: ToolExecutionClaim, result: ToolResult) -> None: ...


@dataclass(slots=True)
class _MemoryToolExecution:
    record_id: str
    status: str
    claim_owner: str
    lease_expires_at: datetime | None
    attempt_count: int
    result: ToolResult | None = None


class InMemoryToolExecutionStore:
    """Process-local fallback for Harness tests without persistence."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], _MemoryToolExecution] = {}
        self._lock = Lock()

    def claim(
        self,
        *,
        run_id: str,
        call: ToolCall,
        lease_seconds: float,
        max_attempts: int,
        reexecution_safe: bool = True,
    ) -> ToolExecutionClaim:
        now = datetime.now(UTC)
        key = (run_id, call.idempotency_key)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.status == "uncertain":
                return _memory_claim("uncertain", run_id, call.idempotency_key, entry)
            if entry is not None and entry.status == "exhausted":
                return _memory_claim("exhausted", run_id, call.idempotency_key, entry)
            if entry is not None and entry.result is not None:
                retry_failed = (
                    entry.result.status == "failed"
                    and entry.result.retryable
                    and entry.attempt_count < max(1, int(max_attempts))
                )
                if not retry_failed:
                    return _memory_claim("replay", run_id, call.idempotency_key, entry)
            if (
                entry is not None
                and entry.result is None
                and entry.claim_owner
                and entry.lease_expires_at is not None
                and entry.lease_expires_at > now
            ):
                return _memory_claim("busy", run_id, call.idempotency_key, entry)
            if entry is not None and entry.result is None:
                if entry.attempt_count >= max(1, int(max_attempts)):
                    entry.status = "exhausted"
                    entry.claim_owner = ""
                    entry.lease_expires_at = None
                    return _memory_claim("exhausted", run_id, call.idempotency_key, entry)
                if not reexecution_safe:
                    entry.status = "uncertain"
                    entry.claim_owner = ""
                    entry.lease_expires_at = None
                    return _memory_claim("uncertain", run_id, call.idempotency_key, entry)
            owner = uuid4().hex
            lease_expires_at = now + timedelta(seconds=max(0.001, float(lease_seconds)))
            if entry is None:
                entry = _MemoryToolExecution(
                    record_id=uuid4().hex,
                    status="running",
                    claim_owner=owner,
                    lease_expires_at=lease_expires_at,
                    attempt_count=1,
                )
                self._entries[key] = entry
            else:
                entry.status = "running"
                entry.claim_owner = owner
                entry.lease_expires_at = lease_expires_at
                entry.attempt_count += 1
                entry.result = None
            return _memory_claim("claimed", run_id, call.idempotency_key, entry)

    def replay(self, claim: ToolExecutionClaim) -> ToolResult:
        with self._lock:
            entry = self._entries.get((claim.run_id, claim.idempotency_key))
            if entry is None or entry.result is None:
                raise LookupError("tool execution result is not available for replay")
            return entry.result.model_copy(deep=True)

    def settle(self, claim: ToolExecutionClaim, result: ToolResult) -> None:
        with self._lock:
            entry = self._entries.get((claim.run_id, claim.idempotency_key))
            if entry is None or entry.claim_owner != claim.claim_owner:
                raise RuntimeError("tool execution claim is no longer owned")
            entry.status = result.status
            entry.claim_owner = ""
            entry.lease_expires_at = None
            entry.result = result.model_copy(deep=True)


def _memory_claim(
    status: Literal["claimed", "replay", "busy", "uncertain", "exhausted"],
    run_id: str,
    idempotency_key: str,
    entry: _MemoryToolExecution,
) -> ToolExecutionClaim:
    retry_after = 0.0
    if status == "busy" and entry.lease_expires_at is not None:
        retry_after = max(
            0.0,
            (entry.lease_expires_at - datetime.now(UTC)).total_seconds(),
        )
    return ToolExecutionClaim(
        status=status,
        run_id=run_id,
        idempotency_key=idempotency_key,
        record_id=entry.record_id,
        claim_owner=entry.claim_owner,
        attempt_count=entry.attempt_count,
        lease_expires_at=entry.lease_expires_at,
        retry_after_seconds=retry_after,
    )


def _copy_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _copy_mapping(item) if isinstance(item, dict) else item
        for key, item in value.items()
    }
