from __future__ import annotations

import copy
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Protocol


TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "canceled"})
RUN_SETTLED_EVENT = "run.settled"


def normalize_runtime_status(status: str) -> str:
    normalized = str(status or "preparing").strip().lower()
    return "canceled" if normalized == "cancelled" else normalized


@dataclass(frozen=True, slots=True)
class RunStateSnapshot:
    """Canonical state projection rebuilt from the durable run journal."""

    run_id: str
    version: int = 0
    sequence: int = 0
    status: str = "preparing"
    settled: bool = False
    settlement_status: str = ""
    last_event_type: str = ""
    checkpoint_type: str = ""
    checkpoint_state: dict[str, Any] = field(default_factory=dict)
    resume_token: str = ""

    def __post_init__(self) -> None:
        normalized = normalize_runtime_status(self.status)
        object.__setattr__(self, "status", normalized)
        if not self.run_id:
            raise ValueError("run_id is required")
        if self.version < 0 or self.sequence < 0:
            raise ValueError("version and sequence must be non-negative")
        terminal = normalized in TERMINAL_RUN_STATUSES
        if self.settled != terminal:
            raise ValueError("settled must match terminal run status")
        if terminal and self.settlement_status != normalized:
            raise ValueError("settlement_status must match terminal run status")
        if not terminal and self.settlement_status:
            raise ValueError("active run cannot have settlement_status")

    @property
    def can_accept_input(self) -> bool:
        return self.settled or self.status == "waiting_user_input"

    @property
    def input_strategy(self) -> str:
        return "follow_up" if self.can_accept_input else "hard_interrupt"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "version": self.version,
            "sequence": self.sequence,
            "status": self.status,
            "settled": self.settled,
            "settlement_status": self.settlement_status,
            "last_event_type": self.last_event_type,
            "checkpoint_type": self.checkpoint_type,
            "checkpoint_state": copy.deepcopy(self.checkpoint_state),
            "resume_token": self.resume_token,
            "can_accept_input": self.can_accept_input,
            "input_strategy": self.input_strategy,
        }


@dataclass(slots=True)
class RunAggregate:
    """Deterministic reducer for ordered durable runtime mutations."""

    run_id: str
    version: int = 0
    sequence: int = 0
    status: str = "preparing"
    settled: bool = False
    settlement_status: str = ""
    last_event_type: str = ""
    checkpoint_type: str = ""
    checkpoint_state: dict[str, Any] = field(default_factory=dict)
    resume_token: str = ""

    def apply(self, mutation: RuntimeStateMutation, *, sequence: int) -> None:
        if mutation.run_id != self.run_id:
            raise ValueError("mutation.run_id must match aggregate run_id")
        if self.settled:
            raise RuntimeStateConflict("settled run cannot accept another mutation")
        if sequence != self.sequence + 1:
            raise RuntimeStateConflict(
                f"runtime event sequence must be contiguous: expected {self.sequence + 1}, "
                f"found {sequence}"
            )
        target_status = normalize_runtime_status(mutation.target_status)
        terminal = target_status in TERMINAL_RUN_STATUSES
        if terminal and mutation.event_type != RUN_SETTLED_EVENT:
            raise ValueError("terminal runtime mutation must use run.settled")
        if mutation.event_type == RUN_SETTLED_EVENT:
            payload_status = normalize_runtime_status(
                str(mutation.event_payload.get("status") or target_status)
            )
            if not terminal or payload_status != target_status:
                raise ValueError("run.settled requires one matching terminal status")
        self.sequence = sequence
        self.version += 1
        self.status = target_status
        self.settled = terminal
        self.settlement_status = target_status if terminal else ""
        self.last_event_type = mutation.event_type
        self.checkpoint_type = mutation.checkpoint_type
        self.checkpoint_state = copy.deepcopy(mutation.checkpoint_state)
        self.resume_token = mutation.resume_token

    def snapshot(self) -> RunStateSnapshot:
        return RunStateSnapshot(
            run_id=self.run_id,
            version=self.version,
            sequence=self.sequence,
            status=self.status,
            settled=self.settled,
            settlement_status=self.settlement_status,
            last_event_type=self.last_event_type,
            checkpoint_type=self.checkpoint_type,
            checkpoint_state=copy.deepcopy(self.checkpoint_state),
            resume_token=self.resume_token,
        )


class RuntimeStateConflict(RuntimeError):
    code = "RUNTIME_STATE_CONFLICT"


@dataclass(frozen=True, slots=True)
class RuntimeStateMutation:
    """One atomic durable mutation of status, event and portable checkpoint."""

    mutation_id: str
    run_id: str
    event_type: str
    target_status: str
    event_payload: dict[str, Any] = field(default_factory=dict)
    event_visibility: str = "internal"
    checkpoint_type: str = "runtime"
    checkpoint_state: dict[str, Any] = field(default_factory=dict)
    resume_token: str = ""
    final_message_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    cancel_requested: bool | None = None
    delete_checkpoint_types: tuple[str, ...] = ()
    expected_version: int | None = None
    expected_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class RuntimeStateReceipt:
    mutation_id: str
    run_id: str
    version: int
    sequence: int
    status: str
    replayed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "run_id": self.run_id,
            "version": self.version,
            "sequence": self.sequence,
            "status": self.status,
            "replayed": self.replayed,
        }


class RuntimeStateStore(Protocol):
    terminal_settlement_owner: str

    def commit(
        self,
        mutation: RuntimeStateMutation,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> RuntimeStateReceipt: ...

    def load_snapshot(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> RunStateSnapshot: ...


@dataclass(slots=True)
class _MemoryRuntimeState:
    version: int = 0
    sequence: int = 0
    status: str = "preparing"
    events: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    receipts: dict[str, RuntimeStateReceipt] = field(default_factory=dict)

    def aggregate(self, run_id: str) -> RunAggregate:
        latest = self.checkpoints[-1] if self.checkpoints else {}
        normalized = normalize_runtime_status(self.status)
        return RunAggregate(
            run_id=run_id,
            version=self.version,
            sequence=self.sequence,
            status=normalized,
            settled=normalized in TERMINAL_RUN_STATUSES,
            settlement_status=(normalized if normalized in TERMINAL_RUN_STATUSES else ""),
            last_event_type=str(self.events[-1].get("event_type") or "") if self.events else "",
            checkpoint_type=str(latest.get("checkpoint_type") or ""),
            checkpoint_state=copy.deepcopy(latest.get("state") or {}),
            resume_token=str(latest.get("resume_token") or ""),
        )


class InMemoryRuntimeStateStore:
    """Atomic reference adapter with optional crash-point injection for tests."""

    terminal_settlement_owner = "runtime"

    def __init__(self, failure_injector: Callable[[str], None] | None = None) -> None:
        self._states: dict[str, _MemoryRuntimeState] = {}
        self._lock = Lock()
        self._failure_injector = failure_injector

    def commit(
        self,
        mutation: RuntimeStateMutation,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> RuntimeStateReceipt:
        del session, user_id
        with self._lock:
            current = self._states.setdefault(mutation.run_id, _MemoryRuntimeState())
            replay = current.receipts.get(mutation.mutation_id)
            if replay is not None:
                return RuntimeStateReceipt(**{**replay.as_dict(), "replayed": True})
            aggregate = current.aggregate(mutation.run_id)
            if (
                mutation.expected_version is not None
                and mutation.expected_version != current.version
            ):
                raise RuntimeStateConflict(
                    f"runtime version changed: expected {mutation.expected_version}, "
                    f"found {current.version}"
                )
            if (
                mutation.expected_sequence is not None
                and mutation.expected_sequence != current.sequence
            ):
                raise RuntimeStateConflict(
                    f"runtime sequence changed: expected {mutation.expected_sequence}, "
                    f"found {current.sequence}"
                )
            before = copy.deepcopy(current)
            try:
                self._fail_at("before_event")
                next_sequence = current.sequence + 1
                aggregate.apply(mutation, sequence=next_sequence)
                current.sequence = next_sequence
                current.events.append(
                    {
                        "sequence": current.sequence,
                        "event_type": mutation.event_type,
                        "payload": copy.deepcopy(mutation.event_payload),
                        "visibility": mutation.event_visibility,
                    }
                )
                self._fail_at("after_event")
                current.checkpoints.append(
                    {
                        "sequence": current.sequence,
                        "checkpoint_type": mutation.checkpoint_type,
                        "state": copy.deepcopy(mutation.checkpoint_state),
                        "resume_token": mutation.resume_token,
                    }
                )
                self._fail_at("after_checkpoint")
                if mutation.delete_checkpoint_types:
                    deleted_types = set(mutation.delete_checkpoint_types)
                    current.checkpoints = [
                        item
                        for item in current.checkpoints
                        if item["checkpoint_type"] not in deleted_types
                    ]
                self._fail_at("after_checkpoint_cleanup")
                current.version = aggregate.version
                current.status = aggregate.status
                receipt = RuntimeStateReceipt(
                    mutation_id=mutation.mutation_id,
                    run_id=mutation.run_id,
                    version=current.version,
                    sequence=current.sequence,
                    status=current.status,
                )
                current.receipts[mutation.mutation_id] = receipt
                self._fail_at("before_commit")
                return receipt
            except Exception:
                self._states[mutation.run_id] = before
                raise

    def snapshot(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            state = copy.deepcopy(self._states.get(run_id) or _MemoryRuntimeState())
        return {
            "version": state.version,
            "sequence": state.sequence,
            "status": state.status,
            "events": state.events,
            "checkpoints": state.checkpoints,
        }

    def load_snapshot(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> RunStateSnapshot:
        del session, user_id
        with self._lock:
            state = copy.deepcopy(self._states.get(run_id) or _MemoryRuntimeState())
        return state.aggregate(run_id).snapshot()

    def _fail_at(self, stage: str) -> None:
        if self._failure_injector is not None:
            self._failure_injector(stage)
