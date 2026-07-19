from __future__ import annotations

import copy
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Protocol


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
    def commit(
        self,
        mutation: RuntimeStateMutation,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> RuntimeStateReceipt: ...


@dataclass(slots=True)
class _MemoryRuntimeState:
    version: int = 0
    sequence: int = 0
    status: str = "preparing"
    events: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    receipts: dict[str, RuntimeStateReceipt] = field(default_factory=dict)


class InMemoryRuntimeStateStore:
    """Atomic reference adapter with optional crash-point injection for tests."""

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
                current.sequence += 1
                current.events.append(
                    {
                        "sequence": current.sequence,
                        "event_type": mutation.event_type,
                        "payload": copy.deepcopy(mutation.event_payload),
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
                current.version += 1
                current.status = mutation.target_status
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

    def _fail_at(self, stage: str) -> None:
        if self._failure_injector is not None:
            self._failure_injector(stage)
