from __future__ import annotations

from threading import Lock
from typing import Any, Callable

from harness_core.events import AgentEventPersistenceError, envelope_runtime_event
from harness_core.runtime_api import RuntimeStreamEvent
from harness_core.scope import RuntimeScope
from harness_core.telemetry import TelemetryPort, TelemetryRecord
from harness_core.type_narrowing import as_dict


EventSink = Callable[[dict[str, Any]], None]


class RuntimeEventEmitter:
    """Own event sequencing, durability, telemetry, and downstream delivery."""

    def __init__(
        self,
        *,
        run_id: str,
        thread_id: str,
        turn_id: str,
        run_version: int,
        initial_sequence: int,
        skill_id: str,
        scope: RuntimeScope,
        run_control: Any,
        telemetry: TelemetryPort,
        event_journal: Any = None,
        event_sink: EventSink | None = None,
        execution_fence: Any = None,
    ) -> None:
        self.run_id = run_id
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.run_version = run_version
        self.skill_id = skill_id
        self.scope = scope
        self.run_control = run_control
        self.telemetry = telemetry
        self.event_journal = event_journal
        self.event_sink = event_sink
        self.execution_fence = execution_fence
        self.events: list[dict[str, Any]] = []
        self.answer_delta_streamed = False
        self.telemetry_error_count = 0
        self.telemetry_last_error = ""
        self._sequence = initial_sequence
        self._lock = Lock()

    @property
    def sequence(self) -> int:
        return self._sequence

    def record_telemetry_error(self, exc: Exception) -> None:
        self.telemetry_error_count += 1
        self.telemetry_last_error = f"{type(exc).__name__}: {exc}"

    def __call__(self, event: dict[str, Any]) -> None:
        if self.execution_fence is not None:
            self.execution_fence.raise_if_lost()
        self.run_control.raise_if_cancelled(self.run_id)
        with self._lock:
            self._sequence += 1
            projected = envelope_runtime_event(
                event,
                run_id=self.run_id,
                thread_id=self.thread_id,
                turn_id=self.turn_id,
                sequence=self._sequence,
                run_version=self.run_version,
                scope=self.scope,
            )
            payload = as_dict(projected.get("payload"))
            payload.setdefault("skill_id", self.skill_id)
            projected["payload"] = payload
            envelope = RuntimeStreamEvent.model_validate(projected)
            if not envelope.ephemeral and self.event_journal is not None:
                try:
                    self.event_journal.append(envelope)
                except Exception as exc:
                    raise AgentEventPersistenceError(
                        f"runtime event journal append failed: {exc}"
                    ) from exc
            projected = envelope.model_dump(mode="json")
            if projected.get("event_type") == "answer.delta":
                self.answer_delta_streamed = True
            if not projected.get("ephemeral"):
                self.events.append(projected)
        try:
            self.telemetry.record(
                TelemetryRecord.from_runtime_event(
                    projected,
                    run_id=self.run_id,
                    thread_id=self.thread_id,
                    turn_id=self.turn_id,
                )
            )
        except Exception as exc:
            self.record_telemetry_error(exc)
        if self.event_sink is not None:
            self.event_sink(projected)
