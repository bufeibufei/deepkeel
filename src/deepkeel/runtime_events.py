from __future__ import annotations

import asyncio
from collections import deque
from threading import Lock
from typing import Any, Callable

from deepkeel.events import AgentEventPersistenceError, envelope_runtime_event
from deepkeel.runtime_api import RuntimeStreamEvent
from deepkeel.scope import scoped_adapter_operation
from deepkeel.scope import RuntimeScope
from deepkeel.telemetry import TelemetryPort, TelemetryRecord
from deepkeel.type_narrowing import as_dict


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
        async_event_journal: Any = None,
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
        self.async_event_journal = async_event_journal
        self.event_sink = event_sink
        self.execution_fence = execution_fence
        self.events: list[dict[str, Any]] = []
        self.answer_delta_streamed = False
        self.telemetry_error_count = 0
        self.telemetry_last_error = ""
        self._sequence = initial_sequence
        self._lock = Lock()
        self._journal_loop = asyncio.get_running_loop() if async_event_journal is not None else None
        self._async_deliveries: deque[tuple[RuntimeStreamEvent | None, dict[str, Any]]] = deque()
        self._async_delivery_limit = 1_024
        self._async_delivery_task: asyncio.Task[None] | None = None
        self._async_delivery_error: Exception | None = None
        self._last_async_control_check = 0.0
        self._async_control_check_interval = 0.25

    @property
    def sequence(self) -> int:
        return self._sequence

    def record_telemetry_error(self, exc: Exception) -> None:
        self.telemetry_error_count += 1
        self.telemetry_last_error = f"{type(exc).__name__}: {exc}"

    def __call__(self, event: dict[str, Any]) -> None:
        self._raise_async_delivery_error()
        if self.execution_fence is not None:
            self.execution_fence.raise_if_lost()
        if self.async_event_journal is None:
            self.run_control.raise_if_cancelled(self.scope.qualify_identity(self.run_id))
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
            if not envelope.ephemeral and self.async_event_journal is None:
                self._persist_sync_event(envelope)
            projected = envelope.model_dump(mode="json")
            if projected.get("event_type") == "answer.delta":
                self.answer_delta_streamed = True
            if not projected.get("ephemeral"):
                self.events.append(projected)
        if self.async_event_journal is None:
            self._record_telemetry(projected)
        if self.async_event_journal is not None:
            self._schedule_async_delivery(
                envelope if not envelope.ephemeral else None,
                projected,
            )
        elif self.event_sink is not None:
            self.event_sink(projected)

    async def flush(self) -> None:
        """Wait until all native async journal writes are durable."""

        while True:
            task = self._async_delivery_task
            if task is None:
                break
            await asyncio.gather(task, return_exceptions=True)
            if not self._async_deliveries:
                break
            self._ensure_async_delivery_task()
        if self.async_event_journal is not None:
            await self._acheck_cancelled(force=True)
        self._raise_async_delivery_error()

    def _record_telemetry(self, projected: dict[str, Any]) -> None:
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

    async def _arecord_telemetry(self, projected: dict[str, Any]) -> None:
        try:
            await asyncio.to_thread(
                self.telemetry.record,
                TelemetryRecord.from_runtime_event(
                    projected,
                    run_id=self.run_id,
                    thread_id=self.thread_id,
                    turn_id=self.turn_id,
                ),
            )
        except Exception as exc:
            self.record_telemetry_error(exc)

    async def _acheck_cancelled(self, *, force: bool = False) -> None:
        now = asyncio.get_running_loop().time()
        if not force and now - self._last_async_control_check < self._async_control_check_interval:
            return
        self._last_async_control_check = now
        await asyncio.to_thread(
            self.run_control.raise_if_cancelled,
            self.scope.qualify_identity(self.run_id),
            force=force,
        )

    def _persist_sync_event(self, envelope: RuntimeStreamEvent) -> None:
        if self.event_journal is not None:
            try:
                append = scoped_adapter_operation(
                    self.event_journal,
                    "append",
                    self.scope,
                )
                if getattr(append, "__name__", "") == "append_scoped":
                    append(envelope, scope=self.scope)
                else:
                    append(envelope)
            except Exception as exc:
                raise AgentEventPersistenceError(
                    f"runtime event journal append failed: {exc}"
                ) from exc

    def _schedule_async_delivery(
        self,
        envelope: RuntimeStreamEvent | None,
        projected: dict[str, Any],
    ) -> None:
        loop = self._journal_loop
        if loop is None:
            raise AgentEventPersistenceError(
                "native async event journal requires an active event loop"
            )
        try:
            producer_loop = asyncio.get_running_loop()
        except RuntimeError:
            producer_loop = None
        if producer_loop is loop:
            self._enqueue_async_delivery(envelope, projected)
            return
        future = asyncio.run_coroutine_threadsafe(
            self._aenqueue_async_delivery(envelope, projected),
            loop,
        )
        try:
            future.result()
        except Exception as exc:
            if isinstance(exc, AgentEventPersistenceError):
                raise
            raise AgentEventPersistenceError(
                f"runtime event delivery enqueue failed: {exc}"
            ) from exc

    async def _aenqueue_async_delivery(
        self,
        envelope: RuntimeStreamEvent | None,
        projected: dict[str, Any],
    ) -> None:
        self._enqueue_async_delivery(envelope, projected)

    def _enqueue_async_delivery(
        self,
        envelope: RuntimeStreamEvent | None,
        projected: dict[str, Any],
    ) -> None:
        self._raise_async_delivery_error()
        if self._merge_answer_delta(envelope, projected):
            return
        if len(self._async_deliveries) >= self._async_delivery_limit:
            raise AgentEventPersistenceError(
                "native async event delivery exceeded its bounded backlog"
            )
        self._async_deliveries.append((envelope, projected))
        self._ensure_async_delivery_task()

    def _merge_answer_delta(
        self,
        envelope: RuntimeStreamEvent | None,
        projected: dict[str, Any],
    ) -> bool:
        if envelope is not None or not self._async_deliveries:
            return False
        previous_envelope, previous = self._async_deliveries[-1]
        if (
            previous_envelope is not None
            or str(previous.get("event_type") or "") != "answer.delta"
            or str(projected.get("event_type") or "") != "answer.delta"
            or not bool(previous.get("ephemeral"))
            or not bool(projected.get("ephemeral"))
        ):
            return False
        previous_payload = dict(previous.get("payload") or {})
        current_payload = dict(projected.get("payload") or {})
        previous_payload["delta"] = (
            f"{previous_payload.get('delta') or ''}{current_payload.get('delta') or ''}"
        )
        previous_payload["merged_count"] = int(previous_payload.get("merged_count") or 1) + 1
        merged = dict(previous)
        merged["payload"] = previous_payload
        self._async_deliveries[-1] = (None, merged)
        return True

    def _ensure_async_delivery_task(self) -> None:
        task = self._async_delivery_task
        if task is not None and not task.done():
            return
        loop = self._journal_loop
        if loop is None:
            return
        task = loop.create_task(self._drain_async_deliveries())
        self._async_delivery_task = task
        task.add_done_callback(self._async_delivery_done)

    async def _drain_async_deliveries(self) -> None:
        while self._async_deliveries:
            envelope, projected = self._async_deliveries.popleft()
            try:
                await self._acheck_cancelled()
            except Exception as exc:
                self._async_delivery_error = exc
                self._async_deliveries.clear()
                return
            if envelope is not None:
                try:
                    append = scoped_adapter_operation(
                        self.async_event_journal,
                        "append",
                        self.scope,
                    )
                    if getattr(append, "__name__", "") == "append_scoped":
                        await append(envelope, scope=self.scope)
                    else:
                        await append(envelope)
                except Exception as exc:
                    self._async_delivery_error = AgentEventPersistenceError(
                        f"runtime event journal append failed: {exc}"
                    )
                    self._async_deliveries.clear()
                    return
            await self._arecord_telemetry(projected)
            if self.event_sink is not None:
                try:
                    self.event_sink(projected)
                except Exception as exc:
                    self._async_delivery_error = exc
                    self._async_deliveries.clear()
                    return

    def _async_delivery_done(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._async_delivery_error = exc

    def _raise_async_delivery_error(self) -> None:
        if self._async_delivery_error is None:
            return
        error = self._async_delivery_error
        self._async_delivery_error = None
        raise error
