from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Event as ThreadEvent
from typing import Any


class RuntimeStreamBackpressureError(RuntimeError):
    """Raised when a same-loop producer outruns every bounded stream buffer."""

    code = "RUNTIME_STREAM_BACKPRESSURE"


class BoundedRuntimeStreamBridge:
    """Bridge synchronous event callbacks into a bounded async stream.

    Producers running outside the Host loop block on the bounded queue. A
    same-loop producer cannot block synchronously, so it uses one bounded
    backlog and one drain task. Consecutive ephemeral answer deltas are merged
    without losing text.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        maxsize: int,
        closed: ThreadEvent,
    ) -> None:
        self.loop = loop
        self.closed = closed
        self.queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(
            maxsize=max(1, int(maxsize))
        )
        self._backlog: deque[tuple[str, Any]] = deque()
        self._backlog_limit = max(8, max(1, int(maxsize)))
        self._drain_task: asyncio.Task[None] | None = None

    @property
    def buffered_items(self) -> int:
        return self.queue.qsize() + len(self._backlog)

    def offer_event(self, event: dict[str, Any]) -> None:
        if self.closed.is_set():
            return
        try:
            producer_loop = asyncio.get_running_loop()
        except RuntimeError:
            producer_loop = None
        if producer_loop is self.loop:
            self._offer_same_loop(("event", event))
            return
        future = asyncio.run_coroutine_threadsafe(
            self.queue.put(("event", event)),
            self.loop,
        )
        while True:
            try:
                future.result(timeout=0.1)
                return
            except FutureTimeoutError:
                if self.closed.is_set():
                    future.cancel()
                    return

    async def put_terminal(self, kind: str, value: Any) -> None:
        await self._wait_backlog_drained()
        if not self.closed.is_set():
            await self.queue.put((kind, value))

    async def get(self) -> tuple[str, Any]:
        return await self.queue.get()

    async def close(self) -> None:
        task = self._drain_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._backlog.clear()

    def _offer_same_loop(self, item: tuple[str, Any]) -> None:
        try:
            self.queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            pass
        if self._merge_answer_delta(item):
            return
        if len(self._backlog) >= self._backlog_limit:
            raise RuntimeStreamBackpressureError(
                "same-loop stream producer exceeded the bounded backlog"
            )
        self._backlog.append(item)
        self._ensure_drain_task()

    def _merge_answer_delta(self, item: tuple[str, Any]) -> bool:
        if not self._backlog or item[0] != "event":
            return False
        previous_kind, previous_value = self._backlog[-1]
        current = item[1]
        if (
            previous_kind != "event"
            or not isinstance(previous_value, dict)
            or not isinstance(current, dict)
            or str(previous_value.get("event_type") or "") != "answer.delta"
            or str(current.get("event_type") or "") != "answer.delta"
            or not bool(previous_value.get("ephemeral"))
            or not bool(current.get("ephemeral"))
        ):
            return False
        previous_payload = dict(previous_value.get("payload") or {})
        current_payload = dict(current.get("payload") or {})
        previous_payload["delta"] = (
            f"{previous_payload.get('delta') or ''}{current_payload.get('delta') or ''}"
        )
        previous_payload["merged_count"] = int(previous_payload.get("merged_count") or 1) + 1
        merged = dict(previous_value)
        merged["payload"] = previous_payload
        self._backlog[-1] = ("event", merged)
        return True

    def _ensure_drain_task(self) -> None:
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = self.loop.create_task(self._drain())

    async def _drain(self) -> None:
        while self._backlog and not self.closed.is_set():
            item = self._backlog[0]
            await self.queue.put(item)
            self._backlog.popleft()

    async def _wait_backlog_drained(self) -> None:
        while self._backlog:
            self._ensure_drain_task()
            task = self._drain_task
            if task is not None:
                await task
