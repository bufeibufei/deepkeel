from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from threading import Event as ThreadEvent
from typing import Any

from deepkeel.runtime_api import RuntimeRequest, RuntimeStreamEvent
from deepkeel.async_ports import run_sync_adapter
from deepkeel.failures import RunCanceledError
from deepkeel.runtime_streaming import BoundedRuntimeStreamBridge


async def stream_runtime_async(
    runtime: Any,
    request: RuntimeRequest,
    *,
    provider: Any = None,
    providers: dict[str, Any] | None = None,
    session: Any = None,
) -> AsyncGenerator[RuntimeStreamEvent, None]:
    """Bridge canonical runtime events into a bounded async generator."""

    prepared = runtime._ensure_request_identity(request)
    loop = asyncio.get_running_loop()
    closed = ThreadEvent()
    bridge = BoundedRuntimeStreamBridge(
        loop=loop,
        maxsize=runtime.async_stream_buffer_size,
        closed=closed,
    )

    def sink(event: dict[str, Any]) -> None:
        if closed.is_set():
            raise RunCanceledError()
        bridge.offer_event(event)

    async def execute() -> None:
        try:
            result = await runtime.arun(
                prepared,
                provider=provider,
                providers=providers,
                session=session,
                event_sink=sink,
            )
            if not closed.is_set():
                await bridge.put_terminal("result", result)
        except BaseException as exc:
            if not closed.is_set():
                await bridge.put_terminal("error", exc)

    task = asyncio.create_task(execute())
    completed = False
    try:
        while True:
            kind, value = await bridge.get()
            if kind == "event":
                yield RuntimeStreamEvent.model_validate(value)
                continue
            if kind == "error":
                raise value
            result = value
            yield RuntimeStreamEvent(
                run_id=result.run_id,
                thread_id=result.thread_id,
                turn_id=result.turn_id,
                event_type="runtime.result",
                title="Runtime result",
                summary=result.final_answer.summary,
                payload={"result": result.model_dump(mode="json")},
                ephemeral=True,
            )
            completed = True
            return
    finally:
        closed.set()
        await bridge.close()
        if not completed and not task.done():
            cancel = getattr(runtime.run_control, "cancel", None)
            if callable(cancel):
                await run_sync_adapter(
                    cancel,
                    prepared.runtime_scope.qualify_identity(prepared.run_id),
                )
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=runtime.async_cancel_timeout_seconds,
                )
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()
