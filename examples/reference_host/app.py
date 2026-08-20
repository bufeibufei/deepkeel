from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, AsyncIterator

from deepkeel.adapter_sdk import InMemoryRuntimeEventJournal, RuntimePorts
from deepkeel.runtime_sdk import (
    AgentHarness,
    InMemoryRunControl,
    InMemoryRuntimeStateStore,
    RunOperations,
    RuntimeRequest,
)


@dataclass(slots=True)
class ReferenceHost:
    app: Any
    harness: AgentHarness
    operations: RunOperations


def create_reference_host(provider: Any) -> ReferenceHost:
    """Build a runnable FastAPI/SSE Host without adding FastAPI to Core dependencies."""

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
    except ImportError as exc:  # pragma: no cover - optional example dependency
        raise RuntimeError("install fastapi and uvicorn to run the reference Host") from exc

    state_store = InMemoryRuntimeStateStore()
    event_journal = InMemoryRuntimeEventJournal()
    run_control = InMemoryRunControl()
    ports = RuntimePorts(
        runtime_state_store=state_store,
        event_journal=event_journal,
        run_control=run_control,
    )
    harness = AgentHarness.create(provider=provider, ports=ports, profile="testing")
    operations = RunOperations(state_store, run_control=run_control)
    app = FastAPI(title="DeepKeel reference Host")

    @app.post("/runs/stream")
    async def stream_run(payload: dict[str, Any]) -> StreamingResponse:
        request = RuntimeRequest.model_validate(payload)

        async def events() -> AsyncIterator[str]:
            async for event in harness.astream(request):
                data = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
                yield f"id: {event.cursor}\nevent: {event.event_type}\ndata: {data}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/runs/{run_id}")
    async def inspect_run(run_id: str, user_id: str = "local-device") -> dict[str, Any]:
        inspection = operations.inspect(run_id, user_id=user_id)
        if not inspection.found:
            raise HTTPException(status_code=404, detail="run not found")
        return inspection.model_dump(mode="json")

    @app.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, user_id: str = "local-device") -> dict[str, Any]:
        receipt = operations.request_cancel(run_id, user_id=user_id)
        return receipt.model_dump(mode="json")

    return ReferenceHost(app=app, harness=harness, operations=operations)
