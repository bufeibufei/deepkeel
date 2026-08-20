from __future__ import annotations

from typing import Any

from deepkeel.runtime_api import RuntimeEventEnvelope
from deepkeel.telemetry import TelemetryRecord


def project_telemetry_record(envelope: RuntimeEventEnvelope) -> TelemetryRecord:
    """Project observability data from the canonical event envelope."""

    return TelemetryRecord.from_runtime_event(
        envelope.model_dump(mode="json"),
        run_id=envelope.run_id,
        thread_id=envelope.thread_id,
        turn_id=envelope.turn_id,
    )


def project_trace_item(envelope: RuntimeEventEnvelope) -> dict[str, Any]:
    """Return the compact trace projection used by diagnostics and support tooling."""

    return {
        "event_id": envelope.event_id,
        "sequence": envelope.sequence,
        "event_type": envelope.event_type,
        "source_event_type": envelope.source_event_type,
        "title": envelope.title,
        "summary": envelope.summary,
        "created_at": (
            envelope.created_at.isoformat() if envelope.created_at is not None else None
        ),
    }


__all__ = ["project_telemetry_record", "project_trace_item"]
