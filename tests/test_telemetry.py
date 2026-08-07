import logging

from datetime import UTC, datetime, timedelta

from deepkeel.telemetry import (
    CompositeTelemetry,
    InMemoryTelemetry,
    LoggingTelemetry,
    TelemetryRecord,
    TraceQuery,
)


def test_runtime_event_telemetry_preserves_correlations_and_excludes_content():
    event = {
        "schema_version": "harness-runtime-event-v1",
        "event_id": "event-1",
        "sequence": 7,
        "run_version": 3,
        "event_type": "model.completed",
        "source_event_type": "model.completed",
        "created_at": "2026-07-20T12:00:00+00:00",
        "payload": {
            "invocation_id": "invoke-1",
            "model_id": "model-1",
            "model_role": "reasoning",
            "status": "completed",
            "prompt": "private prompt",
            "arguments": {"secret": "value"},
            "result": "private result",
        },
    }

    record = TelemetryRecord.from_runtime_event(
        event,
        run_id="run-1",
        thread_id="thread-1",
        turn_id="turn-1",
    )

    assert record.schema_version == "harness-telemetry-v2"
    assert record.telemetry_id
    assert record.event_id == "event-1"
    assert record.sequence == 7
    assert record.run_version == 3
    assert record.component == "model"
    assert record.operation_id == "invoke-1"
    assert len(record.trace_id) == 32
    assert len(record.span_id) == 16
    assert record.privacy_class == "operational_metadata"
    assert record.attributes == {
        "source_event_type": "model.completed",
        "invocation_id": "invoke-1",
        "model_id": "model-1",
        "model_role": "reasoning",
        "status": "completed",
    }


def test_direct_telemetry_records_receive_stable_identity():
    first = TelemetryRecord(
        event_name="runtime.settled",
        run_id="run-1",
        turn_id="turn-1",
        status="completed",
    )
    second = TelemetryRecord(
        event_name="runtime.settled",
        run_id="run-1",
        turn_id="turn-1",
        status="completed",
    )

    assert first.telemetry_id == second.telemetry_id
    assert first.trace_id == second.trace_id
    assert first.span_id == second.span_id


def test_tool_telemetry_projects_nested_identity_without_arguments() -> None:
    event = {
        "event_type": "tool.completed",
        "event_id": "event-tool-1",
        "payload": {
            "tool_result": {
                "name": "calendar.select_dates",
                "tool_call_id": "call-1",
                "status": "succeeded",
                "arguments": {"private": "value"},
            }
        },
    }

    record = TelemetryRecord.from_runtime_event(
        event,
        run_id="run-1",
        thread_id="thread-1",
        turn_id="turn-1",
    )

    assert record.attributes["tool_name"] == "calendar.select_dates"
    assert record.attributes["tool_call_id"] == "call-1"
    assert record.attributes["tool_status"] == "succeeded"
    assert len(record.attributes["argument_digest"]) == 24
    assert "arguments" not in record.attributes


def test_logging_telemetry_emits_structured_trace_payload(caplog):
    telemetry = LoggingTelemetry()
    record = TelemetryRecord(event_name="tool.completed", run_id="run-logging")

    with caplog.at_level(logging.INFO, logger="deepkeel.telemetry"):
        telemetry.record(record)

    assert "harness_telemetry" in caplog.text
    assert record.trace_id in caplog.text
    assert record.span_id in caplog.text


def test_logging_telemetry_suppresses_ephemeral_stream_events_by_default(caplog):
    record = TelemetryRecord(
        event_name="model.delta",
        run_id="run-stream",
        ephemeral=True,
    )

    with caplog.at_level(logging.INFO, logger="deepkeel.telemetry"):
        LoggingTelemetry().record(record)
    assert "harness_telemetry" not in caplog.text

    with caplog.at_level(logging.INFO, logger="deepkeel.telemetry"):
        LoggingTelemetry(include_ephemeral=True).record(record)
    assert "harness_telemetry" in caplog.text


def test_trace_store_queries_deduplicates_and_purges_records() -> None:
    store = InMemoryTelemetry()
    now = datetime.now(UTC)
    old = TelemetryRecord(
        event_name="run.started",
        run_id="run-1",
        occurred_at=now - timedelta(days=2),
    )
    recent = TelemetryRecord(
        event_name="tool.completed",
        run_id="run-1",
        component="tool",
        occurred_at=now,
    )
    store.record(old)
    store.record(old)
    store.record(recent)

    page = store.query(TraceQuery(run_id="run-1", limit=2))
    assert [record.event_name for record in page.records] == [
        "run.started",
        "tool.completed",
    ]
    assert page.truncated is False
    assert store.query(TraceQuery(component="tool", limit=1)).records == [recent]
    assert store.delete_before(now - timedelta(days=1)) == 1
    assert store.query(TraceQuery(run_id="run-1")).records == [recent]


def test_composite_telemetry_fans_out_to_each_destination() -> None:
    first = InMemoryTelemetry()
    second = InMemoryTelemetry()
    event = TelemetryRecord(event_name="run.completed", run_id="run-2")

    CompositeTelemetry((first, second)).record(event)

    assert first.snapshot() == (event,)
    assert second.snapshot() == (event,)


def test_composite_telemetry_isolates_destination_failures() -> None:
    class BrokenTelemetry:
        def record(self, event: TelemetryRecord) -> None:
            raise RuntimeError("trace backend unavailable")

    healthy = InMemoryTelemetry()
    event = TelemetryRecord(event_name="run.completed", run_id="run-isolated")

    CompositeTelemetry((BrokenTelemetry(), healthy)).record(event)

    assert healthy.snapshot() == (event,)
