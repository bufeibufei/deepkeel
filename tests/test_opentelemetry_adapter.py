from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from deepkeel.contrib.otel import OpenTelemetryTelemetry
from deepkeel.telemetry import TelemetryRecord


def _adapter() -> tuple[OpenTelemetryTelemetry, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("deepkeel.tests")
    return OpenTelemetryTelemetry(tracer), exporter


def test_adapter_preserves_trace_identity_and_safe_attributes() -> None:
    adapter, exporter = _adapter()
    event = TelemetryRecord(
        event_name="tool.failed",
        run_id="run-otel",
        user_id="private-user",
        sequence=7,
        status="failed",
        attributes={
            "tool_name": "lookup",
            "error_code": "TOOL_TIMEOUT",
            "prompt": "private prompt",
            "duration_ms": 125,
            "unsafe": {"prompt": "private"},
        },
    )

    adapter.record(event)
    adapter.record(
        TelemetryRecord(
            event_name="runtime.settled",
            run_id="run-otel",
            sequence=8,
            status="failed",
        )
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    span = next(item for item in spans if item.name == "deepkeel.tool.failed")
    root = next(item for item in spans if item.name == "deepkeel.run.segment")
    assert span.context.trace_id == root.context.trace_id
    assert span.parent.span_id == root.context.span_id
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["deepkeel.attr.tool_name"] == "lookup"
    assert span.attributes["deepkeel.attr.error_code"] == "TOOL_TIMEOUT"
    assert "deepkeel.attr.unsafe" not in span.attributes
    assert "deepkeel.attr.prompt" not in span.attributes
    assert span.attributes["deepkeel.attr.duration_ms"] == 125
    assert "deepkeel.user_id" not in span.attributes
    assert root.attributes["deepkeel.runtime_trace_id"] == event.trace_id
    assert root.attributes["deepkeel.segment.operation_count"] == 1
    assert root.events[0].name == "runtime.settled"


def test_adapter_filters_ephemeral_records_and_requires_explicit_user_id_opt_in() -> None:
    adapter, exporter = _adapter()
    adapter.record(
        TelemetryRecord(
            event_name="stream.delta",
            run_id="run-ephemeral",
            user_id="private-user",
            ephemeral=True,
        )
    )
    assert exporter.get_finished_spans() == ()

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    opted_in = OpenTelemetryTelemetry(
        provider.get_tracer("deepkeel.tests.user"),
        include_user_id=True,
    )
    opted_in.record(
        TelemetryRecord(
            event_name="run.completed",
            run_id="run-user",
            user_id="known-user",
        )
    )
    opted_in.shutdown()
    operation = next(
        item for item in exporter.get_finished_spans() if item.name == "deepkeel.run.segment"
    )
    assert operation.attributes["deepkeel.run_id"] == "run-user"
    assert operation.attributes["deepkeel.user_id"] == "known-user"


def test_adapter_groups_events_deduplicates_replay_and_links_resumed_segments() -> None:
    adapter, exporter = _adapter()
    started = TelemetryRecord(
        event_id="event-started",
        event_name="run.started",
        run_id="run-grouped",
        turn_id="turn-1",
        run_version=1,
        sequence=1,
    )
    adapter.record(started)
    adapter.record(started.model_copy(deep=True))
    for sequence in range(2, 5):
        adapter.record(
            TelemetryRecord(
                event_name="answer.delta",
                run_id="run-grouped",
                turn_id="turn-1",
                run_version=1,
                sequence=sequence,
                ephemeral=True,
            )
        )
    adapter.record(
        TelemetryRecord(
            event_name="runtime.settled",
            run_id="run-grouped",
            turn_id="turn-1",
            run_version=1,
            sequence=5,
            status="completed",
        )
    )
    adapter.record(
        TelemetryRecord(
            event_name="run.resumed",
            run_id="run-grouped",
            turn_id="turn-1",
            run_version=2,
            sequence=6,
        )
    )
    adapter.record(
        TelemetryRecord(
            event_name="runtime.settled",
            run_id="run-grouped",
            turn_id="turn-1",
            run_version=2,
            sequence=7,
            status="completed",
        )
    )

    roots = [
        item for item in exporter.get_finished_spans() if item.name == "deepkeel.run.segment"
    ]
    assert len(roots) == 2
    assert roots[0].attributes["deepkeel.segment.event_count"] == 2
    assert roots[0].attributes["deepkeel.segment.delta_count"] == 3
    assert roots[1].attributes["deepkeel.segment.resumed"] is True
    assert roots[1].links[0].context.trace_id == roots[0].context.trace_id
