from __future__ import annotations

from pathlib import Path
import asyncio

from deepkeel.event_projection import project_telemetry_record, project_trace_item
from deepkeel.execution_engine import LangGraphExecutionEngine
from deepkeel.extension_sdk import (
    CapabilityPackageSource,
    CapabilityTrustPolicy,
    capability_source_digest,
    evaluate_capability_trust,
)
from deepkeel.runtime_api import RuntimeEventEnvelope
from deepkeel.runtime_sdk import AgentHarness
from deepkeel.extension_sdk import ToolExecutionContext


class LocalProvider:
    model = "boundary-model"
    model_role = "fast"

    def complete_chat(self, messages, **_kwargs):
        return {
            "message": {"role": "assistant", "content": "ready"},
            "finish_reason": "stop",
            "model": self.model,
        }


def test_runtime_result_summary_omits_internal_execution_payloads() -> None:
    harness = AgentHarness.create(provider=LocalProvider())

    result = harness.run(
        "status",
        context_bundle={"agent_session_id": "summary-run"},
    )
    summary = result.to_summary()

    assert summary.run_id == result.run_id
    assert summary.final_answer.markdown == "ready"
    assert "checkpoint" not in type(summary).model_fields
    assert "diagnostics" not in type(summary).model_fields
    assert "trace" not in type(summary).model_fields
    assert "observations" not in type(summary).model_fields


def test_event_projections_share_the_canonical_envelope_identity() -> None:
    envelope = RuntimeEventEnvelope(
        event_id="event-1",
        sequence=4,
        run_version=2,
        run_id="run-1",
        thread_id="thread-1",
        turn_id="turn-1",
        event_type="tool.completed",
        source_event_type="tool.completed",
        title="Tool completed",
        summary="Lookup finished",
        payload={"tool_call_id": "call-1", "status": "succeeded"},
    )

    telemetry = project_telemetry_record(envelope)
    trace = project_trace_item(envelope)

    assert telemetry.event_id == trace["event_id"] == "event-1"
    assert telemetry.sequence == trace["sequence"] == 4
    assert telemetry.event_name == trace["event_type"] == "tool.completed"


def test_in_process_capability_requires_an_allowlisted_digest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    package = tmp_path / "package.py"
    manifest.write_text('{"id":"demo"}', encoding="utf-8")
    package.write_text("VALUE = 1\n", encoding="utf-8")
    digest = capability_source_digest(manifest, package)
    source = CapabilityPackageSource(
        package_id="demo",
        content_sha256=digest,
    )

    denied = evaluate_capability_trust(source, CapabilityTrustPolicy())
    allowed = evaluate_capability_trust(
        source,
        CapabilityTrustPolicy(allowed_in_process_digests=frozenset({digest})),
    )

    assert denied.trusted is False
    assert allowed.trusted is True


def test_isolated_capability_uses_host_source_allowlist() -> None:
    source = CapabilityPackageSource(
        package_id="remote.search",
        execution_mode="isolated",
        source_uri="https://mcp.example.test/search",
    )
    policy = CapabilityTrustPolicy(
        allowed_isolated_sources=("https://mcp.example.test/",),
    )

    assert evaluate_capability_trust(source, policy).trusted is True


def test_langgraph_adapter_owns_engine_specific_resume_calls() -> None:
    class FakeGraph:
        async def aresume(self, thread_id, payload, **kwargs):
            return {
                "run_id": "run-1",
                "thread_id": thread_id,
                "turn_id": "turn-1",
                "messages": [],
                "observations": [],
                "tool_results": [],
                "artifacts": [],
                "references": [],
                "evidence": [],
                "pending_tool_calls": [],
                "resume_payload": payload,
                "status": "running",
                "stop_reason": "",
            }

    engine = LangGraphExecutionEngine(FakeGraph())  # type: ignore[arg-type]
    result = asyncio.run(
        engine.aresume(
            "thread-1",
            {"approved": True},
            tool_context=ToolExecutionContext(run_id="run-1", user_id="user-1"),
        )
    )

    assert result["thread_id"] == "thread-1"
    assert result["resume_payload"] == {"approved": True}
