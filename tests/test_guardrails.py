from __future__ import annotations

import asyncio

import pytest

from deepkeel.contracts import Artifact, Observation, ToolCall, ToolResult
from deepkeel.guardrails import (
    GuardrailAction,
    GuardrailDecision,
    GuardrailExecutionError,
    GuardrailRequest,
    GuardrailRunner,
    GuardrailScope,
    GuardrailSpec,
    GuardrailStage,
)
from deepkeel.model import ModelInvocation, ModelProviderInfo, ModelTurn
from deepkeel.runtime import HarnessRuntime
from deepkeel.runtime_api import RuntimeRequest
from deepkeel.tool_registry import ToolRegistry, ToolSpec
from deepkeel.tools import ToolExecutionContext, ToolExecutor


def _request(stage: GuardrailStage = GuardrailStage.MODEL_INPUT) -> GuardrailRequest:
    return GuardrailRequest(
        stage=stage,
        operation_id=f"run-1:turn-1:{stage.value}",
        run_id="run-1",
        turn_id="turn-1",
        package_ids=("pack-a",),
        skill_id="skill-a",
        tool_name="demo.read",
        payload={"content": "original"},
    )


def _tool() -> ToolSpec:
    return ToolSpec(
        name="demo.read",
        description="Read demo data",
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        required_args=["query"],
        read_only=True,
    )


def test_guardrails_merge_transforms_in_priority_order_and_replay() -> None:
    calls = 0
    runner = GuardrailRunner()

    def first(_: GuardrailRequest) -> GuardrailDecision:
        nonlocal calls
        calls += 1
        return GuardrailDecision(
            action=GuardrailAction.TRANSFORM,
            payload_patch={"content": "redacted"},
            redactions=("secret",),
        )

    runner.register(
        GuardrailSpec(
            id="first",
            stage=GuardrailStage.MODEL_INPUT,
            priority=10,
            handler=first,
        )
    )
    runner.register(
        GuardrailSpec(
            id="skill",
            stage=GuardrailStage.MODEL_INPUT,
            scope=GuardrailScope.SKILL,
            selector="skill-a",
            handler=lambda _: GuardrailDecision(diagnostics={"checked": True}),
        )
    )

    first_result = asyncio.run(runner.arun(_request()))
    replay = asyncio.run(runner.arun(_request()))

    assert first_result.decision.action == GuardrailAction.TRANSFORM
    assert first_result.decision.payload_patch["content"] == "redacted"
    assert first_result.decision.diagnostics["checked"] is True
    assert calls == 1
    assert all(audit.replayed for audit in replay.audits)


def test_required_guardrail_fails_closed_but_optional_failure_is_isolated() -> None:
    optional = GuardrailRunner()
    optional.register(
        GuardrailSpec(
            id="optional",
            stage=GuardrailStage.INPUT,
            required=False,
            handler=lambda _: (_ for _ in ()).throw(RuntimeError("offline")),
        )
    )
    result = asyncio.run(optional.arun(_request(GuardrailStage.INPUT)))
    assert result.decision.action == GuardrailAction.ALLOW
    assert result.audits[0].status == "failed"

    required = GuardrailRunner()
    required.register(
        GuardrailSpec(
            id="required",
            stage=GuardrailStage.INPUT,
            handler=lambda _: (_ for _ in ()).throw(RuntimeError("offline")),
        )
    )
    with pytest.raises(GuardrailExecutionError, match="offline"):
        asyncio.run(required.arun(_request(GuardrailStage.INPUT)))


def test_tool_guardrails_transform_input_output_and_attach_provenance() -> None:
    registry = ToolRegistry([_tool()])
    runner = GuardrailRunner()
    runner.register(
        GuardrailSpec(
            id="normalize-query",
            stage=GuardrailStage.TOOL_INPUT,
            handler=lambda _: GuardrailDecision(
                action=GuardrailAction.TRANSFORM,
                payload_patch={"arguments": {"query": "safe"}},
            ),
        )
    )
    runner.register(
        GuardrailSpec(
            id="redact-output",
            stage=GuardrailStage.TOOL_OUTPUT,
            handler=lambda _: GuardrailDecision(
                action=GuardrailAction.TRANSFORM,
                payload_patch={"result": {"summary": "redacted"}},
            ),
        )
    )
    executor = ToolExecutor(registry, guardrail_runner=runner)

    def handler(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        assert call.arguments["query"] == "safe"
        return ToolResult(
            call=call,
            status="succeeded",
            summary="secret",
            observation=Observation(
                id="obs-1",
                run_id=context.run_id,
                tool_call_id=call.id,
                source=call.name,
                status="succeeded",
            ),
            artifacts=[
                Artifact(
                    id="artifact-1",
                    run_id=context.run_id,
                    artifact_type="demo.result",
                )
            ],
        )

    executor.register("demo.read", handler)
    result = asyncio.run(
        executor.aexecute(
            ToolCall(id="call-1", name="demo.read", arguments={"query": "secret"}),
            ToolExecutionContext(run_id="run-1", user_id="user-1"),
        )
    )

    assert result.status == "succeeded"
    assert result.summary == "redacted"
    assert result.observation is not None
    assert result.observation.provenance.origin == "tool"
    assert result.observation.provenance.trust_level == "external"
    assert result.artifacts[0].provenance.parent_ids == ["call-1"]


def test_tool_input_guardrail_can_require_approval_before_handler_runs() -> None:
    registry = ToolRegistry([_tool()])
    runner = GuardrailRunner()
    runner.register(
        GuardrailSpec(
            id="approval",
            stage=GuardrailStage.TOOL_INPUT,
            handler=lambda _: GuardrailDecision(
                action=GuardrailAction.REQUIRE_APPROVAL,
                reason="external side effect",
            ),
        )
    )
    executor = ToolExecutor(registry, guardrail_runner=runner)
    called = False

    def handler(call: ToolCall, _: ToolExecutionContext) -> ToolResult:
        nonlocal called
        called = True
        return ToolResult(call=call, status="succeeded")

    executor.register("demo.read", handler)
    result = asyncio.run(
        executor.aexecute(
            ToolCall(id="call-1", name="demo.read", arguments={"query": "x"}),
            ToolExecutionContext(run_id="run-1", user_id="user-1"),
        )
    )

    assert result.status == "requires_user_action"
    assert result.pending_action is not None
    assert result.pending_action.payload["source"] == "guardrail"
    assert called is False


class _Provider:
    info = ModelProviderInfo(
        provider_id="example.guardrail",
        model_id="guardrail-model",
        model_role="fast",
    )

    def invoke(self, request: ModelInvocation, *, on_text_delta=None) -> ModelTurn:
        assert request.messages[-1]["content"] in {"hello", "safe question"}
        if on_text_delta is not None:
            on_text_delta("unsafe ")
            on_text_delta("answer")
        return ModelTurn(
            content="unsafe answer",
            finish_reason="stop",
            model_id=self.info.model_id,
            model_role=self.info.model_role,
        )


def test_model_output_guardrail_buffers_stream_and_releases_transformed_answer() -> None:
    runner = GuardrailRunner()
    runner.register(
        GuardrailSpec(
            id="model-output",
            stage=GuardrailStage.MODEL_OUTPUT,
            handler=lambda _: GuardrailDecision(
                action=GuardrailAction.TRANSFORM,
                payload_patch={"content": "safe answer"},
            ),
        )
    )
    registry = ToolRegistry()
    runtime = HarnessRuntime(
        registry,
        ToolExecutor(registry),
        guardrail_runner=runner,
    )

    streamed: list[dict[str, object]] = []
    result = runtime.run(
        RuntimeRequest(question="hello", context_bundle={"agent_session_id": "guarded-run"}),
        provider=_Provider(),
        event_sink=streamed.append,
    )

    assert result.status.value == "completed"
    assert result.final_answer is not None
    assert result.final_answer.markdown == "safe answer"
    deltas = [
        event for event in streamed if event.get("source_event_type") == "model.delta"
    ]
    assert [event["payload"]["delta"] for event in deltas] == ["safe answer"]
    assert deltas[0]["payload"]["stream_mode"] == "guardrail_buffered"
    assert any(event.event_type == "guardrail.evaluated" for event in result.events)


def test_input_guardrail_transforms_question_before_context_and_model_execution() -> None:
    runner = GuardrailRunner()
    runner.register(
        GuardrailSpec(
            id="input",
            stage=GuardrailStage.INPUT,
            handler=lambda _: GuardrailDecision(
                action=GuardrailAction.TRANSFORM,
                payload_patch={"question": "safe question"},
            ),
        )
    )
    registry = ToolRegistry()
    runtime = HarnessRuntime(
        registry,
        ToolExecutor(registry),
        guardrail_runner=runner,
    )

    result = runtime.run(
        RuntimeRequest(question="raw question", context_bundle={"agent_session_id": "input-run"}),
        provider=_Provider(),
    )

    assert result.status.value == "completed"
    assert result.question == "safe question"
    assert any(
        event.event_type == "guardrail.evaluated"
        and event.payload.get("stage") == "input"
        for event in result.events
    )


def test_final_answer_guardrail_can_block_completion() -> None:
    runner = GuardrailRunner()
    runner.register(
        GuardrailSpec(
            id="final-output",
            stage=GuardrailStage.FINAL_OUTPUT,
            handler=lambda _: GuardrailDecision(
                action=GuardrailAction.BLOCK,
                reason="unsafe final answer",
            ),
        )
    )
    registry = ToolRegistry()
    runtime = HarnessRuntime(
        registry,
        ToolExecutor(registry),
        guardrail_runner=runner,
    )

    result = runtime.run(
        RuntimeRequest(question="hello", context_bundle={"agent_session_id": "blocked-run"}),
        provider=_Provider(),
    )

    assert result.status.value == "failed"
    assert result.final_answer is not None
    assert "unsafe final answer" in result.final_answer.markdown
