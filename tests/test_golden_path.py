from __future__ import annotations

import asyncio

import pytest

from deepkeel.model import ModelInvocation, ModelProviderInfo, ModelTurn
from deepkeel.runtime_sdk import AgentDefaults, AgentHarness, RuntimeRequest


class EchoProvider:
    info = ModelProviderInfo(
        provider_id="test.echo",
        model_id="echo-v1",
        model_role="fast",
    )

    def invoke(self, request: ModelInvocation, *, on_text_delta=None) -> ModelTurn:
        content = f"echo:{request.messages[-1]['content']}"
        if on_text_delta is not None:
            on_text_delta(content)
        return ModelTurn(
            content=content,
            finish_reason="stop",
            model_id=self.info.model_id,
            model_role=self.info.model_role,
        )


def test_agent_harness_runs_string_with_typed_defaults() -> None:
    harness = AgentHarness.create(
        provider=EchoProvider(),
        defaults=AgentDefaults(
            user_id="operator-1",
            tenant_id="tenant-1",
            namespace="support",
        ),
    )

    result = harness.run("hello", thread_id="thread-1")

    assert result.final_answer.markdown == "echo:hello"
    assert result.run_context.user_id == "operator-1"
    assert result.thread_id == "thread-1"


def test_agent_harness_accepts_prebuilt_request_without_hiding_runtime() -> None:
    harness = AgentHarness.create(provider=EchoProvider())
    request = RuntimeRequest(question="advanced", user_id="advanced-user")

    result = harness.run(request)

    assert result.final_answer.markdown == "echo:advanced"
    assert harness.runtime is not None
    with pytest.raises(ValueError, match="only supported when request is a string"):
        harness.run(request, user_id="different")


def test_agent_harness_async_run_and_stream_share_the_same_facade() -> None:
    async def scenario():
        harness = AgentHarness.create(provider=EchoProvider())
        result = await harness.arun("async")
        events = [event async for event in harness.astream("stream")]
        return result, events

    result, events = asyncio.run(scenario())

    assert result.final_answer.markdown == "echo:async"
    assert any(event.event_type == "answer.delta" for event in events)
    assert any(event.event_type == "final_answer" for event in events)
