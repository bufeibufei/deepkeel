from datetime import UTC, datetime, timedelta

import pytest

from harness_core.budget import InMemoryBudgetLedger
from harness_core.contracts import AgentMessage, MessageContentPart
from harness_core.model import RoutedModelGateway, _json_arguments
from harness_core.model_failures import ModelToolArgumentsError
from harness_core.model_health import InMemoryModelHealthStore
from harness_core.model_routing import ModelStepContext
from harness_core.policy import DefaultPolicyEngine


class ProviderError(Exception):
    status_code = 503


class FakeProvider:
    def __init__(self, model: str, role: str, *, fails: bool = False) -> None:
        self.model = model
        self.model_role = role
        self.base_url = "https://provider.example/v1"
        self.fails = fails
        self.calls = 0

    def complete_chat(self, *_args, **_kwargs):
        self.calls += 1
        if self.fails:
            raise ProviderError("temporarily unavailable")
        return {
            "message": {"role": "assistant", "content": self.model},
            "finish_reason": "stop",
            "model": self.model,
        }


class MalformedThenValidToolProvider:
    def __init__(self) -> None:
        self.model = "vision-model"
        self.model_role = "reasoning"
        self.base_url = "https://provider.example/v1"
        self.model_capabilities = {
            "supports_image_input": True,
            "supports_forced_tool_choice": True,
            "source": "test_catalog",
        }
        self.calls: list[list[dict]] = []

    def complete_chat(self, messages, *_args, **_kwargs):
        self.calls.append(messages)
        arguments = (
            '{"subject_scope" "person"}'
            if len(self.calls) == 1
            else '{"subject_scope":"person"}'
        )
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"tool-call-{len(self.calls)}",
                        "type": "function",
                        "function": {
                            "name": "vision.read_face",
                            "arguments": arguments,
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
            "model": self.model,
        }


class MissingThenValidToolProvider(MalformedThenValidToolProvider):
    def complete_chat(self, messages, *_args, **_kwargs):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return {
                "message": {"role": "assistant", "content": "I can analyze the image."},
                "finish_reason": "stop",
                "model": self.model,
            }
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tool-call-repaired",
                        "type": "function",
                        "function": {
                            "name": "vision.read_face",
                            "arguments": '{"subject_scope":"person"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
            "model": self.model,
        }


def _context() -> ModelStepContext:
    return ModelStepContext(
        run_id="run-1",
        user_id="user-1",
        thread_id="thread-1",
        turn_id="turn-1",
        step_index=0,
        message_count=1,
        observation_count=1,
        tool_result_count=0,
        available_roles=("reasoning", "fast"),
        model_policy={"mode": "adaptive", "failure_policy": "auto_fallback"},
    )


def test_in_memory_health_store_opens_and_recovers_circuit() -> None:
    store = InMemoryModelHealthStore(failure_threshold=2, cooldown_seconds=60)
    first = store.record_failure("provider", "model", category="timeout")
    second = store.record_failure("provider", "model", category="timeout")

    assert first.is_available() is True
    assert second.is_available() is False
    assert second.consecutive_failures == 2

    recovered = second.is_available(
        now=datetime.now(UTC) + timedelta(seconds=61)
    )
    assert recovered is True
    assert store.record_success("provider", "model").consecutive_failures == 0


def test_routed_gateway_records_failure_and_falls_back() -> None:
    health = InMemoryModelHealthStore(failure_threshold=1)
    reasoning = FakeProvider("reasoning-model", "reasoning", fails=True)
    fast = FakeProvider("fast-model", "fast")
    gateway = RoutedModelGateway(
        {"reasoning": reasoning, "fast": fast},
        router=None,
        policy_engine=DefaultPolicyEngine(),
        budget_ledger=InMemoryBudgetLedger(),
        model_health_store=health,
    )

    result = gateway.run_turn(
        [AgentMessage(id="message-1", role="user", content="hello")],
        tools=[],
        step_context=_context(),
    )

    assert result.content == "fast-model"
    assert health.snapshot(
        gateway.providers["reasoning"].info.provider_id,
        "reasoning-model",
    ).is_available() is False


def test_routed_gateway_skips_an_open_model_without_invoking_it() -> None:
    health = InMemoryModelHealthStore(failure_threshold=1)
    reasoning = FakeProvider("reasoning-model", "reasoning")
    fast = FakeProvider("fast-model", "fast")
    gateway = RoutedModelGateway(
        {"reasoning": reasoning, "fast": fast},
        router=None,
        policy_engine=DefaultPolicyEngine(),
        budget_ledger=InMemoryBudgetLedger(),
        model_health_store=health,
    )
    provider_id = gateway.providers["reasoning"].info.provider_id
    health.record_failure(
        provider_id,
        "reasoning-model",
        category="rate_limited",
        immediate=True,
    )

    result = gateway.run_turn(
        [AgentMessage(id="message-1", role="user", content="hello")],
        tools=[],
        step_context=_context(),
    )

    assert result.content == "fast-model"
    assert reasoning.calls == 0
    assert fast.calls == 1


def test_routed_gateway_fails_explicitly_when_all_models_are_open() -> None:
    health = InMemoryModelHealthStore(failure_threshold=1)
    reasoning = FakeProvider("reasoning-model", "reasoning")
    fast = FakeProvider("fast-model", "fast")
    gateway = RoutedModelGateway(
        {"reasoning": reasoning, "fast": fast},
        router=None,
        policy_engine=DefaultPolicyEngine(),
        budget_ledger=InMemoryBudgetLedger(),
        model_health_store=health,
    )
    for adapter in gateway.providers.values():
        health.record_failure(
            adapter.info.provider_id,
            adapter.info.model_id,
            category="rate_limited",
            immediate=True,
        )

    with pytest.raises(RuntimeError, match="model invocation attempts exhausted"):
        gateway.run_turn(
            [AgentMessage(id="message-1", role="user", content="hello")],
            tools=[],
            step_context=_context(),
        )

    assert reasoning.calls == 0
    assert fast.calls == 0


def test_routed_gateway_repairs_malformed_forced_tool_arguments_once() -> None:
    provider = MalformedThenValidToolProvider()
    routes: list[dict] = []
    gateway = RoutedModelGateway(
        {"reasoning": provider},
        router=None,
        policy_engine=DefaultPolicyEngine(),
        budget_ledger=InMemoryBudgetLedger(),
    )
    context = ModelStepContext(
        run_id="run-tool-repair",
        user_id="user-1",
        thread_id="thread-1",
        turn_id="turn-1",
        step_index=0,
        message_count=1,
        observation_count=0,
        tool_result_count=0,
        available_roles=("reasoning",),
        model_policy={
            "mode": "single",
            "primary_role": "reasoning",
            "failure_policy": "retry_selected",
        },
        forced_tool_name="vision.read_face",
    )
    message = AgentMessage(
        id="message-vision",
        role="user",
        content="Read this face image.",
        content_parts=[
            MessageContentPart(
                type="image",
                uri="attachment://face-image",
                reference_id="face-image",
                media_type="image/jpeg",
            )
        ],
    )

    result = gateway.run_turn(
        [message],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "vision.read_face",
                    "description": "Read visible facial features.",
                    "parameters": {
                        "type": "object",
                        "properties": {"subject_scope": {"type": "string"}},
                        "required": ["subject_scope"],
                    },
                },
            }
        ],
        system_prompt="Use the supplied image as evidence.",
        step_context=context,
        on_route=routes.append,
    )

    assert provider.calls and len(provider.calls) == 2
    assert result.tool_calls[0].name == "vision.read_face"
    assert result.tool_calls[0].arguments == {"subject_scope": "person"}
    assert provider.calls[1][0]["role"] == "system"
    assert "invalid or truncated JSON arguments" in provider.calls[1][0]["content"]
    assert "attachment://face-image" in str(provider.calls[1])
    assert any(
        route.get("failure_category") == "tool_arguments_invalid"
        for route in routes
    )


def test_routed_gateway_repairs_missing_forced_tool_call_once() -> None:
    provider = MissingThenValidToolProvider()
    gateway = RoutedModelGateway(
        {"reasoning": provider},
        router=None,
        policy_engine=DefaultPolicyEngine(),
        budget_ledger=InMemoryBudgetLedger(),
    )
    context = ModelStepContext(
        run_id="run-tool-contract-repair",
        user_id="user-1",
        thread_id="thread-1",
        turn_id="turn-1",
        step_index=0,
        message_count=1,
        observation_count=0,
        tool_result_count=0,
        available_roles=("reasoning",),
        model_policy={
            "mode": "single",
            "primary_role": "reasoning",
            "failure_policy": "retry_selected",
        },
        forced_tool_name="vision.read_face",
    )

    result = gateway.run_turn(
        [AgentMessage(id="message-vision", role="user", content="Read this image.")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "vision.read_face",
                    "description": "Read visible facial features.",
                    "parameters": {
                        "type": "object",
                        "properties": {"subject_scope": {"type": "string"}},
                        "required": ["subject_scope"],
                    },
                },
            }
        ],
        system_prompt="Use the image as evidence.",
        step_context=context,
    )

    assert result.tool_calls[0].name == "vision.read_face"
    assert len(provider.calls) == 2
    assert "violated the required tool contract" in provider.calls[1][0]["content"]
    assert "vision.read_face" in provider.calls[1][0]["content"]


def test_json_arguments_repairs_structural_eof_truncation() -> None:
    parsed = _json_arguments(
        '{"quality":{"suitable":true},'
        '"observations":[{"region":"entry","visible_feature":"clear path"}],'
        '"conclusion":"keep the route unobstructed'
    )

    assert parsed == {
        "quality": {"suitable": True},
        "observations": [
            {"region": "entry", "visible_feature": "clear path"}
        ],
        "conclusion": "keep the route unobstructed",
    }


def test_json_arguments_repairs_structural_trailing_comma() -> None:
    assert _json_arguments('{"quality":{"suitable":true},') == {
        "quality": {"suitable": True}
    }


def test_json_arguments_rejects_non_structural_json_corruption() -> None:
    with pytest.raises(ModelToolArgumentsError):
        _json_arguments('{"subject_scope" "person"}')
