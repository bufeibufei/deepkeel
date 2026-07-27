from datetime import UTC, datetime, timedelta

import pytest

from harness_core.budget import InMemoryBudgetLedger
from harness_core.contracts import AgentMessage
from harness_core.model import RoutedModelGateway
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
