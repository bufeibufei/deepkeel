from harness_core.budget import InMemoryBudgetLedger
from harness_core.model import (
    MODEL_FAILURE_AUTO_FALLBACK,
    MODEL_FAILURE_FAIL_FAST,
    MODEL_FAILURE_RETRY_SELECTED,
    RoutedModelGateway,
)
from harness_core.model_routing import ModelRouteDecision
from harness_core.policy import DefaultPolicyEngine


class FakeProvider:
    def __init__(self, model: str, role: str) -> None:
        self.model = model
        self.model_role = role
        self.base_url = "https://provider.example/v1"

    def complete_chat(self, *_args, **_kwargs):
        return {"content": "ok"}


def _gateway(*, same_model: bool = False) -> RoutedModelGateway:
    return RoutedModelGateway(
        {
            "fast": FakeProvider("model-a", "fast"),
            "reasoning": FakeProvider(
                "model-a" if same_model else "model-b",
                "reasoning",
            ),
        },
        router=None,
        policy_engine=DefaultPolicyEngine(),
        budget_ledger=InMemoryBudgetLedger(),
    )


def _route() -> ModelRouteDecision:
    return ModelRouteDecision(
        role="reasoning",
        reason="test",
        router_id="test-router",
    )


def test_fail_fast_only_attempts_the_selected_binding() -> None:
    attempts = _gateway()._attempts(
        _route(),
        failure_policy=MODEL_FAILURE_FAIL_FAST,
    )

    assert [(item[0].role, item[2]) for item in attempts] == [
        ("reasoning", "primary")
    ]


def test_retry_selected_never_crosses_model_roles() -> None:
    attempts = _gateway()._attempts(
        _route(),
        failure_policy=MODEL_FAILURE_RETRY_SELECTED,
    )

    assert [(item[0].role, item[2]) for item in attempts] == [
        ("reasoning", "primary"),
        ("reasoning", "retry"),
    ]
    assert attempts[0][1].info.model_id == attempts[1][1].info.model_id


def test_auto_fallback_uses_only_a_distinct_configured_binding() -> None:
    attempts = _gateway()._attempts(
        _route(),
        failure_policy=MODEL_FAILURE_AUTO_FALLBACK,
    )

    assert [(item[0].role, item[2]) for item in attempts] == [
        ("reasoning", "primary"),
        ("fast", "fallback"),
    ]
    assert attempts[1][1].info.model_id == "model-a"


def test_auto_fallback_retries_when_both_roles_bind_the_same_model() -> None:
    attempts = _gateway(same_model=True)._attempts(
        _route(),
        failure_policy=MODEL_FAILURE_AUTO_FALLBACK,
    )

    assert [(item[0].role, item[2]) for item in attempts] == [
        ("reasoning", "primary"),
        ("reasoning", "retry"),
    ]
