from __future__ import annotations

from typing import Any

from deepkeel.budget import BudgetLedger
from deepkeel.context_compaction import WorkingContextCompactor
from deepkeel.model import (
    ModelInvocationRecorder,
    ModelInvocationStore,
    RoutedModelGateway,
)
from deepkeel.model_health import ModelHealthStore
from deepkeel.model_routing import ModelRouter
from deepkeel.policy import PolicyEngine


def build_runtime_model_gateway(
    providers: dict[str, Any],
    *,
    router: ModelRouter,
    policy_engine: PolicyEngine,
    budget_ledger: BudgetLedger,
    invocation_recorder: ModelInvocationRecorder | None,
    invocation_store: ModelInvocationStore | None,
    model_health_store: ModelHealthStore,
    context_compactor: WorkingContextCompactor | None = None,
) -> RoutedModelGateway:
    """Compose the governed model pipeline outside the orchestration runtime."""

    return RoutedModelGateway(
        providers,
        router=router,
        policy_engine=policy_engine,
        budget_ledger=budget_ledger,
        invocation_recorder=invocation_recorder,
        invocation_store=invocation_store,
        model_health_store=model_health_store,
        context_compactor=context_compactor,
    )
