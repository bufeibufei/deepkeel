from __future__ import annotations

from typing import Any

from harness_core.budget import (
    ELAPSED_SECONDS,
    INPUT_TOKENS,
    MODEL_CALLS,
    MODEL_RETRIES,
    OUTPUT_TOKENS,
    TOOL_CALLS,
    TOOL_CONCURRENCY,
)
from harness_core.model import ModelProviderAdapter
from harness_core.type_narrowing import as_dict

def _resolved_model_policy(
    value: dict[str, Any] | None,
    *,
    provider: Any,
    providers: dict[str, Any] | None,
    max_steps: int,
) -> dict[str, Any]:
    policy = dict(value) if isinstance(value, dict) else {}
    catalog = _model_providers(provider, providers, policy)
    if policy.get("mode") not in {"single", "adaptive"}:
        policy["mode"] = "adaptive" if len(catalog) > 1 else "single"
    primary = str(policy.get("primary_role") or "reasoning")
    if primary not in catalog:
        primary = "reasoning" if "reasoning" in catalog else next(iter(catalog), "reasoning")
    policy["primary_role"] = primary
    policy["available_roles"] = list(catalog)
    budget = as_dict(policy.get("budget"))
    policy["budget"] = {
        "max_model_calls": _positive_limit(budget.get("max_model_calls"), max_steps),
        "max_tool_calls": _positive_limit(budget.get("max_tool_calls"), 0),
        "max_elapsed_seconds": _positive_number(
            budget.get("max_elapsed_seconds"),
            900.0,
        ),
        "max_total_elapsed_seconds": _positive_number(
            budget.get("max_total_elapsed_seconds"),
            0,
        ),
        "max_request_seconds": _positive_number(
            budget.get("max_request_seconds"),
            0,
        ),
        "max_input_tokens_total": _positive_limit(
            budget.get("max_input_tokens_total"),
            0,
        ),
        "max_input_tokens_per_call": _positive_limit(
            budget.get("max_input_tokens_per_call"),
            0,
        ),
        "max_output_tokens_total": _positive_limit(
            budget.get("max_output_tokens_total"),
            0,
        ),
        "max_output_tokens_per_call": _positive_limit(
            budget.get("max_output_tokens_per_call"),
            0,
        ),
        "max_model_retries": _positive_limit(
            budget.get("max_model_retries"),
            0,
        ),
        "max_parallel_tools": _positive_limit(
            budget.get("max_parallel_tools"),
            4,
        ),
        "roles": {
            str(role): dict(limits)
            for role, limits in as_dict(budget.get("roles")).items()
            if isinstance(limits, dict)
        },
    }
    return policy


def _model_providers(
    provider: Any,
    providers: dict[str, Any] | None,
    model_policy: dict[str, Any],
) -> dict[str, Any]:
    catalog = {
        str(role): candidate
        for role, candidate in (providers or {}).items()
        if candidate is not None
    }
    if provider is not None:
        adapter_role = (
            provider.info.model_role if isinstance(provider, ModelProviderAdapter) else ""
        )
        role = str(
            adapter_role
            or getattr(provider, "model_role", "")
            or model_policy.get("primary_role")
            or "reasoning"
        )
        catalog.setdefault(role, provider)
        catalog.setdefault("reasoning", provider)
    return catalog


def _budget_limits(model_policy: dict[str, Any]) -> dict[str, float]:
    budget = as_dict(model_policy.get("budget"))
    return {
        MODEL_CALLS: float(budget.get("max_model_calls") or 0),
        TOOL_CALLS: float(budget.get("max_tool_calls") or 0),
        INPUT_TOKENS: float(budget.get("max_input_tokens_total") or 0),
        OUTPUT_TOKENS: float(budget.get("max_output_tokens_total") or 0),
        MODEL_RETRIES: float(budget.get("max_model_retries") or 0),
        TOOL_CONCURRENCY: float(budget.get("max_parallel_tools") or 4),
        ELAPSED_SECONDS: float(budget.get("max_total_elapsed_seconds") or 0),
    }


def _positive_limit(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _positive_number(value: Any, default: float) -> float:
    if value is None:
        return max(0.0, float(default))
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return max(0.0, float(default))


def _max_elapsed_seconds(model_policy: dict[str, Any]) -> float:
    budget = as_dict(model_policy.get("budget"))
    return _positive_number(budget.get("max_elapsed_seconds"), 900.0)


def _prior_budget_state(
    durable_state: dict[str, Any],
    short_context: dict[str, Any],
) -> dict[str, Any]:
    durable_runtime = as_dict(durable_state.get("runtime"))
    previous_runtime = as_dict(short_context.get("previous_runtime"))
    sources = (
        durable_state,
        durable_runtime.get("checkpoint"),
        previous_runtime.get("checkpoint"),
    )
    for source in sources:
        if not isinstance(source, dict):
            continue
        snapshot = source.get("budget_state")
        if isinstance(snapshot, dict):
            return snapshot
    return {}


def _merge_skill_activation(
    *,
    durable_state: dict[str, Any],
    session_projection: dict[str, Any],
    explicit: dict[str, Any] | None,
) -> dict[str, Any]:
    durable_context = as_dict(durable_state.get("context_snapshot"))
    projected_context = as_dict(session_projection.get("context_snapshot"))
    sources = (
        durable_context.get("skill_activation"),
        durable_state.get("skill_activation"),
        projected_context.get("skill_activation"),
        session_projection.get("skill_activation"),
        explicit,
    )
    merged: dict[str, Any] = {}
    for source in sources:
        if isinstance(source, dict):
            merged.update(source)
    return merged


def _prior_diagnostics(
    durable_state: dict[str, Any],
    short_context: dict[str, Any],
) -> dict[str, Any]:
    durable_runtime = as_dict(durable_state.get("runtime"))
    previous_runtime = as_dict(short_context.get("previous_runtime"))
    for runtime in (durable_runtime, previous_runtime):
        diagnostics = as_dict(runtime.get("diagnostics"))
        if diagnostics:
            return diagnostics
    return {}
