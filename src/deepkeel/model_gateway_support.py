from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Literal

from deepkeel.budget import (
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    BudgetExceededError,
    BudgetLedger,
    BudgetPolicy,
    BudgetRequest,
    BudgetSnapshot,
    preview_budget,
)
from deepkeel.context_compaction import (
    ContextInputBudgetError,
    ModelInputContextResult,
    prepare_model_input_context,
)
from deepkeel.context_contracts import ModelContextProfile
from deepkeel.context_window import ConservativeTokenEstimator
from deepkeel.contracts import AgentMessage
from deepkeel.model_capabilities import ModelCapabilities
from deepkeel.model_failures import ModelToolArgumentsError, ModelToolContractError
from deepkeel.model_invocations import ModelTurn
from deepkeel.model_provider_contracts import (
    AsyncModelProviderAdapter,
    ModelProviderAdapter,
)
from deepkeel.model_routing import ModelRouteDecision, ModelStepContext
from deepkeel.type_narrowing import as_dict

MODEL_FAILURE_AUTO_FALLBACK = "auto_fallback"


MODEL_FAILURE_RETRY_SELECTED = "retry_selected"


MODEL_FAILURE_FAIL_FAST = "fail_fast"


VALID_MODEL_FAILURE_POLICIES = frozenset(
    {
        MODEL_FAILURE_AUTO_FALLBACK,
        MODEL_FAILURE_RETRY_SELECTED,
        MODEL_FAILURE_FAIL_FAST,
    }
)


DEFAULT_MAX_OUTPUT_TOKENS_BY_ROLE = {
    "fast": 8_192,
    "reasoning": 16_384,
}


DEFAULT_MAX_OUTPUT_TOKENS_PER_CALL = 16_384


MIN_CONTEXT_RESERVE_TOKENS = 2_048


CONTEXT_RESERVE_RATIO = 0.08


def _model_failure_policy(model_policy: dict[str, Any]) -> str:
    value = str(model_policy.get("failure_policy") or MODEL_FAILURE_AUTO_FALLBACK).strip().lower()
    return value if value in VALID_MODEL_FAILURE_POLICIES else MODEL_FAILURE_AUTO_FALLBACK


def _tool_choice(
    step_context: ModelStepContext | None,
    capabilities: ModelCapabilities | None = None,
) -> str | dict[str, Any]:
    forced_tool_name = str(
        step_context.forced_tool_name if step_context is not None else ""
    ).strip()
    if not forced_tool_name:
        return "auto"
    if capabilities is not None and capabilities.supports_forced_tool_choice is False:
        # Preserve the semantic contract through prompt instructions and
        # response validation when a provider only accepts automatic tools.
        return "auto"
    return {
        "type": "function",
        "function": {"name": forced_tool_name},
    }


def _validate_forced_tool_turn(
    turn: ModelTurn,
    step_context: ModelStepContext | None,
) -> None:
    expected = str(step_context.forced_tool_name if step_context is not None else "").strip()
    if not expected:
        return
    actual = [str(call.name or "").strip() for call in turn.tool_calls]
    if len(actual) != 1 or actual[0] != expected:
        raise ModelToolContractError(expected, actual)


def _model_call_limit(model_policy: dict[str, Any]) -> float | None:
    budget = as_dict(model_policy.get("budget"))
    value = budget.get("max_model_calls")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _prepare_model_context(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    provider: ModelProviderAdapter | AsyncModelProviderAdapter,
    route: ModelRouteDecision,
    provider_capabilities: ModelCapabilities,
    budget_policy: BudgetPolicy,
    budget_snapshot: BudgetSnapshot,
    step_context: ModelStepContext,
    per_call_input_limit: float | None,
    token_estimator: ConservativeTokenEstimator,
) -> ModelInputContextResult:
    output_tokens = _remaining_output_tokens(
        budget_policy,
        budget_snapshot,
        route.role,
        capabilities=provider_capabilities,
        estimated_input_tokens=0,
    )
    try:
        return prepare_model_input_context(
            messages,
            tools,
            profile=ModelContextProfile(
                model_id=provider.info.model_id,
                model_role=route.role,
                context_window_tokens=provider_capabilities.context_window_tokens,
                max_output_tokens=output_tokens,
                source=provider_capabilities.source,
            ),
            configured_input_limit=(
                int(per_call_input_limit) if per_call_input_limit is not None else None
            ),
            estimator=token_estimator,
            thread_id=step_context.thread_id,
            subject_id=str(step_context.governance_scope.get("subject_id") or ""),
        )
    except ContextInputBudgetError as exc:
        limit = int(per_call_input_limit or provider_capabilities.context_window_tokens or 1)
        estimated = token_estimator.estimate({"messages": messages, "tools": tools})
        raise BudgetExceededError(
            preview_budget(
                budget_snapshot,
                BudgetRequest(
                    run_id=step_context.accounting_run_id,
                    metric=INPUT_TOKENS,
                    amount=max(limit + 1, estimated),
                    limit=limit,
                    metadata={"reason": ContextInputBudgetError.code},
                ),
            )
        ) from exc


def _remaining_output_tokens(
    policy: BudgetPolicy,
    snapshot,
    role: str,
    *,
    capabilities: ModelCapabilities,
    estimated_input_tokens: int,
) -> int | None:
    total_limit = policy.limit("max_output_tokens_total")
    configured_per_call_limit = policy.limit(
        "max_output_tokens_per_call",
        role=role,
    )
    per_call_limit = configured_per_call_limit or DEFAULT_MAX_OUTPUT_TOKENS_BY_ROLE.get(
        role,
        DEFAULT_MAX_OUTPUT_TOKENS_PER_CALL,
    )
    remaining_total = (
        max(0, int(total_limit - float(snapshot.usage.get(OUTPUT_TOKENS) or 0)))
        if total_limit is not None
        else None
    )
    context_remaining = _context_output_capacity(
        capabilities.context_window_tokens,
        estimated_input_tokens,
    )
    candidates = [
        int(value)
        for value in (
            remaining_total,
            per_call_limit,
            capabilities.max_output_tokens,
            context_remaining,
        )
        if value is not None
    ]
    if not candidates:
        return None
    available = min(candidates)
    if available <= 0:
        decision = preview_budget(
            snapshot,
            BudgetRequest(
                run_id=snapshot.run_id,
                metric=OUTPUT_TOKENS,
                amount=1,
                limit=total_limit or per_call_limit,
            ),
        )
        raise BudgetExceededError(decision)
    return available


def _context_output_capacity(
    context_window_tokens: int | None,
    estimated_input_tokens: int,
) -> int | None:
    if context_window_tokens is None:
        return None
    context_window = max(1, int(context_window_tokens))
    estimated_input = max(0, int(estimated_input_tokens))
    unreserved = max(1, context_window - estimated_input)
    reserve = min(
        max(
            MIN_CONTEXT_RESERVE_TOKENS,
            int(context_window * CONTEXT_RESERVE_RATIO),
        ),
        max(0, unreserved - 1),
    )
    return max(1, unreserved - reserve)


def _reasoning_effort(
    capabilities: ModelCapabilities,
    role: str,
) -> Literal["low", "medium", "high"] | None:
    if capabilities.supports_reasoning_effort is not True:
        return None
    return "low" if role == "fast" else "high"


def _provider_usage(raw: dict[str, Any] | None) -> dict[str, Any]:
    value = raw if isinstance(raw, dict) else {}
    if isinstance(value.get("usage"), dict):
        return as_dict(value["usage"])
    nested = as_dict(value.get("raw"))
    return as_dict(nested.get("usage"))


def provider_messages_from_agent(
    messages: list[AgentMessage],
    *,
    system_prompt: str = "",
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if system_prompt:
        result.append({"role": "system", "content": system_prompt, "_context_tier": "L1"})
    for message in messages:
        content: str | list[dict[str, Any]] = message.content
        if message.content_parts:
            parts: list[dict[str, Any]] = []
            has_equivalent_text = any(
                part.type == "text" and part.text.strip() == message.content.strip()
                for part in message.content_parts
            )
            if message.content.strip() and not has_equivalent_text:
                parts.append({"type": "text", "text": message.content})
            for part in message.content_parts:
                if part.type == "text":
                    parts.append({"type": "text", "text": part.text})
                    continue
                image_url: dict[str, Any] = {"url": part.uri}
                if part.detail != "auto":
                    image_url["detail"] = part.detail
                parts.append({"type": "image_url", "image_url": image_url})
            content = parts
        payload: dict[str, Any] = {"role": message.role, "content": content}
        context_tier = str(message.metadata.get("context_tier") or "").strip().upper()
        if context_tier in {"L1", "L2", "L3"}:
            payload["_context_tier"] = context_tier
        if str(message.metadata.get("context_retention") or "").strip().lower() in {
            "pinned",
            "protected",
        }:
            payload["_context_protected"] = True
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        result.append(payload)
    return result


def _system_prompt_for_attempt(
    system_prompt: str,
    *,
    previous_failure: dict[str, Any],
    forced_tool_name: str = "",
) -> str:
    failure_category = previous_failure.get("failure_category")
    if failure_category not in {
        "tool_arguments_invalid",
        "tool_contract_violation",
    }:
        return system_prompt
    expected = str(forced_tool_name or "").strip()
    target = f" `{expected}`" if expected else ""
    if failure_category == "tool_contract_violation":
        instruction = (
            "The previous response violated the required tool contract because it did not "
            f"call the required tool{target}. Call that tool exactly once now. Do not answer "
            "with prose and do not call any other tool. Populate a complete arguments object "
            "that matches the supplied tool schema."
        )
    else:
        instruction = (
            "The previous native tool call contained invalid or truncated JSON arguments. "
            f"Retry the required tool call{target} now. Emit exactly one complete JSON object "
            "that matches the tool schema. Include every required field, close every string, "
            "array, and object, and do not add prose outside the tool call."
        )
    return f"{system_prompt.strip()}\n\n{instruction}".strip()
