from __future__ import annotations

import asyncio
import inspect
import json
from copy import deepcopy
from typing import Any, Callable, Literal, Protocol, runtime_checkable
from uuid import uuid4

from deepkeel.budget import (
    INPUT_TOKENS,
    MODEL_CALLS,
    MODEL_RETRIES,
    OUTPUT_TOKENS,
    BudgetPolicy,
    BudgetExceededError,
    BudgetLedger,
    BudgetRequest,
    preview_budget,
)
from deepkeel.context_window import ConservativeTokenEstimator
from deepkeel.context_compaction import (
    ContextInputBudgetError,
    ModelInputContextResult,
    prepare_model_input_context,
)
from deepkeel.context_contracts import ModelContextProfile
from deepkeel.contracts import AgentMessage, ToolCall
from deepkeel.deadlines import ensure_time_remaining, remaining_timeout_ceiling
from deepkeel.model_failures import (
    ModelToolArgumentsError,
    ModelToolContractError,
    classify_model_failure,
    provider_fingerprint,
)
from deepkeel.model_health import InMemoryModelHealthStore, ModelHealthStore
from deepkeel.model_invocations import (
    InMemoryModelInvocationRecorder,
    InMemoryModelInvocationStore,
    ModelInvocation,
    ModelInvocationClaim,
    ModelInvocationConflict,
    ModelInvocationEnvelope,
    ModelInvocationRecord,
    ModelInvocationRecorder,
    ModelInvocationStore,
    ModelInvocationUnavailable,
    ModelProviderInfo,
    ModelTurn,
)
from deepkeel.model_capabilities import (
    InMemoryModelCapabilityRegistry,
    ModelCapabilities,
    ResponseContract,
    ResponseFormat,
    StructuredOutputAttempt,
    negotiate_structured_output,
    response_format_not_supported,
    response_format_payload,
    structured_output_prompt,
)
from deepkeel.model_routing import (
    AdaptiveStepModelRouter,
    ModelRouteDecision,
    ModelRouter,
    ModelStepContext,
)
from deepkeel.model_step_execution import (
    ModelAttemptExecutionError,
    execute_model_attempt,
    record_failed_attempt_usage,
    record_successful_attempt_usage,
)
from deepkeel.model_gateway import RoutedModelGateway
from deepkeel.model_gateway_support import (
    CONTEXT_RESERVE_RATIO,
    DEFAULT_MAX_OUTPUT_TOKENS_BY_ROLE,
    DEFAULT_MAX_OUTPUT_TOKENS_PER_CALL,
    MIN_CONTEXT_RESERVE_TOKENS,
    MODEL_FAILURE_AUTO_FALLBACK,
    MODEL_FAILURE_FAIL_FAST,
    MODEL_FAILURE_RETRY_SELECTED,
    VALID_MODEL_FAILURE_POLICIES,
    _context_output_capacity,
    _model_call_limit,
    _model_failure_policy,
    _prepare_model_context,
    _provider_usage,
    _reasoning_effort,
    _remaining_output_tokens,
    _system_prompt_for_attempt,
    _tool_choice,
    _validate_forced_tool_turn,
    provider_messages_from_agent,
)
from deepkeel.model_native_provider import (
    NativeChatProviderAdapter,
    _assemble_streamed_turn,
    _call_supported,
    _content_text,
    _json_arguments,
    _messages_with_structured_contract,
    _repair_truncated_json_object,
    _tool_call_from_stream,
    _turn_from_completion,
)
from deepkeel.model_provider_execution import (
    _adapter_fingerprint,
    _ainvoke_provider,
    _as_provider_adapter,
)
from deepkeel.model_provider_contracts import (
    AsyncModelProviderAdapter,
    ModelGateway,
    ModelProviderAdapter,
    ModelRouteSink,
)
from deepkeel.policy import (
    PolicyDeniedError,
    PolicyEngine,
    PolicyRequest,
)
from deepkeel.tool_registry import ToolRegistry, ToolSpec
from deepkeel.type_narrowing import as_dict, as_list


def model_tools_from_registry(
    registry: ToolRegistry,
    allowed_names: set[str] | None = None,
    parameter_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    result = []
    for spec in registry.list_tools():
        if allowed_names is not None and spec.name not in allowed_names:
            continue
        override = (parameter_overrides or {}).get(spec.name)
        parameters = (
            deepcopy(override) if isinstance(override, dict) else _model_parameters_schema(spec)
        )
        result.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": _model_tool_description(spec),
                    "parameters": parameters,
                },
            }
        )
    return result


def _model_parameters_schema(spec) -> dict[str, Any]:
    formal_schema = getattr(spec, "formal_parameters_schema", None)
    schema = formal_schema() if callable(formal_schema) else {}
    if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
        return deepcopy(schema)
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _model_tool_description(spec: ToolSpec) -> str:
    parts = [spec.description.strip()]
    policy = spec.usage_policy if isinstance(spec.usage_policy, dict) else {}
    if policy.get("when_to_use"):
        parts.append(f"Use when: {policy['when_to_use']}")
    if policy.get("when_not_to_use"):
        parts.append(f"Do not use when: {policy['when_not_to_use']}")
    return "\n".join(part for part in parts if part)
