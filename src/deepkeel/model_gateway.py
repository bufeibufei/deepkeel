from __future__ import annotations

import asyncio
import inspect
import json
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Literal, Protocol, runtime_checkable
from uuid import uuid4

from deepkeel.async_ports import run_sync_adapter
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
    WorkingContextCompactor,
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
    scoped_model_health_provider_id as _health_provider_id,
)
from deepkeel.policy import (
    PolicyDeniedError,
    PolicyEngine,
    PolicyRequest,
)
from deepkeel.tool_registry import ToolRegistry, ToolSpec
from deepkeel.type_narrowing import as_dict, as_list
from deepkeel.model_gateway_support import (
    MODEL_FAILURE_AUTO_FALLBACK,
    MODEL_FAILURE_FAIL_FAST,
    MODEL_FAILURE_RETRY_SELECTED,
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
from deepkeel.model_provider_contracts import (
    AsyncModelProviderAdapter,
    ModelProviderAdapter,
    ModelRouteSink,
)
from deepkeel.model_provider_execution import (
    _adapter_fingerprint,
    _ainvoke_provider,
    _as_provider_adapter,
)
from deepkeel.model_turn_execution import RoutedModelTurnExecution


class RoutedModelGateway:
    """Routes and governs every model step before invoking a provider adapter."""

    def __init__(
        self,
        providers: dict[str, Any],
        *,
        router: ModelRouter | None,
        policy_engine: PolicyEngine,
        budget_ledger: BudgetLedger,
        invocation_recorder: ModelInvocationRecorder | None = None,
        invocation_store: ModelInvocationStore | None = None,
        model_health_store: ModelHealthStore | None = None,
        request_timeout: int = 300,
        context_compactor: WorkingContextCompactor | None = None,
    ) -> None:
        self.providers = {
            str(role): _as_provider_adapter(provider)
            for role, provider in providers.items()
            if provider is not None
        }
        self.router = router or AdaptiveStepModelRouter()
        self.policy_engine = policy_engine
        self.budget_ledger = budget_ledger
        self.invocation_recorder = invocation_recorder
        self.invocation_store = invocation_store
        self.model_health_store = model_health_store or InMemoryModelHealthStore()
        self.request_timeout = request_timeout
        self.context_compactor = context_compactor

    def run_turn(
        self,
        messages: list[AgentMessage],
        *,
        tools: list[dict[str, Any]],
        system_prompt: str = "",
        on_text_delta: Callable[[str], None] | None = None,
        step_context: ModelStepContext | None = None,
        on_route: ModelRouteSink | None = None,
    ) -> ModelTurn:
        """Compatibility entrypoint for synchronous graph hosts."""

        return asyncio.run(
            self.arun_turn(
                messages,
                tools=tools,
                system_prompt=system_prompt,
                on_text_delta=on_text_delta,
                step_context=step_context,
                on_route=on_route,
            )
        )

    async def arun_turn(
        self,
        messages: list[AgentMessage],
        *,
        tools: list[dict[str, Any]],
        system_prompt: str = "",
        on_text_delta: Callable[[str], None] | None = None,
        step_context: ModelStepContext | None = None,
        on_route: ModelRouteSink | None = None,
    ) -> ModelTurn:
        step_context = _normalized_step_context(
            step_context,
            message_count=len(messages),
            available_roles=tuple(self.providers),
        )
        return await RoutedModelTurnExecution(
            self,
            messages,
            tools=tools,
            system_prompt=system_prompt,
            on_text_delta=on_text_delta,
            step_context=step_context,
            on_route=on_route,
        ).run()

    async def _routing_context(
        self,
        context: ModelStepContext,
        *,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]],
        requires_image_input: bool,
        token_estimator: ConservativeTokenEstimator,
    ) -> ModelStepContext:
        configured_profiles = as_dict(context.model_policy.get("role_profiles"))
        profiles: dict[str, dict[str, Any]] = {}
        for role, provider in self.providers.items():
            health_provider_id = _health_provider_id(provider.info.provider_id, context)
            health = await run_sync_adapter(
                self.model_health_store.snapshot,
                health_provider_id,
                provider.info.model_id,
            )
            configured = as_dict(configured_profiles.get(role))
            declared_capabilities = getattr(
                provider.info,
                "capabilities",
                ModelCapabilities(source="adapter_unknown"),
            )
            capabilities = declared_capabilities.model_dump(mode="json")
            capabilities["supports_native_tools"] = bool(
                getattr(provider.info, "supports_native_tools", True)
            )
            capabilities["supports_streaming"] = bool(
                getattr(provider.info, "supports_streaming", True)
            )
            provider_metadata = as_dict(getattr(provider.info, "metadata", {}))
            profiles[role] = {
                "provider_id": provider.info.provider_id,
                "model_id": provider.info.model_id,
                "health": health.as_dict(),
                "capabilities": capabilities,
                "latency_tier": str(
                    configured.get("latency_tier")
                    or provider_metadata.get("latency_tier")
                    or "unknown"
                ),
                "cost_tier": str(
                    configured.get("cost_tier")
                    or provider_metadata.get("cost_tier")
                    or "unknown"
                ),
            }
        budget = await run_sync_adapter(
            self.budget_ledger.snapshot,
            context.accounting_run_id,
        )
        return replace(
            context,
            estimated_input_tokens=token_estimator.estimate(
                {
                    "messages": [message.model_dump(mode="json") for message in messages],
                    "tools": tools,
                }
            ),
            requires_native_tools=bool(tools or context.forced_tool_name),
            requires_image_input=requires_image_input,
            provider_profiles=profiles,
            budget_snapshot=budget.as_dict(),
        )

    def _attempts(
        self,
        primary_route: ModelRouteDecision,
        *,
        failure_policy: str = MODEL_FAILURE_AUTO_FALLBACK,
        requires_native_tools: bool = False,
    ) -> list[
        tuple[
            ModelRouteDecision,
            ModelProviderAdapter | AsyncModelProviderAdapter,
            str,
        ]
    ]:
        primary = self.providers.get(primary_route.role)
        if primary is None:
            raise RuntimeError(f"model provider for role {primary_route.role!r} is unavailable")
        result = [(primary_route, primary, "primary")]
        if failure_policy == MODEL_FAILURE_FAIL_FAST:
            return result
        if failure_policy == MODEL_FAILURE_RETRY_SELECTED:
            return [
                *result,
                (
                    ModelRouteDecision(
                        role=primary_route.role,
                        reason="retry selected model",
                        router_id=primary_route.router_id,
                        metadata=dict(primary_route.metadata),
                    ),
                    primary,
                    "retry",
                ),
            ]
        primary_fingerprint = _adapter_fingerprint(primary)
        preferred_roles = (
            ("reasoning", "fast") if primary_route.role == "fast" else ("fast", "reasoning")
        )
        ordered_roles = tuple(dict.fromkeys((*preferred_roles, *self.providers)))
        for role in ordered_roles:
            provider = self.providers.get(role)
            if provider is None or _adapter_fingerprint(provider) == primary_fingerprint:
                continue
            if requires_native_tools and not provider.info.supports_native_tools:
                continue
            result.append(
                (
                    ModelRouteDecision(
                        role=role,
                        reason="transient provider fallback",
                        router_id=primary_route.router_id,
                        metadata={
                            **primary_route.metadata,
                            "fallback_from": primary_route.role,
                        },
                    ),
                    provider,
                    "fallback",
                )
            )
            return result
        result.append(
            (
                ModelRouteDecision(
                    role=primary_route.role,
                    reason="transient provider retry",
                    router_id=primary_route.router_id,
                    metadata=dict(primary_route.metadata),
                ),
                primary,
                "retry",
            )
        )
        return result


def _normalized_step_context(
    context: ModelStepContext | None,
    *,
    message_count: int,
    available_roles: tuple[str, ...],
) -> ModelStepContext:
    if context is None:
        return ModelStepContext(
            run_id="local-run",
            user_id="local-user",
            thread_id="local-thread",
            turn_id="local-turn",
            step_index=0,
            message_count=message_count,
            observation_count=0,
            tool_result_count=0,
            available_roles=available_roles,
        )
    return replace(context, available_roles=available_roles)
