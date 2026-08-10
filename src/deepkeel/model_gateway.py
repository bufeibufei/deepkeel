from __future__ import annotations

import asyncio
import inspect
import json
from copy import deepcopy
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
        if step_context is None:
            step_context = ModelStepContext(
                run_id="local-run",
                user_id="local-user",
                thread_id="local-thread",
                turn_id="local-turn",
                step_index=0,
                message_count=len(messages),
                observation_count=0,
                tool_result_count=0,
                available_roles=tuple(self.providers),
            )
        else:
            step_context = ModelStepContext(
                run_id=step_context.run_id,
                user_id=step_context.user_id,
                thread_id=step_context.thread_id,
                turn_id=step_context.turn_id,
                step_index=step_context.step_index,
                message_count=step_context.message_count,
                observation_count=step_context.observation_count,
                tool_result_count=step_context.tool_result_count,
                available_roles=tuple(self.providers),
                observation_sources=step_context.observation_sources,
                tool_result_names=step_context.tool_result_names,
                model_policy=step_context.model_policy,
                skill_activation=step_context.skill_activation,
                policy_phase=step_context.policy_phase,
                forced_tool_name=step_context.forced_tool_name,
                governance_scope=step_context.governance_scope,
                deadline_monotonic=step_context.deadline_monotonic,
                operational_run_id=step_context.operational_run_id,
            )
        requires_image_input = any(
            any(part.type == "image" for part in message.content_parts) for message in messages
        )
        primary_route = self.router.route(step_context)
        failure_policy = _model_failure_policy(step_context.model_policy)
        attempts = self._attempts(
            primary_route,
            failure_policy=failure_policy,
            requires_native_tools=bool(tools or step_context.forced_tool_name),
        )
        if requires_image_input:
            attempts = [
                attempt
                for attempt in attempts
                if attempt[1].info.capabilities.supports_image_input is True
            ]
            if not attempts:
                raise RuntimeError("no configured model provider declares image input support")
        previous_failure: dict[str, Any] = {}
        token_estimator = ConservativeTokenEstimator()
        budget_policy = BudgetPolicy.from_mapping(
            step_context.model_policy.get("budget")
            if isinstance(step_context.model_policy.get("budget"), dict)
            else {}
        )

        for attempt_index, (route, provider, retry_kind) in enumerate(attempts, start=1):
            health_provider_id = _health_provider_id(provider.info.provider_id, step_context)
            health = await run_sync_adapter(
                self.model_health_store.snapshot,
                health_provider_id,
                provider.info.model_id,
            )
            if not health.is_available():
                route_payload = {
                    **route.as_dict(),
                    "model_id": provider.info.model_id,
                    "provider_id": provider.info.provider_id,
                    "attempt_index": attempt_index,
                    "retry_kind": retry_kind,
                    "health": health.as_dict(),
                    "skipped": "model_circuit_open",
                }
                previous_failure = {
                    "fallback_from": route.role,
                    "failure_category": "model_circuit_open",
                }
                if on_route is not None:
                    on_route(route_payload)
                continue
            request_timeout = remaining_timeout_ceiling(
                step_context.deadline_monotonic,
                maximum=self.request_timeout,
            )
            role_request_limit = budget_policy.limit(
                "max_request_seconds",
                role=route.role,
            )
            if role_request_limit is not None:
                request_timeout = min(
                    request_timeout,
                    max(1, int(role_request_limit)),
                )
            policy = self.policy_engine.evaluate(
                PolicyRequest(
                    action="model.invoke",
                    resource_type="model",
                    resource_id=route.role,
                    run_id=step_context.run_id,
                    user_id=step_context.user_id,
                    tenant_id=str(step_context.governance_scope.get("tenant_id") or ""),
                    context={
                        "skill_activation": step_context.skill_activation,
                        "model_policy": step_context.model_policy,
                        "route": route.as_dict(),
                        "attempt_index": attempt_index,
                        "governance_scope": step_context.governance_scope,
                        "runtime_policy": (
                            step_context.model_policy.get("runtime_policy")
                            if isinstance(step_context.model_policy.get("runtime_policy"), dict)
                            else {}
                        ),
                    },
                )
            )
            budget = (
                await run_sync_adapter(
                    self.budget_ledger.consume,
                    BudgetRequest(
                        run_id=step_context.accounting_run_id,
                        metric=MODEL_CALLS,
                        limit=_model_call_limit(step_context.model_policy),
                        operation_id=(
                            f"model-step:{step_context.step_index}:attempt:{attempt_index}"
                        ),
                        metadata={
                            "role": route.role,
                            "step_index": step_context.step_index,
                            "attempt_index": attempt_index,
                        },
                    ),
                )
                if policy.allowed
                else None
            )
            attempt_system_prompt = _system_prompt_for_attempt(
                system_prompt,
                previous_failure=previous_failure,
                forced_tool_name=step_context.forced_tool_name,
            )
            provider_messages = provider_messages_from_agent(
                messages,
                system_prompt=attempt_system_prompt,
            )
            provider_capabilities = getattr(
                provider.info,
                "capabilities",
                ModelCapabilities(source="adapter_unknown"),
            )
            per_call_input_limit = budget_policy.limit(
                "max_input_tokens_per_call",
                role=route.role,
            )
            budget_snapshot = await run_sync_adapter(
                self.budget_ledger.snapshot,
                step_context.accounting_run_id,
            )
            prepared_model_context = _prepare_model_context(
                provider_messages,
                tools,
                provider=provider,
                route=route,
                provider_capabilities=provider_capabilities,
                budget_policy=budget_policy,
                budget_snapshot=budget_snapshot,
                step_context=step_context,
                per_call_input_limit=per_call_input_limit,
                token_estimator=token_estimator,
                context_compactor=self.context_compactor,
            )
            provider_messages = prepared_model_context.messages
            estimated_input_tokens = token_estimator.estimate(
                {"messages": provider_messages, "tools": tools}
            )
            if per_call_input_limit is not None and estimated_input_tokens > per_call_input_limit:
                raise BudgetExceededError(
                    preview_budget(
                        budget_snapshot,
                        BudgetRequest(
                            run_id=step_context.accounting_run_id,
                            metric=INPUT_TOKENS,
                            amount=estimated_input_tokens,
                            limit=per_call_input_limit,
                        ),
                    )
                )
            input_budget = (
                preview_budget(
                    budget_snapshot,
                    BudgetRequest(
                        run_id=step_context.accounting_run_id,
                        metric=INPUT_TOKENS,
                        amount=estimated_input_tokens,
                        limit=budget_policy.limit("max_input_tokens_total"),
                        metadata={"role": route.role},
                    ),
                )
                if policy.allowed
                else None
            )
            retry_budget = (
                await run_sync_adapter(
                    self.budget_ledger.consume,
                    BudgetRequest(
                        run_id=step_context.accounting_run_id,
                        metric=MODEL_RETRIES,
                        limit=budget_policy.limit("max_model_retries"),
                        operation_id=(
                            f"model-retry:{step_context.step_index}:attempt:{attempt_index}"
                        ),
                        metadata={"role": route.role, "retry_kind": retry_kind},
                    ),
                )
                if policy.allowed and attempt_index > 1
                else None
            )
            max_output_tokens = _remaining_output_tokens(
                budget_policy,
                budget_snapshot,
                route.role,
                capabilities=provider_capabilities,
                estimated_input_tokens=estimated_input_tokens,
            )
            reasoning_effort = _reasoning_effort(
                provider_capabilities,
                route.role,
            )
            route_payload = {
                **route.as_dict(),
                "model_id": provider.info.model_id,
                "provider_id": provider.info.provider_id,
                "requested_role": primary_route.role,
                "preferred_model_id": self.providers[primary_route.role].info.model_id,
                "actual_model_id": provider.info.model_id,
                "failure_policy": failure_policy,
                "fallback_allowed": failure_policy == MODEL_FAILURE_AUTO_FALLBACK,
                "policy": policy.as_dict(),
                "budget": budget.as_dict() if budget is not None else {},
                "budget_metrics": {
                    INPUT_TOKENS: input_budget.as_dict() if input_budget is not None else {},
                    MODEL_RETRIES: retry_budget.as_dict() if retry_budget is not None else {},
                },
                "forced_tool_name": step_context.forced_tool_name,
                "attempt_index": attempt_index,
                "retry_kind": retry_kind,
                "max_output_tokens": max_output_tokens,
                "reasoning_effort": reasoning_effort,
                "model_limits": {
                    "context_window_tokens": provider_capabilities.context_window_tokens,
                    "max_output_tokens": provider_capabilities.max_output_tokens,
                    "capability_source": provider_capabilities.source,
                },
                "context_manifest": prepared_model_context.diagnostics,
                "required_capabilities": {
                    "native_tools": bool(tools or step_context.forced_tool_name),
                    "forced_tool_choice": bool(step_context.forced_tool_name),
                    "streaming": on_text_delta is not None,
                    "image_input": requires_image_input,
                },
                "provider_capabilities": provider_capabilities.model_dump(mode="json"),
                "repair_strategy": (
                    "native_tool_arguments_json"
                    if previous_failure.get("failure_category") == "tool_arguments_invalid"
                    else "forced_tool_contract"
                    if previous_failure.get("failure_category") == "tool_contract_violation"
                    else ""
                ),
                **previous_failure,
            }
            if not policy.allowed:
                if on_route is not None:
                    on_route(route_payload)
                raise PolicyDeniedError(policy)
            if budget is not None and not budget.allowed:
                if on_route is not None:
                    on_route(route_payload)
                raise BudgetExceededError(budget)
            if input_budget is not None and not input_budget.allowed:
                if on_route is not None:
                    on_route(route_payload)
                raise BudgetExceededError(input_budget)
            if retry_budget is not None and not retry_budget.allowed:
                if on_route is not None:
                    on_route(route_payload)
                raise BudgetExceededError(retry_budget)

            try:
                outcome = await execute_model_attempt(
                    provider=provider,
                    route=route,
                    step_context=step_context,
                    attempt_index=attempt_index,
                    retry_kind=retry_kind,
                    provider_messages=provider_messages,
                    tools=tools,
                    tool_choice=_tool_choice(step_context, provider_capabilities),
                    request_timeout=request_timeout,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                    estimated_input_tokens=estimated_input_tokens,
                    route_payload=route_payload,
                    token_estimator=token_estimator,
                    invocation_recorder=self.invocation_recorder,
                    invocation_store=self.invocation_store,
                    model_health_store=self.model_health_store,
                    invoke_provider=_ainvoke_provider,
                    validate_turn=_validate_forced_tool_turn,
                    on_text_delta=on_text_delta,
                )
                turn = outcome.turn
            except ModelAttemptExecutionError as attempt_error:
                exc = attempt_error.cause
                failed_input_budget, failed_output_budget = await run_sync_adapter(
                    record_failed_attempt_usage,
                    ledger=self.budget_ledger,
                    policy=budget_policy,
                    step_context=step_context,
                    role=route.role,
                    attempt_index=attempt_index,
                    estimated_input_tokens=estimated_input_tokens,
                    streamed_output=attempt_error.streamed_output,
                    token_estimator=token_estimator,
                    route_payload=route_payload,
                )
                if not failed_input_budget.allowed:
                    if on_route is not None:
                        on_route(route_payload)
                    raise BudgetExceededError(failed_input_budget) from exc
                if failed_output_budget is not None and not failed_output_budget.allowed:
                    if on_route is not None:
                        on_route(route_payload)
                    raise BudgetExceededError(failed_output_budget) from exc
                failure = classify_model_failure(exc)
                if failure.retryable and failure.degrades_provider_health:
                    failed_health = await run_sync_adapter(
                        self.model_health_store.record_failure,
                        health_provider_id,
                        provider.info.model_id,
                        category=failure.category,
                        immediate=failure.category == "rate_limited",
                        retry_after_seconds=failure.retry_after_seconds,
                    )
                    route_payload["health"] = failed_health.as_dict()
                can_retry = (
                    failure.retryable
                    and not attempt_error.visible_delta_emitted
                    and attempt_index < len(attempts)
                )
                if not can_retry:
                    if on_route is not None:
                        on_route(route_payload)
                    raise exc
                previous_failure = {
                    "fallback_from": route.role,
                    "failure_category": failure.category,
                    "failure_status_code": failure.status_code,
                }
                if on_route is not None:
                    on_route(route_payload)
                continue
            settled_input_budget, output_budget = await run_sync_adapter(
                record_successful_attempt_usage,
                ledger=self.budget_ledger,
                policy=budget_policy,
                step_context=step_context,
                role=route.role,
                attempt_index=attempt_index,
                estimated_input_tokens=estimated_input_tokens,
                turn=turn,
                provider_usage=_provider_usage(turn.raw),
                token_estimator=token_estimator,
                route_payload=route_payload,
            )
            if on_route is not None:
                on_route(route_payload)
            if not settled_input_budget.allowed:
                raise BudgetExceededError(settled_input_budget)
            if not output_budget.allowed:
                raise BudgetExceededError(output_budget)
            return turn.model_copy(
                update={
                    "model_role": route.role,
                    "raw": {**turn.raw, "harness_route": route_payload},
                }
            )

        raise RuntimeError("model invocation attempts exhausted")

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
