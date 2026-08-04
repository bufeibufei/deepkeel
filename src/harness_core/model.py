from __future__ import annotations

import asyncio
import inspect
import json
from copy import deepcopy
from typing import Any, Callable, Literal, Protocol, runtime_checkable
from uuid import uuid4

from harness_core.budget import (
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
from harness_core.context_window import ConservativeTokenEstimator
from harness_core.context_compaction import prepare_model_input_context
from harness_core.context_contracts import ModelContextProfile
from harness_core.contracts import AgentMessage, ToolCall
from harness_core.deadlines import ensure_time_remaining, remaining_timeout_ceiling
from harness_core.model_failures import (
    ModelToolArgumentsError,
    ModelToolContractError,
    classify_model_failure,
    provider_fingerprint,
)
from harness_core.model_health import InMemoryModelHealthStore, ModelHealthStore
from harness_core.model_invocations import (
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
from harness_core.model_capabilities import (
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
from harness_core.model_routing import (
    AdaptiveStepModelRouter,
    ModelRouteDecision,
    ModelRouter,
    ModelStepContext,
)
from harness_core.model_step_execution import (
    ModelAttemptExecutionError,
    execute_model_attempt,
    record_failed_attempt_usage,
    record_successful_attempt_usage,
)
from harness_core.policy import (
    PolicyDeniedError,
    PolicyEngine,
    PolicyRequest,
)
from harness_core.tool_registry import ToolRegistry, ToolSpec
from harness_core.type_narrowing import as_dict, as_list


ModelRouteSink = Callable[[dict[str, Any]], None]
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


@runtime_checkable
class ModelProviderAdapter(Protocol):
    """Explicit provider boundary used by routed model execution."""

    @property
    def info(self) -> ModelProviderInfo: ...

    def invoke(
        self,
        request: ModelInvocation,
        *,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ModelTurn: ...


@runtime_checkable
class AsyncModelProviderAdapter(Protocol):
    """Native async provider boundary used without a worker-thread bridge."""

    @property
    def info(self) -> ModelProviderInfo: ...

    async def ainvoke(
        self,
        request: ModelInvocation,
        *,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ModelTurn: ...


class ModelGateway(Protocol):
    """Provider-neutral model execution port used by the graph."""

    def run_turn(
        self,
        messages: list[AgentMessage],
        *,
        tools: list[dict[str, Any]],
        system_prompt: str = "",
        on_text_delta: Callable[[str], None] | None = None,
        step_context: ModelStepContext | None = None,
        on_route: ModelRouteSink | None = None,
    ) -> ModelTurn: ...

    async def arun_turn(
        self,
        messages: list[AgentMessage],
        *,
        tools: list[dict[str, Any]],
        system_prompt: str = "",
        on_text_delta: Callable[[str], None] | None = None,
        step_context: ModelStepContext | None = None,
        on_route: ModelRouteSink | None = None,
    ) -> ModelTurn: ...


class NativeChatProviderAdapter:
    """Adapter for providers exposing stream_chat or complete_chat."""

    def __init__(
        self,
        provider,
        *,
        request_timeout: int = 300,
        model_capabilities: InMemoryModelCapabilityRegistry | None = None,
    ):
        self.provider = provider
        self.request_timeout = request_timeout
        self.model_capabilities = model_capabilities or InMemoryModelCapabilityRegistry()

    @property
    def info(self) -> ModelProviderInfo:
        fingerprint = provider_fingerprint(self.provider)
        supports_streaming = callable(getattr(self.provider, "stream_chat", None))
        supports_completion = callable(getattr(self.provider, "complete_chat", None))
        return ModelProviderInfo(
            provider_id=(
                ":".join(part for part in fingerprint if part)
                or type(self.provider).__name__
            ),
            model_id=str(getattr(self.provider, "model", "") or ""),
            model_role=str(getattr(self.provider, "model_role", "") or "reasoning"),
            supports_streaming=supports_streaming,
            supports_native_tools=supports_streaming or supports_completion,
            capabilities=self.model_capabilities.capabilities_for(self.provider),
        )

    def invoke(
        self,
        request: ModelInvocation,
        *,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ModelTurn:
        if request.response_contract is not None and callable(
            getattr(self.provider, "complete_chat", None)
        ):
            return self._invoke_structured(
                request,
                on_text_delta=on_text_delta,
            )
        if callable(getattr(self.provider, "stream_chat", None)):
            events = self._call_stream_chat(
                request.messages,
                request.tools,
                tool_choice=request.tool_choice,
                request_timeout=request.request_timeout,
                max_output_tokens=request.max_output_tokens,
                reasoning_effort=request.reasoning_effort,
            )
            return _assemble_streamed_turn(
                events,
                provider=self.provider,
                on_text_delta=on_text_delta,
            )
        if callable(getattr(self.provider, "complete_chat", None)):
            response = self._call_complete_chat(
                request.messages,
                request.tools,
                tool_choice=request.tool_choice,
                request_timeout=request.request_timeout,
                max_output_tokens=request.max_output_tokens,
                reasoning_effort=request.reasoning_effort,
            )
            return _turn_from_completion(
                response,
                self.provider,
                on_text_delta=on_text_delta,
            )
        raise RuntimeError("provider does not support native tool calls")

    def _invoke_structured(
        self,
        request: ModelInvocation,
        *,
        on_text_delta: Callable[[str], None] | None,
    ) -> ModelTurn:
        contract = request.response_contract
        if contract is None:
            raise RuntimeError("structured model invocation requires a response contract")
        decision = negotiate_structured_output(
            self.model_capabilities.capabilities_for(self.provider),
            contract,
        )
        attempts: list[StructuredOutputAttempt] = []
        for mode in decision.candidate_formats:
            messages = _messages_with_structured_contract(request.messages, contract, mode)
            try:
                response = self._call_complete_chat(
                    messages,
                    request.tools,
                    tool_choice=request.tool_choice,
                    request_timeout=request.request_timeout,
                    max_output_tokens=request.max_output_tokens,
                    reasoning_effort=request.reasoning_effort,
                    response_format=response_format_payload(mode, contract),
                )
            except Exception as exc:
                if not response_format_not_supported(exc):
                    raise
                self.model_capabilities.mark_response_format(
                    self.provider,
                    mode,
                    supported=False,
                )
                attempts.append(
                    StructuredOutputAttempt(
                        response_format=mode,
                        outcome="unsupported",
                        detail=str(exc)[:300],
                    )
                )
                continue
            self.model_capabilities.mark_response_format(
                self.provider,
                mode,
                supported=True,
            )
            attempts.append(StructuredOutputAttempt(response_format=mode, outcome="completed"))
            turn = _turn_from_completion(
                response,
                self.provider,
                on_text_delta=on_text_delta,
            )
            diagnostics = {
                "requested_format": decision.requested_format.value,
                "effective_format": mode.value,
                "capability_source": self.info.capabilities.source,
                "degraded": mode != decision.requested_format,
                "attempts": [attempt.model_dump(mode="json") for attempt in attempts],
            }
            return turn.model_copy(
                update={"raw": {**turn.raw, "structured_output": diagnostics}}
            )
        raise RuntimeError("model provider has no supported structured output format")

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
        deadline_monotonic = (
            step_context.deadline_monotonic if step_context is not None else None
        )
        request_timeout = remaining_timeout_ceiling(
            deadline_monotonic,
            maximum=self.request_timeout,
        )
        ensure_time_remaining(deadline_monotonic)

        def checked_delta(delta: str) -> None:
            ensure_time_remaining(deadline_monotonic)
            if on_text_delta is not None:
                on_text_delta(delta)

        if on_route is not None:
            on_route(
                {
                    "role": self.info.model_role,
                    "model_id": self.info.model_id,
                    "reason": "static provider",
                    "router_id": "static-provider",
                    "step_index": step_context.step_index if step_context is not None else 0,
                }
            )
        provider_messages = provider_messages_from_agent(messages, system_prompt=system_prompt)
        tool_choice = _tool_choice(step_context, self.info.capabilities)
        forced_tool_name = str(
            step_context.forced_tool_name if step_context is not None else ""
        ).strip()
        turn = self.invoke(
            ModelInvocation(
                messages=provider_messages,
                tools=tools,
                tool_choice=tool_choice,
                request_timeout=request_timeout,
            ),
            on_text_delta=None if forced_tool_name else checked_delta,
        )
        _validate_forced_tool_turn(turn, step_context)
        ensure_time_remaining(deadline_monotonic)
        return turn

    def _call_stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any],
        request_timeout: int,
        max_output_tokens: int | None,
        reasoning_effort: str | None,
    ):
        return _call_supported(
            self.provider.stream_chat,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            request_timeout=request_timeout,
            max_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )

    def _call_complete_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any],
        request_timeout: int,
        max_output_tokens: int | None,
        reasoning_effort: str | None,
        response_format: dict[str, Any] | None = None,
    ):
        return _call_supported(
            self.provider.complete_chat,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            request_timeout=request_timeout,
            max_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
        )


def _messages_with_structured_contract(
    messages: list[dict[str, Any]],
    contract: ResponseContract,
    response_format: ResponseFormat,
) -> list[dict[str, Any]]:
    copied = deepcopy(messages)
    if response_format == ResponseFormat.JSON_SCHEMA:
        return copied
    instruction = structured_output_prompt("", contract, response_format).strip()
    if copied and str(copied[0].get("role") or "") == "system":
        copied[0] = {
            **copied[0],
            "content": f"{copied[0].get('content') or ''}\n{instruction}".strip(),
        }
    else:
        copied.insert(0, {"role": "system", "content": instruction})
    return copied


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
            )
        requires_image_input = any(
            any(part.type == "image" for part in message.content_parts)
            for message in messages
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
                raise RuntimeError(
                    "no configured model provider declares image input support"
                )
        previous_failure: dict[str, Any] = {}
        token_estimator = ConservativeTokenEstimator()
        budget_policy = BudgetPolicy.from_mapping(
            step_context.model_policy.get("budget")
            if isinstance(step_context.model_policy.get("budget"), dict)
            else {}
        )

        for attempt_index, (route, provider, retry_kind) in enumerate(attempts, start=1):
            health = self.model_health_store.snapshot(
                provider.info.provider_id,
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
            budget = self.budget_ledger.consume(
                BudgetRequest(
                    run_id=step_context.run_id,
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
                )
            ) if policy.allowed else None
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
            prepared_model_context = prepare_model_input_context(
                provider_messages,
                tools,
                profile=ModelContextProfile(
                    model_id=provider.info.model_id,
                    model_role=route.role,
                    context_window_tokens=provider_capabilities.context_window_tokens,
                    max_output_tokens=_remaining_output_tokens(
                        budget_policy, self.budget_ledger.snapshot(step_context.run_id),
                        route.role,
                        capabilities=provider_capabilities,
                        estimated_input_tokens=0,
                    ),
                    source=provider_capabilities.source,
                ),
                configured_input_limit=(
                    int(per_call_input_limit)
                    if per_call_input_limit is not None
                    else None
                ),
                estimator=token_estimator,
                thread_id=step_context.thread_id,
                subject_id=str(step_context.governance_scope.get("subject_id") or ""),
            )
            provider_messages = prepared_model_context.messages
            estimated_input_tokens = token_estimator.estimate(
                {"messages": provider_messages, "tools": tools}
            )
            if per_call_input_limit is not None and estimated_input_tokens > per_call_input_limit:
                raise BudgetExceededError(
                    preview_budget(
                        self.budget_ledger.snapshot(step_context.run_id),
                        BudgetRequest(
                            run_id=step_context.run_id,
                            metric=INPUT_TOKENS,
                            amount=estimated_input_tokens,
                            limit=per_call_input_limit,
                        ),
                    )
                )
            input_budget = (
                preview_budget(
                    self.budget_ledger.snapshot(step_context.run_id),
                    BudgetRequest(
                        run_id=step_context.run_id,
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
                self.budget_ledger.consume(
                    BudgetRequest(
                        run_id=step_context.run_id,
                        metric=MODEL_RETRIES,
                        limit=budget_policy.limit("max_model_retries"),
                        operation_id=(
                            f"model-retry:{step_context.step_index}:attempt:{attempt_index}"
                        ),
                        metadata={"role": route.role, "retry_kind": retry_kind},
                    )
                )
                if policy.allowed and attempt_index > 1
                else None
            )
            max_output_tokens = _remaining_output_tokens(
                budget_policy,
                self.budget_ledger.snapshot(step_context.run_id),
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
                    if previous_failure.get("failure_category")
                    == "tool_arguments_invalid"
                    else "forced_tool_contract"
                    if previous_failure.get("failure_category")
                    == "tool_contract_violation"
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
                failed_input_budget, failed_output_budget = record_failed_attempt_usage(
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
                    route_payload["health"] = self.model_health_store.record_failure(
                        provider.info.provider_id,
                        provider.info.model_id,
                        category=failure.category,
                        immediate=failure.category == "rate_limited",
                        retry_after_seconds=failure.retry_after_seconds,
                    ).as_dict()
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
            settled_input_budget, output_budget = record_successful_attempt_usage(
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
            raise RuntimeError(
                f"model provider for role {primary_route.role!r} is unavailable"
            )
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
            ("reasoning", "fast")
            if primary_route.role == "fast"
            else ("fast", "reasoning")
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


def _model_failure_policy(model_policy: dict[str, Any]) -> str:
    value = str(
        model_policy.get("failure_policy") or MODEL_FAILURE_AUTO_FALLBACK
    ).strip().lower()
    return (
        value
        if value in VALID_MODEL_FAILURE_POLICIES
        else MODEL_FAILURE_AUTO_FALLBACK
    )


def _as_provider_adapter(provider: Any) -> ModelProviderAdapter | AsyncModelProviderAdapter:
    return (
        provider
        if isinstance(provider, (ModelProviderAdapter, AsyncModelProviderAdapter))
        else NativeChatProviderAdapter(provider)
    )


def _adapter_fingerprint(
    provider: ModelProviderAdapter | AsyncModelProviderAdapter,
) -> tuple[str, str]:
    info = provider.info
    return (info.provider_id, info.model_id)


async def _ainvoke_provider(
    provider: ModelProviderAdapter | AsyncModelProviderAdapter,
    request: ModelInvocation,
    *,
    on_text_delta: Callable[[str], None] | None = None,
) -> ModelTurn:
    if isinstance(provider, AsyncModelProviderAdapter):
        return await asyncio.wait_for(
            provider.ainvoke(request, on_text_delta=on_text_delta),
            timeout=max(1, request.request_timeout),
        )
    return await asyncio.wait_for(
        asyncio.to_thread(
            provider.invoke,
            request,
            on_text_delta=on_text_delta,
        ),
        timeout=max(1, request.request_timeout),
    )


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
    expected = str(
        step_context.forced_tool_name if step_context is not None else ""
    ).strip()
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
        result.append(
            {"role": "system", "content": system_prompt, "_context_tier": "L1"}
        )
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
        parameters = deepcopy(override) if isinstance(override, dict) else _model_parameters_schema(spec)
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


def _assemble_streamed_turn(
    events,
    *,
    provider,
    on_text_delta: Callable[[str], None] | None,
) -> ModelTurn:
    content_parts: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    finish_reason = ""
    last_event: dict[str, Any] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        last_event = event
        choices = event.get("choices") if isinstance(event.get("choices"), list) else []
        if not choices or not isinstance(choices[0], dict):
            continue
        choice = choices[0]
        finish_reason = str(choice.get("finish_reason") or finish_reason)
        delta = as_dict(choice.get("delta"))
        text = _content_text(delta.get("content"))
        if text:
            content_parts.append(text)
            if on_text_delta is not None:
                on_text_delta(text)
        for raw_call in as_list(delta.get("tool_calls")):
            if not isinstance(raw_call, dict):
                continue
            index = int(raw_call.get("index") or 0)
            target = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            target["id"] = str(raw_call.get("id") or target["id"])
            function = as_dict(raw_call.get("function"))
            target["name"] += str(function.get("name") or "")
            target["arguments"] += str(function.get("arguments") or "")
    tool_calls = [_tool_call_from_stream(index, value) for index, value in sorted(calls.items())]
    return ModelTurn(
        content="".join(content_parts),
        tool_calls=tool_calls,
        finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
        model_id=str(getattr(provider, "model", "") or ""),
        model_role=str(getattr(provider, "model_role", "") or ""),
        raw=last_event,
    )


def _turn_from_completion(
    response: dict[str, Any],
    provider,
    *,
    on_text_delta: Callable[[str], None] | None,
) -> ModelTurn:
    message = as_dict(response.get("message"))
    content = _content_text(message.get("content"))
    if content and on_text_delta is not None:
        on_text_delta(content)
    calls = []
    for index, raw_call in enumerate(as_list(message.get("tool_calls"))):
        if not isinstance(raw_call, dict):
            continue
        function = as_dict(raw_call.get("function"))
        calls.append(
            ToolCall(
                id=str(raw_call.get("id") or f"model-call-{index}-{uuid4()}"),
                name=str(function.get("name") or ""),
                arguments=_json_arguments(function.get("arguments")),
            )
        )
    return ModelTurn(
        content=content,
        tool_calls=calls,
        finish_reason=str(response.get("finish_reason") or ("tool_calls" if calls else "stop")),
        model_id=str(response.get("model") or getattr(provider, "model", "") or ""),
        model_role=str(getattr(provider, "model_role", "") or ""),
        raw=response,
    )


def _tool_call_from_stream(index: int, value: dict[str, str]) -> ToolCall:
    name = value.get("name", "").strip()
    if not name:
        raise RuntimeError(f"model tool call {index} did not contain a function name")
    return ToolCall(
        id=value.get("id") or f"model-call-{index}-{uuid4()}",
        name=name,
        arguments=_json_arguments(value.get("arguments")),
    )


def _json_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "{}").strip() or "{}"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        repaired = _repair_truncated_json_object(text, exc)
        if repaired is not None:
            return repaired
        raise ModelToolArgumentsError(
            "invalid_json",
            character_count=len(text),
        ) from exc
    if not isinstance(parsed, dict):
        raise ModelToolArgumentsError(
            "not_an_object",
            character_count=len(text),
        )
    return parsed


def _repair_truncated_json_object(
    text: str,
    error: json.JSONDecodeError,
) -> dict[str, Any] | None:
    """Close only structurally incomplete JSON objects truncated at EOF."""

    stripped = text.strip()
    error_at_end = error.pos >= max(0, len(stripped) - 1)
    if not stripped.startswith("{") or (
        "unterminated string" not in error.msg.lower() and not error_at_end
    ):
        return None

    expected_closers: list[str] = []
    in_string = False
    escaped = False
    for character in stripped:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            expected_closers.append("}")
        elif character == "[":
            expected_closers.append("]")
        elif character in {"}", "]"}:
            if not expected_closers or expected_closers.pop() != character:
                return None

    if escaped or not expected_closers:
        return None

    candidate = stripped
    if in_string:
        candidate += '"'
    elif candidate.endswith(","):
        candidate = candidate[:-1].rstrip()
    candidate += "".join(reversed(expected_closers))
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


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


def _model_tool_description(spec: ToolSpec) -> str:
    parts = [spec.description.strip()]
    policy = spec.usage_policy if isinstance(spec.usage_policy, dict) else {}
    if policy.get("when_to_use"):
        parts.append(f"Use when: {policy['when_to_use']}")
    if policy.get("when_not_to_use"):
        parts.append(f"Do not use when: {policy['when_not_to_use']}")
    return "\n".join(part for part in parts if part)


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text") or "")
            for item in value
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
    return ""


def _call_supported(callable_obj, *args, **kwargs):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        supported = kwargs
    else:
        accepts_kwargs = any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        )
        supported = kwargs if accepts_kwargs else {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
    return callable_obj(*args, **supported)
