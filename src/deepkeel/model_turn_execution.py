from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from deepkeel.async_ports import run_sync_adapter
from deepkeel.budget import (
    INPUT_TOKENS,
    MODEL_CALLS,
    MODEL_RETRIES,
    BudgetExceededError,
    BudgetPolicy,
    BudgetRequest,
    preview_budget,
)
from deepkeel.context_window import ConservativeTokenEstimator
from deepkeel.contracts import AgentMessage
from deepkeel.deadlines import remaining_timeout_ceiling
from deepkeel.model_capabilities import ModelCapabilities
from deepkeel.model_failures import classify_model_failure
from deepkeel.model_gateway_support import (
    MODEL_FAILURE_AUTO_FALLBACK,
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
from deepkeel.model_invocations import ModelTurn
from deepkeel.model_provider_contracts import (
    AsyncModelProviderAdapter,
    ModelProviderAdapter,
    ModelRouteSink,
)
from deepkeel.model_provider_execution import _ainvoke_provider
from deepkeel.model_routing import ModelRouteDecision, ModelStepContext
from deepkeel.model_step_execution import (
    ModelAttemptExecutionError,
    execute_model_attempt,
    record_failed_attempt_usage,
    record_successful_attempt_usage,
    scoped_model_health_provider_id,
)
from deepkeel.policy import PolicyDeniedError, PolicyRequest
from deepkeel.type_narrowing import as_dict


Provider = ModelProviderAdapter | AsyncModelProviderAdapter


@dataclass(frozen=True, slots=True)
class _AttemptCandidate:
    route: ModelRouteDecision
    provider: Provider
    retry_kind: str
    index: int


@dataclass(frozen=True, slots=True)
class _PreparedAttempt:
    candidate: _AttemptCandidate
    health_provider_id: str
    budget_policy: BudgetPolicy
    provider_messages: list[dict[str, Any]]
    capabilities: ModelCapabilities
    estimated_input_tokens: int
    request_timeout: int
    max_output_tokens: int | None
    reasoning_effort: Literal["low", "medium", "high"] | None
    route_payload: dict[str, Any]


class RoutedModelTurnExecution:
    """Govern and settle one routed model step across bounded attempts."""

    def __init__(
        self,
        owner: Any,
        messages: list[AgentMessage],
        *,
        tools: list[dict[str, Any]],
        system_prompt: str,
        on_text_delta: Callable[[str], None] | None,
        step_context: ModelStepContext,
        on_route: ModelRouteSink | None,
    ) -> None:
        self.owner = owner
        self.messages = messages
        self.tools = tools
        self.system_prompt = system_prompt
        self.on_text_delta = on_text_delta
        self.context = step_context
        self.on_route = on_route
        self.token_estimator = ConservativeTokenEstimator()
        self.requires_image_input = any(
            any(part.type == "image" for part in message.content_parts)
            for message in messages
        )

    async def run(self) -> ModelTurn:
        self.context = await self.owner._routing_context(
            self.context,
            messages=self.messages,
            tools=self.tools,
            requires_image_input=self.requires_image_input,
            token_estimator=self.token_estimator,
        )
        primary = self.owner.router.route(self.context)
        failure_policy = _model_failure_policy(self.context.model_policy)
        attempts = self._candidate_attempts(primary, failure_policy)
        previous_failure: dict[str, Any] = {}
        budget_policy = BudgetPolicy.from_mapping(as_dict(self.context.model_policy.get("budget")))
        for offset, candidate in enumerate(attempts):
            if await self._skip_open_circuit(candidate):
                previous_failure = {
                    "fallback_from": candidate.route.role,
                    "failure_category": "model_circuit_open",
                }
                continue
            prepared = await self._prepare_attempt(
                candidate,
                primary=primary,
                failure_policy=failure_policy,
                budget_policy=budget_policy,
                previous_failure=previous_failure,
            )
            turn, previous_failure = await self._execute_attempt(
                prepared,
                total_attempts=len(attempts),
                next_candidate=(
                    attempts[offset + 1] if offset + 1 < len(attempts) else None
                ),
            )
            if turn is not None:
                return turn
        raise RuntimeError("model invocation attempts exhausted")

    def _candidate_attempts(
        self,
        primary: ModelRouteDecision,
        failure_policy: str,
    ) -> list[_AttemptCandidate]:
        raw = self.owner._attempts(
            primary,
            failure_policy=failure_policy,
            requires_native_tools=bool(self.tools or self.context.forced_tool_name),
        )
        if self.requires_image_input:
            raw = [item for item in raw if _supports_image_input(item[1])]
            if not raw:
                raise RuntimeError("no configured model provider declares image input support")
        return [
            _AttemptCandidate(route=route, provider=provider, retry_kind=kind, index=index)
            for index, (route, provider, kind) in enumerate(raw, start=1)
        ]

    async def _skip_open_circuit(self, candidate: _AttemptCandidate) -> bool:
        info = candidate.provider.info
        health_provider_id = scoped_model_health_provider_id(info.provider_id, self.context)
        health = await run_sync_adapter(
            self.owner.model_health_store.snapshot,
            health_provider_id,
            info.model_id,
        )
        if health.is_available():
            return False
        self._emit_route(
            {
                **candidate.route.as_dict(),
                "model_id": info.model_id,
                "provider_id": info.provider_id,
                "attempt_index": candidate.index,
                "retry_kind": candidate.retry_kind,
                "health": health.as_dict(),
                "skipped": "model_circuit_open",
            }
        )
        return True

    async def _prepare_attempt(
        self,
        candidate: _AttemptCandidate,
        *,
        primary: ModelRouteDecision,
        failure_policy: str,
        budget_policy: BudgetPolicy,
        previous_failure: dict[str, Any],
    ) -> _PreparedAttempt:
        route = candidate.route
        provider = candidate.provider
        policy = self.owner.policy_engine.evaluate(
            PolicyRequest(
                action="model.invoke",
                resource_type="model",
                resource_id=route.role,
                run_id=self.context.run_id,
                user_id=self.context.user_id,
                tenant_id=str(self.context.governance_scope.get("tenant_id") or ""),
                context=self._policy_context(route, candidate.index),
            )
        )
        call_budget = await self._consume_call_budget(
            route.role, candidate.index, allowed=policy.allowed
        )
        budget_snapshot = await run_sync_adapter(
            self.owner.budget_ledger.snapshot, self.context.accounting_run_id
        )
        capabilities = getattr(
            provider.info,
            "capabilities",
            ModelCapabilities(source="adapter_unknown"),
        )
        attempt_prompt = _system_prompt_for_attempt(
            self.system_prompt,
            previous_failure=previous_failure,
            forced_tool_name=self.context.forced_tool_name,
        )
        prepared_context = _prepare_model_context(
            provider_messages_from_agent(self.messages, system_prompt=attempt_prompt),
            self.tools,
            provider=provider,
            route=route,
            provider_capabilities=capabilities,
            budget_policy=budget_policy,
            budget_snapshot=budget_snapshot,
            step_context=self.context,
            per_call_input_limit=budget_policy.limit(
                "max_input_tokens_per_call", role=route.role
            ),
            token_estimator=self.token_estimator,
            context_compactor=self.owner.context_compactor,
        )
        estimated_input = self.token_estimator.estimate(
            {"messages": prepared_context.messages, "tools": self.tools}
        )
        input_budget = self._preview_input_budget(
            budget_policy,
            budget_snapshot,
            route.role,
            estimated_input,
            allowed=policy.allowed,
        )
        retry_budget = await self._consume_retry_budget(
            budget_policy, candidate, allowed=policy.allowed
        )
        max_output = _remaining_output_tokens(
            budget_policy,
            budget_snapshot,
            route.role,
            capabilities=capabilities,
            estimated_input_tokens=estimated_input,
        )
        route_payload = self._route_payload(
            candidate,
            primary=primary,
            failure_policy=failure_policy,
            policy=policy,
            call_budget=call_budget,
            input_budget=input_budget,
            retry_budget=retry_budget,
            capabilities=capabilities,
            max_output=max_output,
            prepared_context=prepared_context,
            previous_failure=previous_failure,
        )
        self._enforce_governance(
            route_payload,
            policy=policy,
            call_budget=call_budget,
            input_budget=input_budget,
            retry_budget=retry_budget,
        )
        return _PreparedAttempt(
            candidate=candidate,
            health_provider_id=scoped_model_health_provider_id(
                provider.info.provider_id, self.context
            ),
            budget_policy=budget_policy,
            provider_messages=prepared_context.messages,
            capabilities=capabilities,
            estimated_input_tokens=estimated_input,
            request_timeout=self._request_timeout(route.role, budget_policy),
            max_output_tokens=max_output,
            reasoning_effort=_reasoning_effort(capabilities, route.role),
            route_payload=route_payload,
        )

    def _policy_context(self, route: ModelRouteDecision, attempt_index: int) -> dict[str, Any]:
        return {
            "skill_activation": self.context.skill_activation,
            "model_policy": self.context.model_policy,
            "route": route.as_dict(),
            "attempt_index": attempt_index,
            "governance_scope": self.context.governance_scope,
            "runtime_policy": as_dict(self.context.model_policy.get("runtime_policy")),
        }

    async def _consume_call_budget(
        self,
        role: str,
        attempt_index: int,
        *,
        allowed: bool,
    ) -> Any:
        if not allowed:
            return None
        return await run_sync_adapter(
            self.owner.budget_ledger.consume,
            BudgetRequest(
                run_id=self.context.accounting_run_id,
                metric=MODEL_CALLS,
                limit=_model_call_limit(self.context.model_policy),
                operation_id=f"model-step:{self.context.step_index}:attempt:{attempt_index}",
                metadata={
                    "role": role,
                    "step_index": self.context.step_index,
                    "attempt_index": attempt_index,
                },
            ),
        )

    def _preview_input_budget(
        self,
        policy: BudgetPolicy,
        snapshot: Any,
        role: str,
        estimated_input: int,
        *,
        allowed: bool,
    ) -> Any:
        per_call_limit = policy.limit("max_input_tokens_per_call", role=role)
        if per_call_limit is not None and estimated_input > per_call_limit:
            raise BudgetExceededError(
                preview_budget(
                    snapshot,
                    BudgetRequest(
                        run_id=self.context.accounting_run_id,
                        metric=INPUT_TOKENS,
                        amount=estimated_input,
                        limit=per_call_limit,
                    ),
                )
            )
        if not allowed:
            return None
        return preview_budget(
            snapshot,
            BudgetRequest(
                run_id=self.context.accounting_run_id,
                metric=INPUT_TOKENS,
                amount=estimated_input,
                limit=policy.limit("max_input_tokens_total"),
                metadata={"role": role},
            ),
        )

    async def _consume_retry_budget(
        self,
        policy: BudgetPolicy,
        candidate: _AttemptCandidate,
        *,
        allowed: bool,
    ) -> Any:
        if not allowed or candidate.index <= 1:
            return None
        return await run_sync_adapter(
            self.owner.budget_ledger.consume,
            BudgetRequest(
                run_id=self.context.accounting_run_id,
                metric=MODEL_RETRIES,
                limit=policy.limit("max_model_retries"),
                operation_id=(
                    f"model-retry:{self.context.step_index}:attempt:{candidate.index}"
                ),
                metadata={
                    "role": candidate.route.role,
                    "retry_kind": candidate.retry_kind,
                },
            ),
        )

    def _route_payload(
        self,
        candidate: _AttemptCandidate,
        *,
        primary: ModelRouteDecision,
        failure_policy: str,
        policy: Any,
        call_budget: Any,
        input_budget: Any,
        retry_budget: Any,
        capabilities: ModelCapabilities,
        max_output: int | None,
        prepared_context: Any,
        previous_failure: dict[str, Any],
    ) -> dict[str, Any]:
        provider = candidate.provider
        return {
            **candidate.route.as_dict(),
            "model_id": provider.info.model_id,
            "provider_id": provider.info.provider_id,
            "requested_role": primary.role,
            "preferred_model_id": self.owner.providers[primary.role].info.model_id,
            "actual_model_id": provider.info.model_id,
            "failure_policy": failure_policy,
            "fallback_allowed": failure_policy == MODEL_FAILURE_AUTO_FALLBACK,
            "policy": policy.as_dict(),
            "budget": call_budget.as_dict() if call_budget is not None else {},
            "budget_metrics": {
                INPUT_TOKENS: input_budget.as_dict() if input_budget is not None else {},
                MODEL_RETRIES: retry_budget.as_dict() if retry_budget is not None else {},
            },
            "forced_tool_name": self.context.forced_tool_name,
            "attempt_index": candidate.index,
            "retry_kind": candidate.retry_kind,
            "max_output_tokens": max_output,
            "reasoning_effort": _reasoning_effort(capabilities, candidate.route.role),
            "model_limits": {
                "context_window_tokens": capabilities.context_window_tokens,
                "max_output_tokens": capabilities.max_output_tokens,
                "capability_source": capabilities.source,
            },
            "context_manifest": prepared_context.diagnostics,
            "required_capabilities": {
                "native_tools": bool(self.tools or self.context.forced_tool_name),
                "forced_tool_choice": bool(self.context.forced_tool_name),
                "streaming": self.on_text_delta is not None,
                "image_input": self.requires_image_input,
            },
            "provider_capabilities": capabilities.model_dump(mode="json"),
            "repair_strategy": _repair_strategy(previous_failure),
            **previous_failure,
        }

    def _enforce_governance(
        self,
        route_payload: dict[str, Any],
        *,
        policy: Any,
        call_budget: Any,
        input_budget: Any,
        retry_budget: Any,
    ) -> None:
        failures = (
            (not policy.allowed, PolicyDeniedError(policy)),
            (
                call_budget is not None and not call_budget.allowed,
                BudgetExceededError(call_budget) if call_budget is not None else None,
            ),
            (
                input_budget is not None and not input_budget.allowed,
                BudgetExceededError(input_budget) if input_budget is not None else None,
            ),
            (
                retry_budget is not None and not retry_budget.allowed,
                BudgetExceededError(retry_budget) if retry_budget is not None else None,
            ),
        )
        for denied, error in failures:
            if denied and error is not None:
                self._emit_route(route_payload)
                raise error

    async def _execute_attempt(
        self,
        prepared: _PreparedAttempt,
        *,
        total_attempts: int,
        next_candidate: _AttemptCandidate | None,
    ) -> tuple[ModelTurn | None, dict[str, Any]]:
        candidate = prepared.candidate
        try:
            outcome = await execute_model_attempt(
                provider=candidate.provider,
                route=candidate.route,
                step_context=self.context,
                attempt_index=candidate.index,
                retry_kind=candidate.retry_kind,
                provider_messages=prepared.provider_messages,
                tools=self.tools,
                tool_choice=_tool_choice(self.context, prepared.capabilities),
                request_timeout=prepared.request_timeout,
                max_output_tokens=prepared.max_output_tokens,
                reasoning_effort=prepared.reasoning_effort,
                estimated_input_tokens=prepared.estimated_input_tokens,
                route_payload=prepared.route_payload,
                token_estimator=self.token_estimator,
                invocation_recorder=self.owner.invocation_recorder,
                invocation_store=self.owner.invocation_store,
                model_health_store=self.owner.model_health_store,
                invoke_provider=_ainvoke_provider,
                validate_turn=_validate_forced_tool_turn,
                on_text_delta=self.on_text_delta,
            )
        except ModelAttemptExecutionError as error:
            return await self._handle_attempt_failure(
                prepared,
                error,
                total_attempts=total_attempts,
                next_candidate=next_candidate,
            )
        return await self._settle_success(prepared, outcome.turn), {}

    async def _handle_attempt_failure(
        self,
        prepared: _PreparedAttempt,
        error: ModelAttemptExecutionError,
        *,
        total_attempts: int,
        next_candidate: _AttemptCandidate | None,
    ) -> tuple[None, dict[str, Any]]:
        candidate = prepared.candidate
        cause = error.cause
        input_budget, output_budget = await run_sync_adapter(
            record_failed_attempt_usage,
            ledger=self.owner.budget_ledger,
            policy=prepared.budget_policy,
            step_context=self.context,
            role=candidate.route.role,
            attempt_index=candidate.index,
            estimated_input_tokens=prepared.estimated_input_tokens,
            streamed_output=error.streamed_output,
            token_estimator=self.token_estimator,
            route_payload=prepared.route_payload,
        )
        if not input_budget.allowed:
            self._emit_route(prepared.route_payload)
            raise BudgetExceededError(input_budget) from cause
        if output_budget is not None and not output_budget.allowed:
            self._emit_route(prepared.route_payload)
            raise BudgetExceededError(output_budget) from cause
        failure = classify_model_failure(cause)
        if failure.retryable and failure.degrades_provider_health:
            health = await run_sync_adapter(
                self.owner.model_health_store.record_failure,
                prepared.health_provider_id,
                candidate.provider.info.model_id,
                category=failure.category,
                immediate=failure.category in {
                    "rate_limited",
                    "provider_capability_unsupported",
                },
                retry_after_seconds=failure.retry_after_seconds,
            )
            prepared.route_payload["health"] = health.as_dict()
        can_retry = (
            failure.retryable
            and not error.visible_delta_emitted
            and candidate.index < total_attempts
            and (
                not failure.fallback_only
                or _is_distinct_candidate(candidate, next_candidate)
            )
        )
        self._emit_route(prepared.route_payload)
        if not can_retry:
            raise cause
        return None, {
            "fallback_from": candidate.route.role,
            "failure_category": failure.category,
            "failure_status_code": failure.status_code,
        }

    async def _settle_success(
        self,
        prepared: _PreparedAttempt,
        turn: ModelTurn,
    ) -> ModelTurn:
        candidate = prepared.candidate
        input_budget, output_budget = await run_sync_adapter(
            record_successful_attempt_usage,
            ledger=self.owner.budget_ledger,
            policy=prepared.budget_policy,
            step_context=self.context,
            role=candidate.route.role,
            attempt_index=candidate.index,
            estimated_input_tokens=prepared.estimated_input_tokens,
            turn=turn,
            provider_usage=_provider_usage(turn.raw),
            token_estimator=self.token_estimator,
            route_payload=prepared.route_payload,
        )
        self._emit_route(prepared.route_payload)
        if not input_budget.allowed:
            raise BudgetExceededError(input_budget)
        if not output_budget.allowed:
            raise BudgetExceededError(output_budget)
        return turn.model_copy(
            update={
                "model_role": candidate.route.role,
                "raw": {**turn.raw, "harness_route": prepared.route_payload},
            }
        )

    def _request_timeout(self, role: str, policy: BudgetPolicy) -> int:
        timeout = remaining_timeout_ceiling(
            self.context.deadline_monotonic,
            maximum=self.owner.request_timeout,
        )
        limit = policy.limit("max_request_seconds", role=role)
        return min(timeout, max(1, int(limit))) if limit is not None else timeout

    def _emit_route(self, payload: dict[str, Any]) -> None:
        if self.on_route is not None:
            self.on_route(payload)


def _supports_image_input(provider: Provider) -> bool:
    capabilities = getattr(provider.info, "capabilities", None)
    return getattr(capabilities, "supports_image_input", None) is True


def _is_distinct_candidate(
    current: _AttemptCandidate,
    candidate: _AttemptCandidate | None,
) -> bool:
    if candidate is None:
        return False
    current_info = current.provider.info
    candidate_info = candidate.provider.info
    return (
        current_info.provider_id,
        current_info.model_id,
    ) != (
        candidate_info.provider_id,
        candidate_info.model_id,
    )


def _repair_strategy(previous_failure: dict[str, Any]) -> str:
    category = previous_failure.get("failure_category")
    if category == "tool_arguments_invalid":
        return "native_tool_arguments_json"
    if category == "tool_contract_violation":
        return "forced_tool_contract"
    return ""
