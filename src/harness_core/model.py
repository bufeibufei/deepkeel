from __future__ import annotations

import inspect
import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Callable, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from harness_core.budget import (
    INPUT_TOKENS,
    MODEL_CALLS,
    MODEL_RETRIES,
    OUTPUT_TOKENS,
    BudgetPolicy,
    BudgetExceededError,
    BudgetLedger,
    BudgetRequest,
    BudgetSnapshot,
    UsageReport,
    preview_budget,
)
from harness_core.context_window import ConservativeTokenEstimator
from harness_core.contracts import AgentMessage, ToolCall
from harness_core.deadlines import ensure_time_remaining, remaining_timeout_ceiling
from harness_core.model_failures import classify_model_failure, provider_fingerprint
from harness_core.model_routing import (
    AdaptiveStepModelRouter,
    ModelRouteDecision,
    ModelRouter,
    ModelStepContext,
)
from harness_core.policy import (
    PolicyDeniedError,
    PolicyEngine,
    PolicyRequest,
)
from harness_core.tool_registry import ToolRegistry, ToolSpec
from harness_core.type_narrowing import as_dict, as_list


ModelRouteSink = Callable[[dict[str, Any]], None]


class ModelTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = ""
    model_id: str = ""
    model_role: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelProviderInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str
    model_id: str = ""
    model_role: str = "reasoning"
    supports_streaming: bool = True
    supports_native_tools: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] = "auto"
    request_timeout: int = 300
    max_output_tokens: int | None = None


class ModelInvocationEnvelope(BaseModel):
    """Exact, access-controlled replay input for one governed model attempt."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "model-invocation-envelope-v1"
    invocation_id: str
    run_id: str
    thread_id: str
    turn_id: str
    step_index: int = 0
    attempt_index: int = 1
    retry_kind: str = "primary"
    provider_id: str = ""
    model_id: str = ""
    model_role: str = ""
    router_id: str = ""
    route_reason: str = ""
    estimated_input_tokens: int = 0
    request: ModelInvocation
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def request_fingerprint(self) -> str:
        encoded = json.dumps(
            self.request.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "invocation_id": self.invocation_id,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "step_index": self.step_index,
            "attempt_index": self.attempt_index,
            "retry_kind": self.retry_kind,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_role": self.model_role,
            "router_id": self.router_id,
            "route_reason": self.route_reason,
            "estimated_input_tokens": self.estimated_input_tokens,
            "message_count": len(self.request.messages),
            "tool_count": len(self.request.tools),
            "tool_names": [
                str((tool.get("function") or {}).get("name") or tool.get("name") or "")
                for tool in self.request.tools
                if isinstance(tool, dict)
            ],
            "request_fingerprint": self.request_fingerprint,
            "created_at": self.created_at.isoformat(),
        }


class ModelInvocationRecorder(Protocol):
    def record(self, envelope: ModelInvocationEnvelope) -> None: ...

    def get(self, invocation_id: str) -> ModelInvocationEnvelope | None: ...


class InMemoryModelInvocationRecorder:
    """Reference recorder that keeps exact prompts outside ordinary event payloads."""

    def __init__(self) -> None:
        self._records: dict[str, ModelInvocationEnvelope] = {}
        self._lock = Lock()

    def record(self, envelope: ModelInvocationEnvelope) -> None:
        with self._lock:
            existing = self._records.get(envelope.invocation_id)
            if existing is not None and existing.request_fingerprint != envelope.request_fingerprint:
                raise ValueError("invocation_id cannot be reused with a different request")
            self._records[envelope.invocation_id] = envelope.model_copy(deep=True)

    def get(self, invocation_id: str) -> ModelInvocationEnvelope | None:
        with self._lock:
            envelope = self._records.get(str(invocation_id or ""))
            return envelope.model_copy(deep=True) if envelope is not None else None


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


class NativeChatProviderAdapter:
    """Adapter for providers exposing stream_chat or complete_chat."""

    def __init__(self, provider, *, request_timeout: int = 300):
        self.provider = provider
        self.request_timeout = request_timeout

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
        )

    def invoke(
        self,
        request: ModelInvocation,
        *,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ModelTurn:
        if callable(getattr(self.provider, "stream_chat", None)):
            events = self._call_stream_chat(
                request.messages,
                request.tools,
                tool_choice=request.tool_choice,
                request_timeout=request.request_timeout,
                max_output_tokens=request.max_output_tokens,
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
            )
            return _turn_from_completion(
                response,
                self.provider,
                on_text_delta=on_text_delta,
            )
        raise RuntimeError("provider does not support native tool calls")

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
        tool_choice = _tool_choice(step_context)
        turn = self.invoke(
            ModelInvocation(
                messages=provider_messages,
                tools=tools,
                tool_choice=tool_choice,
                request_timeout=request_timeout,
            ),
            on_text_delta=checked_delta,
        )
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
    ):
        return _call_supported(
            self.provider.stream_chat,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            request_timeout=request_timeout,
            max_tokens=max_output_tokens,
        )

    def _call_complete_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any],
        request_timeout: int,
        max_output_tokens: int | None,
    ):
        return _call_supported(
            self.provider.complete_chat,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            request_timeout=request_timeout,
            max_tokens=max_output_tokens,
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
                model_policy=step_context.model_policy,
                skill_activation=step_context.skill_activation,
                policy_phase=step_context.policy_phase,
                forced_tool_name=step_context.forced_tool_name,
                governance_scope=step_context.governance_scope,
                deadline_monotonic=step_context.deadline_monotonic,
            )
        primary_route = self.router.route(step_context)
        attempts = self._attempts(primary_route)
        previous_failure: dict[str, Any] = {}
        token_estimator = ConservativeTokenEstimator()
        budget_policy = BudgetPolicy.from_mapping(
            step_context.model_policy.get("budget")
            if isinstance(step_context.model_policy.get("budget"), dict)
            else {}
        )

        for attempt_index, (route, provider, retry_kind) in enumerate(attempts, start=1):
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
            provider_messages = provider_messages_from_agent(
                messages,
                system_prompt=system_prompt,
            )
            estimated_input_tokens = token_estimator.estimate(
                {"messages": provider_messages, "tools": tools}
            )
            per_call_input_limit = budget_policy.limit(
                "max_input_tokens_per_call",
                role=route.role,
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
            )
            route_payload = {
                **route.as_dict(),
                "model_id": provider.info.model_id,
                "provider_id": provider.info.provider_id,
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

            visible_delta_emitted = False
            streamed_output_parts: list[str] = []

            def tracked_delta(delta: str) -> None:
                nonlocal visible_delta_emitted
                ensure_time_remaining(step_context.deadline_monotonic)
                if delta:
                    visible_delta_emitted = True
                    streamed_output_parts.append(delta)
                    if (
                        max_output_tokens is not None
                        and token_estimator.estimate("".join(streamed_output_parts))
                        > max_output_tokens
                    ):
                        raise BudgetExceededError(
                            preview_budget(
                                BudgetSnapshot(run_id=step_context.run_id),
                                BudgetRequest(
                                    run_id=step_context.run_id,
                                    metric=OUTPUT_TOKENS,
                                    amount=token_estimator.estimate(
                                        "".join(streamed_output_parts)
                                    ),
                                    limit=max_output_tokens,
                                ),
                            )
                        )
                if on_text_delta is not None:
                    on_text_delta(delta)

            try:
                ensure_time_remaining(step_context.deadline_monotonic)
                if not provider.info.supports_native_tools:
                    raise RuntimeError("provider does not support native tool calls")
                invocation = ModelInvocation(
                    messages=provider_messages,
                    tools=tools,
                    tool_choice=_tool_choice(step_context),
                    request_timeout=request_timeout,
                    max_output_tokens=route_payload["max_output_tokens"],
                )
                envelope = ModelInvocationEnvelope(
                    invocation_id=(
                        f"{step_context.run_id}:model:{step_context.step_index}:"
                        f"attempt:{attempt_index}"
                    ),
                    run_id=step_context.run_id,
                    thread_id=step_context.thread_id,
                    turn_id=step_context.turn_id,
                    step_index=step_context.step_index,
                    attempt_index=attempt_index,
                    retry_kind=retry_kind,
                    provider_id=provider.info.provider_id,
                    model_id=provider.info.model_id,
                    model_role=route.role,
                    router_id=route.router_id,
                    route_reason=route.reason,
                    estimated_input_tokens=estimated_input_tokens,
                    request=invocation,
                )
                route_payload["invocation"] = envelope.public_snapshot()
                if self.invocation_recorder is not None:
                    try:
                        self.invocation_recorder.record(envelope)
                        route_payload["invocation"]["recorded"] = True
                    except Exception as recorder_error:
                        route_payload["invocation"]["recorded"] = False
                        route_payload["invocation"]["recorder_error"] = type(
                            recorder_error
                        ).__name__
                turn = provider.invoke(
                    invocation,
                    on_text_delta=tracked_delta,
                )
                ensure_time_remaining(step_context.deadline_monotonic)
            except Exception as exc:
                failed_output_tokens = (
                    token_estimator.estimate("".join(streamed_output_parts))
                    if streamed_output_parts
                    else 0
                )
                failed_usage = UsageReport(
                    input_tokens=estimated_input_tokens,
                    output_tokens=failed_output_tokens,
                    total_tokens=estimated_input_tokens + failed_output_tokens,
                    source="estimated_failure",
                )
                failed_input_budget = self.budget_ledger.consume(
                    BudgetRequest(
                        run_id=step_context.run_id,
                        metric=INPUT_TOKENS,
                        amount=estimated_input_tokens,
                        limit=budget_policy.limit("max_input_tokens_total"),
                        operation_id=(
                            f"model-input:{step_context.step_index}:attempt:{attempt_index}"
                        ),
                        metadata={"role": route.role, "usage_source": "estimated_failure"},
                    )
                )
                route_payload["budget_metrics"][INPUT_TOKENS] = failed_input_budget.as_dict()
                failed_output_budget = None
                if failed_output_tokens:
                    failed_output_budget = self.budget_ledger.consume(
                        BudgetRequest(
                            run_id=step_context.run_id,
                            metric=OUTPUT_TOKENS,
                            amount=failed_output_tokens,
                            limit=budget_policy.limit("max_output_tokens_total"),
                            operation_id=(
                                f"model-output:{step_context.step_index}:attempt:{attempt_index}"
                            ),
                            metadata={
                                "role": route.role,
                                "usage_source": "estimated_failure",
                            },
                        )
                    )
                    route_payload["budget_metrics"][OUTPUT_TOKENS] = (
                        failed_output_budget.as_dict()
                    )
                route_payload["usage"] = failed_usage.as_dict()
                if not failed_input_budget.allowed:
                    if on_route is not None:
                        on_route(route_payload)
                    raise BudgetExceededError(failed_input_budget) from exc
                if failed_output_budget is not None and not failed_output_budget.allowed:
                    if on_route is not None:
                        on_route(route_payload)
                    raise BudgetExceededError(failed_output_budget) from exc
                failure = classify_model_failure(exc)
                can_retry = (
                    failure.retryable
                    and not visible_delta_emitted
                    and attempt_index < len(attempts)
                )
                if not can_retry:
                    if on_route is not None:
                        on_route(route_payload)
                    raise
                previous_failure = {
                    "fallback_from": route.role,
                    "failure_category": failure.category,
                    "failure_status_code": failure.status_code,
                }
                if on_route is not None:
                    on_route(route_payload)
                continue
            estimated_output_tokens = token_estimator.estimate(
                {
                    "content": turn.content,
                    "tool_calls": [call.model_dump() for call in turn.tool_calls],
                }
            )
            usage = UsageReport.from_provider(
                _provider_usage(turn.raw),
                estimated_input=estimated_input_tokens,
                estimated_output=estimated_output_tokens,
            )
            settled_input_budget = self.budget_ledger.consume(
                BudgetRequest(
                    run_id=step_context.run_id,
                    metric=INPUT_TOKENS,
                    amount=usage.input_tokens,
                    limit=budget_policy.limit("max_input_tokens_total"),
                    operation_id=(
                        f"model-input:{step_context.step_index}:attempt:{attempt_index}"
                    ),
                    metadata={"role": route.role, "usage_source": usage.source},
                )
            )
            output_budget = self.budget_ledger.consume(
                BudgetRequest(
                    run_id=step_context.run_id,
                    metric=OUTPUT_TOKENS,
                    amount=usage.output_tokens,
                    limit=budget_policy.limit("max_output_tokens_total"),
                    operation_id=(
                        f"model-output:{step_context.step_index}:attempt:{attempt_index}"
                    ),
                    metadata={"role": route.role, "usage_source": usage.source},
                )
            )
            route_payload["budget_metrics"][INPUT_TOKENS] = settled_input_budget.as_dict()
            route_payload["budget_metrics"][OUTPUT_TOKENS] = output_budget.as_dict()
            route_payload["usage"] = usage.as_dict()
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
    ) -> list[tuple[ModelRouteDecision, ModelProviderAdapter, str]]:
        primary = self.providers.get(primary_route.role)
        if primary is None:
            raise RuntimeError(
                f"model provider for role {primary_route.role!r} is unavailable"
            )
        result = [(primary_route, primary, "primary")]
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


def _as_provider_adapter(provider: Any) -> ModelProviderAdapter:
    return (
        provider
        if isinstance(provider, ModelProviderAdapter)
        else NativeChatProviderAdapter(provider)
    )


def _adapter_fingerprint(provider: ModelProviderAdapter) -> tuple[str, str, str]:
    info = provider.info
    return (info.provider_id, info.model_id, info.model_role)


def _tool_choice(step_context: ModelStepContext | None) -> str | dict[str, Any]:
    forced_tool_name = str(
        step_context.forced_tool_name if step_context is not None else ""
    ).strip()
    if not forced_tool_name:
        return "auto"
    return {
        "type": "function",
        "function": {"name": forced_tool_name},
    }


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
) -> int | None:
    total_limit = policy.limit("max_output_tokens_total")
    per_call_limit = policy.limit("max_output_tokens_per_call", role=role)
    remaining_total = (
        max(0, int(total_limit - float(snapshot.usage.get(OUTPUT_TOKENS) or 0)))
        if total_limit is not None
        else None
    )
    candidates = [
        int(value)
        for value in (remaining_total, per_call_limit)
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
        result.append({"role": "system", "content": system_prompt})
    for message in messages:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
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
        raise RuntimeError("model returned invalid native tool arguments") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("model native tool arguments must be a JSON object")
    return parsed


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
