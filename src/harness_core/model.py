from __future__ import annotations

import asyncio
import inspect
import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any, Callable, Literal, Protocol, runtime_checkable
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
from harness_core.model_failures import (
    ModelToolContractError,
    classify_model_failure,
    provider_fingerprint,
)
from harness_core.model_health import InMemoryModelHealthStore, ModelHealthStore
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
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] = "auto"
    request_timeout: int = 300
    max_output_tokens: int | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    response_contract: ResponseContract | None = None


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


class ModelInvocationConflict(RuntimeError):
    """Raised when an invocation identity is reused with different input."""


class ModelInvocationUnavailable(RuntimeError):
    """Raised when a prior invocation cannot be safely repeated or replayed."""


class ModelInvocationClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "model-invocation-claim-v1"
    invocation_id: str
    outcome: str
    claim_token: str = ""
    result: ModelTurn | None = None
    failure_type: str = ""
    failure_message: str = ""


class ModelInvocationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "model-invocation-record-v1"
    envelope: ModelInvocationEnvelope
    status: str = "running"
    claim_token: str = ""
    claim_expires_at: datetime | None = None
    result: ModelTurn | None = None
    failure_type: str = ""
    failure_message: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelInvocationStore(Protocol):
    """Atomic ownership and settlement boundary for provider invocations."""

    def claim(
        self,
        envelope: ModelInvocationEnvelope,
        *,
        lease_seconds: float = 300.0,
    ) -> ModelInvocationClaim: ...

    def complete(
        self,
        invocation_id: str,
        *,
        claim_token: str,
        result: ModelTurn,
    ) -> ModelInvocationRecord: ...

    def fail(
        self,
        invocation_id: str,
        *,
        claim_token: str,
        failure_type: str,
        failure_message: str,
    ) -> ModelInvocationRecord: ...

    def get_record(self, invocation_id: str) -> ModelInvocationRecord | None: ...


class InMemoryModelInvocationStore:
    """Thread-safe reference store with fail-closed ambiguous recovery."""

    def __init__(self) -> None:
        self._records: dict[str, ModelInvocationRecord] = {}
        self._lock = Lock()

    def claim(
        self,
        envelope: ModelInvocationEnvelope,
        *,
        lease_seconds: float = 300.0,
    ) -> ModelInvocationClaim:
        now = datetime.now(UTC)
        with self._lock:
            record = self._records.get(envelope.invocation_id)
            if record is not None:
                if record.envelope.request_fingerprint != envelope.request_fingerprint:
                    raise ModelInvocationConflict(
                        "invocation_id cannot be reused with a different request"
                    )
                if record.status == "completed" and record.result is not None:
                    return ModelInvocationClaim(
                        invocation_id=envelope.invocation_id,
                        outcome="replay",
                        result=record.result.model_copy(deep=True),
                    )
                if record.status == "failed":
                    return ModelInvocationClaim(
                        invocation_id=envelope.invocation_id,
                        outcome="failed",
                        failure_type=record.failure_type,
                        failure_message=record.failure_message,
                    )
                if record.claim_expires_at is None or record.claim_expires_at > now:
                    return ModelInvocationClaim(
                        invocation_id=envelope.invocation_id,
                        outcome="in_progress",
                    )
                return ModelInvocationClaim(
                    invocation_id=envelope.invocation_id,
                    outcome="uncertain",
                    failure_type="claim_expired",
                    failure_message=(
                        "the previous provider invocation expired without a durable result"
                    ),
                )

            claim_token = uuid4().hex
            self._records[envelope.invocation_id] = ModelInvocationRecord(
                envelope=envelope.model_copy(deep=True),
                status="running",
                claim_token=claim_token,
                claim_expires_at=now + timedelta(seconds=max(1.0, float(lease_seconds))),
                updated_at=now,
            )
            return ModelInvocationClaim(
                invocation_id=envelope.invocation_id,
                outcome="acquired",
                claim_token=claim_token,
            )

    def complete(
        self,
        invocation_id: str,
        *,
        claim_token: str,
        result: ModelTurn,
    ) -> ModelInvocationRecord:
        return self._settle(
            invocation_id,
            claim_token=claim_token,
            status="completed",
            result=result,
        )

    def fail(
        self,
        invocation_id: str,
        *,
        claim_token: str,
        failure_type: str,
        failure_message: str,
    ) -> ModelInvocationRecord:
        return self._settle(
            invocation_id,
            claim_token=claim_token,
            status="failed",
            failure_type=failure_type,
            failure_message=failure_message,
        )

    def get_record(self, invocation_id: str) -> ModelInvocationRecord | None:
        with self._lock:
            record = self._records.get(str(invocation_id or ""))
            return record.model_copy(deep=True) if record is not None else None

    def _settle(
        self,
        invocation_id: str,
        *,
        claim_token: str,
        status: str,
        result: ModelTurn | None = None,
        failure_type: str = "",
        failure_message: str = "",
    ) -> ModelInvocationRecord:
        with self._lock:
            record = self._records.get(str(invocation_id or ""))
            if record is None:
                raise ModelInvocationConflict("cannot settle an unknown invocation")
            if record.status in {"completed", "failed"}:
                same_result = status == record.status and (
                    status == "failed"
                    or (record.result is not None and record.result == result)
                )
                if same_result:
                    return record.model_copy(deep=True)
                raise ModelInvocationConflict("invocation is already settled")
            if not claim_token or claim_token != record.claim_token:
                raise ModelInvocationConflict("model invocation claim token changed")
            settled = record.model_copy(
                update={
                    "status": status,
                    "claim_token": "",
                    "claim_expires_at": None,
                    "result": result.model_copy(deep=True) if result is not None else None,
                    "failure_type": str(failure_type or ""),
                    "failure_message": str(failure_message or "")[:500],
                    "updated_at": datetime.now(UTC),
                },
                deep=True,
            )
            self._records[invocation_id] = settled
            return settled.model_copy(deep=True)


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
            provider_capabilities = getattr(
                provider.info,
                "capabilities",
                ModelCapabilities(source="adapter_unknown"),
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
                "required_capabilities": {
                    "native_tools": bool(tools or step_context.forced_tool_name),
                    "forced_tool_choice": bool(step_context.forced_tool_name),
                    "streaming": on_text_delta is not None,
                    "image_input": requires_image_input,
                },
                "provider_capabilities": provider_capabilities.model_dump(mode="json"),
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
            forced_tool_name = str(step_context.forced_tool_name or "").strip()

            def tracked_delta(delta: str) -> None:
                nonlocal visible_delta_emitted
                ensure_time_remaining(step_context.deadline_monotonic)
                if delta:
                    if on_text_delta is not None and not forced_tool_name:
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
                if on_text_delta is not None and not forced_tool_name:
                    on_text_delta(delta)

            envelope: ModelInvocationEnvelope | None = None
            claim_token = ""
            try:
                ensure_time_remaining(step_context.deadline_monotonic)
                if not provider.info.supports_native_tools:
                    raise RuntimeError("provider does not support native tool calls")
                invocation = ModelInvocation(
                    messages=provider_messages,
                    tools=tools,
                    tool_choice=_tool_choice(step_context, provider_capabilities),
                    request_timeout=request_timeout,
                    max_output_tokens=route_payload["max_output_tokens"],
                    reasoning_effort=route_payload["reasoning_effort"],
                )
                envelope = ModelInvocationEnvelope(
                    invocation_id=(
                        f"{step_context.run_id}:turn:{step_context.turn_id}:"
                        f"model:{step_context.step_index}:"
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
                if self.invocation_store is not None:
                    claim = self.invocation_store.claim(
                        envelope,
                        lease_seconds=max(30.0, float(request_timeout) + 30.0),
                    )
                    route_payload["invocation"]["claim_outcome"] = claim.outcome
                    if claim.outcome == "replay" and claim.result is not None:
                        turn = claim.result.model_copy(deep=True)
                        route_payload["invocation"]["replayed"] = True
                    elif claim.outcome == "acquired":
                        claim_token = claim.claim_token
                        turn = await _ainvoke_provider(
                            provider,
                            invocation,
                            on_text_delta=tracked_delta,
                        )
                    else:
                        detail = claim.failure_message or (
                            "the model invocation is already executing"
                            if claim.outcome == "in_progress"
                            else "the model invocation cannot be safely replayed"
                        )
                        raise ModelInvocationUnavailable(
                            f"model invocation {claim.outcome}: {detail}"
                        )
                else:
                    turn = await _ainvoke_provider(
                        provider,
                        invocation,
                        on_text_delta=tracked_delta,
                    )
                _validate_forced_tool_turn(turn, step_context)
                ensure_time_remaining(step_context.deadline_monotonic)
                self.model_health_store.record_success(
                    provider.info.provider_id,
                    provider.info.model_id,
                )
                if (
                    self.invocation_store is not None
                    and envelope is not None
                    and claim_token
                ):
                    self.invocation_store.complete(
                        envelope.invocation_id,
                        claim_token=claim_token,
                        result=turn,
                    )
            except Exception as exc:
                if (
                    self.invocation_store is not None
                    and envelope is not None
                    and claim_token
                ):
                    try:
                        self.invocation_store.fail(
                            envelope.invocation_id,
                            claim_token=claim_token,
                            failure_type=type(exc).__name__,
                            failure_message=str(exc),
                        )
                    except Exception as settlement_error:
                        route_payload["invocation"]["settlement_error"] = type(
                            settlement_error
                        ).__name__
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
                if failure.retryable:
                    route_payload["health"] = self.model_health_store.record_failure(
                        provider.info.provider_id,
                        provider.info.model_id,
                        category=failure.category,
                        immediate=failure.category == "rate_limited",
                        retry_after_seconds=failure.retry_after_seconds,
                    ).as_dict()
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
        result.append({"role": "system", "content": system_prompt})
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
