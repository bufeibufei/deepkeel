from __future__ import annotations

import inspect
import json
from copy import deepcopy
from typing import Any, Callable, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from harness_core.budget import (
    MODEL_CALLS,
    BudgetExceededError,
    BudgetLedger,
    BudgetRequest,
)
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


ModelRouteSink = Callable[[dict[str, Any]], None]


class ModelTurn(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = ""
    model_id: str = ""
    model_role: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


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


class HarnessModelAdapter:
    def __init__(self, provider, *, request_timeout: int = 300):
        self.provider = provider
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
                    "role": str(getattr(self.provider, "model_role", "") or "reasoning"),
                    "model_id": str(getattr(self.provider, "model", "") or ""),
                    "reason": "static provider",
                    "router_id": "static-provider",
                    "step_index": step_context.step_index if step_context is not None else 0,
                }
            )
        provider_messages = provider_messages_from_agent(messages, system_prompt=system_prompt)
        tool_choice = _tool_choice(step_context)
        if callable(getattr(self.provider, "stream_chat", None)):
            events = self._call_stream_chat(
                provider_messages,
                tools,
                tool_choice=tool_choice,
                request_timeout=request_timeout,
            )
            turn = _assemble_streamed_turn(
                events,
                provider=self.provider,
                on_text_delta=checked_delta,
            )
        elif callable(getattr(self.provider, "complete_chat", None)):
            response = self._call_complete_chat(
                provider_messages,
                tools,
                tool_choice=tool_choice,
                request_timeout=request_timeout,
            )
            turn = _turn_from_completion(
                response,
                self.provider,
                on_text_delta=checked_delta,
            )
        else:
            raise RuntimeError("provider does not support native tool calls")
        ensure_time_remaining(deadline_monotonic)
        return turn

    def _call_stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any],
        request_timeout: int,
    ):
        return _call_supported(
            self.provider.stream_chat,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            request_timeout=request_timeout,
        )

    def _call_complete_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any],
        request_timeout: int,
    ):
        return _call_supported(
            self.provider.complete_chat,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            request_timeout=request_timeout,
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
        request_timeout: int = 300,
    ) -> None:
        self.providers = {str(role): provider for role, provider in providers.items() if provider is not None}
        self.router = router or AdaptiveStepModelRouter()
        self.policy_engine = policy_engine
        self.budget_ledger = budget_ledger
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

        for attempt_index, (route, provider, retry_kind) in enumerate(attempts, start=1):
            request_timeout = remaining_timeout_ceiling(
                step_context.deadline_monotonic,
                maximum=self.request_timeout,
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
            route_payload = {
                **route.as_dict(),
                "model_id": str(getattr(provider, "model", "") or ""),
                "policy": policy.as_dict(),
                "budget": budget.as_dict() if budget is not None else {},
                "forced_tool_name": step_context.forced_tool_name,
                "attempt_index": attempt_index,
                "retry_kind": retry_kind,
                **previous_failure,
            }
            if on_route is not None:
                on_route(route_payload)
            if not policy.allowed:
                raise PolicyDeniedError(policy)
            if budget is not None and not budget.allowed:
                raise BudgetExceededError(budget)

            visible_delta_emitted = False

            def tracked_delta(delta: str) -> None:
                nonlocal visible_delta_emitted
                if delta:
                    visible_delta_emitted = True
                if on_text_delta is not None:
                    on_text_delta(delta)

            try:
                turn = HarnessModelAdapter(
                    provider,
                    request_timeout=request_timeout,
                ).run_turn(
                    messages,
                    tools=tools,
                    system_prompt=system_prompt,
                    on_text_delta=tracked_delta,
                    step_context=step_context,
                )
            except Exception as exc:
                failure = classify_model_failure(exc)
                can_retry = (
                    failure.retryable
                    and not visible_delta_emitted
                    and attempt_index < len(attempts)
                )
                if not can_retry:
                    raise
                previous_failure = {
                    "fallback_from": route.role,
                    "failure_category": failure.category,
                    "failure_status_code": failure.status_code,
                }
                continue
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
    ) -> list[tuple[ModelRouteDecision, Any, str]]:
        primary = self.providers.get(primary_route.role)
        if primary is None:
            raise RuntimeError(
                f"model provider for role {primary_route.role!r} is unavailable"
            )
        result = [(primary_route, primary, "primary")]
        primary_fingerprint = provider_fingerprint(primary)
        preferred_roles = (
            ("reasoning", "fast")
            if primary_route.role == "fast"
            else ("fast", "reasoning")
        )
        ordered_roles = tuple(dict.fromkeys((*preferred_roles, *self.providers)))
        for role in ordered_roles:
            provider = self.providers.get(role)
            if provider is None or provider_fingerprint(provider) == primary_fingerprint:
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
    budget = model_policy.get("budget") if isinstance(model_policy.get("budget"), dict) else {}
    value = budget.get("max_model_calls")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


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

    schema = spec.input_schema if isinstance(spec.input_schema, dict) else {}
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            str(name): _json_schema_for_example(example)
            for name, example in schema.items()
        },
        "additionalProperties": False,
    }
    required = [name for name in spec.required_args if "." not in name]
    if required:
        parameters["required"] = required
    groups = [
        [name for name in group if "." not in name]
        for group in spec.required_arg_groups
    ]
    groups = [group for group in groups if group]
    if groups:
        parameters["anyOf"] = [
            {"required": [name]}
            for group in groups
            for name in group
        ]
    return parameters


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
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        text = _content_text(delta.get("content"))
        if text:
            content_parts.append(text)
            if on_text_delta is not None:
                on_text_delta(text)
        for raw_call in delta.get("tool_calls", []) if isinstance(delta.get("tool_calls"), list) else []:
            if not isinstance(raw_call, dict):
                continue
            index = int(raw_call.get("index") or 0)
            target = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            target["id"] = str(raw_call.get("id") or target["id"])
            function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
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
    message = response.get("message") if isinstance(response.get("message"), dict) else {}
    content = _content_text(message.get("content"))
    if content and on_text_delta is not None:
        on_text_delta(content)
    calls = []
    for index, raw_call in enumerate(message.get("tool_calls", []) if isinstance(message.get("tool_calls"), list) else []):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
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


def _json_schema_for_example(example: Any) -> dict[str, Any]:
    if isinstance(example, bool):
        return {"type": "boolean", "default": example}
    if isinstance(example, int):
        return {"type": "integer", "default": example}
    if isinstance(example, float):
        return {"type": "number", "default": example}
    if isinstance(example, list):
        item = example[0] if example else "string"
        item_schema = _json_schema_for_example(item)
        item_schema.pop("default", None)
        return {"type": "array", "items": item_schema}
    if isinstance(example, dict):
        return {
            "type": "object",
            "properties": {str(key): _json_schema_for_example(value) for key, value in example.items()},
            "additionalProperties": False,
        }
    text = str(example or "string")
    if "|" in text and all(part.strip() for part in text.split("|")):
        return {"type": "string", "enum": [part.strip() for part in text.split("|")]}
    normalized = text.strip().lower()
    if normalized in {"int", "integer"}:
        return {"type": "integer"}
    if normalized in {"bool", "boolean"}:
        return {"type": "boolean"}
    if normalized in {"array", "list"}:
        return {"type": "array", "items": {}}
    if normalized in {"object", "dict"}:
        return {"type": "object"}
    return {"type": "string"}


def _model_tool_description(spec: ToolSpec) -> str:
    parts = [spec.description.strip()]
    policy = spec.usage_policy if isinstance(spec.usage_policy, dict) else {}
    if policy.get("when_to_use"):
        parts.append(f"适用：{policy['when_to_use']}")
    if policy.get("when_not_to_use"):
        parts.append(f"不适用：{policy['when_not_to_use']}")
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
