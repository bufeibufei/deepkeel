from __future__ import annotations

import inspect
import json
from copy import deepcopy
from typing import Any, Callable
from uuid import uuid4

from deepkeel.contracts import AgentMessage, ToolCall
from deepkeel.deadlines import ensure_time_remaining, remaining_timeout_ceiling
from deepkeel.model_capabilities import (
    InMemoryModelCapabilityRegistry,
    ResponseContract,
    ResponseFormat,
    StructuredOutputAttempt,
    negotiate_structured_output,
    response_format_not_supported,
    response_format_payload,
    structured_output_prompt,
)
from deepkeel.model_failures import ModelToolArgumentsError, provider_fingerprint
from deepkeel.model_gateway_support import (
    _tool_choice,
    _validate_forced_tool_turn,
    provider_messages_from_agent,
)
from deepkeel.model_invocations import ModelInvocation, ModelProviderInfo, ModelTurn
from deepkeel.model_provider_contracts import ModelRouteSink
from deepkeel.model_routing import ModelStepContext
from deepkeel.type_narrowing import as_dict, as_list


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
                ":".join(part for part in fingerprint if part) or type(self.provider).__name__
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
            return turn.model_copy(update={"raw": {**turn.raw, "structured_output": diagnostics}})
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
        deadline_monotonic = step_context.deadline_monotonic if step_context is not None else None
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
            item.kind == inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values()
        )
        supported = (
            kwargs
            if accepts_kwargs
            else {key: value for key, value in kwargs.items() if key in signature.parameters}
        )
    return callable_obj(*args, **supported)
