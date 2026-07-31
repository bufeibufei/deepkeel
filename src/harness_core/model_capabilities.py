from __future__ import annotations

import json
from enum import StrEnum
from threading import Lock
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from harness_core.model_failures import provider_fingerprint


class ResponseFormat(StrEnum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class ModelCapabilities(BaseModel):
    """Provider-neutral feature evidence for one concrete provider/model pair."""

    model_config = ConfigDict(extra="forbid")

    supports_streaming: bool | None = None
    supports_native_tools: bool | None = None
    supports_forced_tool_choice: bool | None = None
    supports_reasoning: bool | None = None
    supports_reasoning_effort: bool | None = None
    supports_image_input: bool | None = None
    context_window_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    completion_limit_parameter: Literal[
        "max_tokens",
        "max_completion_tokens",
    ] | None = None
    supported_response_formats: set[ResponseFormat] = Field(
        default_factory=lambda: {ResponseFormat.TEXT}
    )
    unsupported_response_formats: set[ResponseFormat] = Field(default_factory=set)
    source: str = "unknown"

    def support_for(self, response_format: ResponseFormat) -> bool | None:
        if response_format in self.unsupported_response_formats:
            return False
        if response_format in self.supported_response_formats:
            return True
        return None


class ResponseContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "structured_result"
    schema_: dict[str, Any] = Field(alias="schema")
    strictness: Literal["required", "prefer_strict", "best_effort"] = "prefer_strict"


class StructuredOutputDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_format: ResponseFormat
    candidate_formats: tuple[ResponseFormat, ...]
    capability_source: str = "unknown"


class StructuredOutputAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_format: ResponseFormat
    outcome: Literal["completed", "unsupported", "failed"]
    detail: str = ""


class StructuredOutputInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    requested_format: ResponseFormat
    effective_format: ResponseFormat
    capability_source: str = "unknown"
    attempts: list[StructuredOutputAttempt] = Field(default_factory=list)

    def diagnostics(self) -> dict[str, Any]:
        degraded = self.effective_format != self.requested_format
        return {
            "requested_format": self.requested_format.value,
            "effective_format": self.effective_format.value,
            "capability_source": self.capability_source,
            "degraded": degraded,
            "degradation_reason": (
                "model_response_format_not_supported" if degraded else ""
            ),
            "attempts": [attempt.model_dump(mode="json") for attempt in self.attempts],
        }


class ModelCapabilityUnsatisfiedError(RuntimeError):
    code = "MODEL_CAPABILITY_UNSATISFIED"


class InMemoryModelCapabilityRegistry:
    """Learns provider capabilities without coupling Core to a vendor catalog."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str, str], ModelCapabilities] = {}
        self._lock = Lock()

    def capabilities_for(self, provider: Any) -> ModelCapabilities:
        key = provider_fingerprint(provider)
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                return cached.model_copy(deep=True)
            declared = model_capabilities_for_provider(provider)
            self._items[key] = declared.model_copy(deep=True)
            return declared

    def mark_response_format(
        self,
        provider: Any,
        response_format: ResponseFormat,
        *,
        supported: bool,
        source: str = "runtime_observation",
    ) -> ModelCapabilities:
        key = provider_fingerprint(provider)
        with self._lock:
            current = self._items.get(key) or model_capabilities_for_provider(provider)
            supported_formats = set(current.supported_response_formats)
            unsupported_formats = set(current.unsupported_response_formats)
            if supported:
                supported_formats.add(response_format)
                unsupported_formats.discard(response_format)
            else:
                unsupported_formats.add(response_format)
                supported_formats.discard(response_format)
            updated = current.model_copy(
                update={
                    "supported_response_formats": supported_formats,
                    "unsupported_response_formats": unsupported_formats,
                    "source": source,
                },
                deep=True,
            )
            self._items[key] = updated
            return updated.model_copy(deep=True)


def negotiate_structured_output(
    capabilities: ModelCapabilities,
    contract: ResponseContract,
) -> StructuredOutputDecision:
    schema_support = capabilities.support_for(ResponseFormat.JSON_SCHEMA)
    object_support = capabilities.support_for(ResponseFormat.JSON_OBJECT)
    candidates: list[ResponseFormat] = []
    if schema_support is not False:
        candidates.append(ResponseFormat.JSON_SCHEMA)
    if contract.strictness != "required" and object_support is not False:
        candidates.append(ResponseFormat.JSON_OBJECT)
    if contract.strictness == "best_effort":
        candidates.append(ResponseFormat.TEXT)
    if not candidates:
        raise ModelCapabilityUnsatisfiedError(
            "no configured model response format can satisfy the structured output contract"
        )
    return StructuredOutputDecision(
        requested_format=ResponseFormat.JSON_SCHEMA,
        candidate_formats=tuple(dict.fromkeys(candidates)),
        capability_source=capabilities.source,
    )


def response_format_payload(
    response_format: ResponseFormat,
    contract: ResponseContract,
) -> dict[str, Any] | None:
    if response_format == ResponseFormat.JSON_SCHEMA:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": contract.name,
                "strict": True,
                "schema": contract.schema_,
            },
        }
    if response_format == ResponseFormat.JSON_OBJECT:
        return {"type": "json_object"}
    return None


def structured_output_prompt(
    system_prompt: str,
    contract: ResponseContract,
    response_format: ResponseFormat,
) -> str:
    if response_format == ResponseFormat.JSON_SCHEMA:
        return system_prompt
    schema = json.dumps(contract.schema_, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{system_prompt}\nReturn one JSON object matching this JSON Schema exactly: {schema}. "
        "Do not wrap the object in Markdown."
    )


def response_format_not_supported(exc: BaseException) -> bool:
    message = str(exc or "").lower()
    return (
        "response_format" in message
        and any(token in message for token in ("not supported", "unsupported", "not valid"))
        and any(token in message for token in ("json_schema", "json_object"))
    )


def model_capabilities_for_provider(provider: Any) -> ModelCapabilities:
    declared = getattr(provider, "model_capabilities", None)
    if callable(declared):
        declared = declared()
    if isinstance(declared, ModelCapabilities):
        return declared.model_copy(deep=True)
    if isinstance(declared, dict):
        try:
            known = {
                key: value
                for key, value in declared.items()
                if key in ModelCapabilities.model_fields
            }
            return ModelCapabilities.model_validate(known)
        except (TypeError, ValueError):
            pass
    return ModelCapabilities(source="runtime_unknown")
