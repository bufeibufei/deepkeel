import json

import pytest

from harness_core.model_capabilities import (
    InMemoryModelCapabilityRegistry,
    ModelCapabilities,
    ModelCapabilityUnsatisfiedError,
    ResponseContract,
    ResponseFormat,
    negotiate_structured_output,
)
from harness_core.model import ModelInvocation, NativeChatProviderAdapter
from harness_core.subagents.execution_support import _invoke_provider


SCHEMA = {
    "type": "object",
    "properties": {"conclusion": {"type": "string"}},
    "required": ["conclusion"],
    "additionalProperties": False,
}


class RuntimeLearningProvider:
    model = "dynamic-model"
    base_url = "https://provider.example/v1"

    def __init__(self) -> None:
        self.formats: list[str] = []

    def complete(self, _system_prompt, _user_prompt, **kwargs):
        mode = (kwargs.get("response_format") or {}).get("type") or "text"
        self.formats.append(mode)
        if mode == "json_schema":
            raise RuntimeError(
                "response_format.type json_schema is not supported by this model"
            )
        return json.dumps({"conclusion": "compatible output"})


def test_structured_output_learns_unsupported_schema_and_reuses_json_object() -> None:
    provider = RuntimeLearningProvider()
    registry = InMemoryModelCapabilityRegistry()

    first = _invoke_provider(
        provider,
        "Return JSON.",
        "Review the input.",
        timeout_seconds=30,
        max_tokens=500,
        output_schema=SCHEMA,
        capability_registry=registry,
    )
    second = _invoke_provider(
        provider,
        "Return JSON.",
        "Review another input.",
        timeout_seconds=30,
        max_tokens=500,
        output_schema=SCHEMA,
        capability_registry=registry,
    )

    assert first.effective_format == ResponseFormat.JSON_OBJECT
    assert first.diagnostics()["degraded"] is True
    assert second.effective_format == ResponseFormat.JSON_OBJECT
    assert provider.formats == ["json_schema", "json_object", "json_object"]


def test_declared_capability_skips_known_unsupported_schema() -> None:
    provider = RuntimeLearningProvider()
    provider.model_capabilities = ModelCapabilities(
        supported_response_formats={ResponseFormat.TEXT, ResponseFormat.JSON_OBJECT},
        unsupported_response_formats={ResponseFormat.JSON_SCHEMA},
        source="provider_catalog",
    )

    result = _invoke_provider(
        provider,
        "Return JSON.",
        "Review the input.",
        timeout_seconds=30,
        max_tokens=500,
        output_schema=SCHEMA,
        capability_registry=InMemoryModelCapabilityRegistry(),
    )

    assert result.effective_format == ResponseFormat.JSON_OBJECT
    assert provider.formats == ["json_object"]


def test_required_schema_rejects_model_known_not_to_support_it() -> None:
    capabilities = ModelCapabilities(
        unsupported_response_formats={ResponseFormat.JSON_SCHEMA},
        source="provider_catalog",
    )
    contract = ResponseContract(name="required", schema=SCHEMA, strictness="required")

    with pytest.raises(ModelCapabilityUnsatisfiedError):
        negotiate_structured_output(capabilities, contract)


class DeclaredChatProvider:
    model = "compatible-chat"
    base_url = "https://provider.example/v1"
    model_capabilities = {
        "supported_response_formats": {"text", "json_object"},
        "unsupported_response_formats": {"json_schema"},
        "source": "provider_catalog",
    }

    def __init__(self) -> None:
        self.formats: list[str] = []

    def complete_chat(self, _messages, **kwargs):
        self.formats.append((kwargs.get("response_format") or {}).get("type") or "text")
        return {
            "message": {"content": json.dumps({"conclusion": "adapter output"})},
            "finish_reason": "stop",
        }


def test_native_provider_adapter_negotiates_response_contract() -> None:
    provider = DeclaredChatProvider()
    adapter = NativeChatProviderAdapter(provider)

    turn = adapter.invoke(
        ModelInvocation(
            messages=[{"role": "user", "content": "Review"}],
            response_contract=ResponseContract(name="review", schema=SCHEMA),
        )
    )

    assert provider.formats == ["json_object"]
    assert turn.raw["structured_output"]["effective_format"] == "json_object"
    assert adapter.info.capabilities.source == "runtime_observation"
