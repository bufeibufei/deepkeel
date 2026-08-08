from __future__ import annotations

import asyncio

from deepkeel.adapter_sdk import acertify_model_provider
from deepkeel.contracts import ToolCall
from deepkeel.model_capabilities import ModelCapabilities, ResponseFormat
from deepkeel.model_invocations import ModelInvocation, ModelProviderInfo, ModelTurn


class CertifiedAsyncProvider:
    info = ModelProviderInfo(
        provider_id="certification.fixture",
        model_id="fixture-v1",
        supports_streaming=True,
        supports_native_tools=True,
        capabilities=ModelCapabilities(
            supports_streaming=True,
            supports_native_tools=True,
            supports_forced_tool_choice=True,
            supported_response_formats={ResponseFormat.TEXT, ResponseFormat.JSON_SCHEMA},
            source="fixture",
        ),
    )

    async def ainvoke(self, request: ModelInvocation, *, on_text_delta=None) -> ModelTurn:
        if request.tools:
            return ModelTurn(
                tool_calls=[
                    ToolCall(
                        id="certification-call",
                        name="certification_echo",
                        arguments={"value": "ready"},
                    )
                ],
                finish_reason="tool_calls",
            )
        content = '{"ready": true}' if request.response_contract is not None else "ready"
        if on_text_delta is not None:
            on_text_delta(content)
        return ModelTurn(content=content, finish_reason="stop")


class BrokenToolProvider(CertifiedAsyncProvider):
    async def ainvoke(self, request: ModelInvocation, *, on_text_delta=None) -> ModelTurn:
        del on_text_delta
        if request.tools:
            return ModelTurn(content="I did not call the tool", finish_reason="stop")
        return ModelTurn(content="ready", finish_reason="stop")


def test_model_provider_live_certification_exercises_declared_contracts() -> None:
    report = asyncio.run(
        acertify_model_provider(CertifiedAsyncProvider(), live_probe=True)
    )

    assert report.certified is True
    assert report.issues == ()


def test_model_provider_certification_rejects_false_tool_capability() -> None:
    report = asyncio.run(acertify_model_provider(BrokenToolProvider(), live_probe=True))

    assert report.certified is False
    assert any(issue.code == "NATIVE_TOOL_PROBE_FAILED" for issue in report.issues)
