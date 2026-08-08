from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal

from jsonschema import Draft202012Validator

from deepkeel.model_capabilities import ResponseContract, ResponseFormat
from deepkeel.model_invocations import ModelInvocation, ModelTurn
from deepkeel.model_provider_contracts import (
    AsyncModelProviderAdapter,
    ModelProviderAdapter,
)
from deepkeel.model_provider_execution import _ainvoke_provider


CertificationSeverity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class ModelProviderCertificationIssue:
    code: str
    message: str
    severity: CertificationSeverity = "error"

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class ModelProviderCertificationReport:
    provider_id: str
    model_id: str
    live_probe: bool
    issues: tuple[ModelProviderCertificationIssue, ...] = ()

    @property
    def certified(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "certified": self.certified,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "live_probe": self.live_probe,
            "issues": [issue.as_dict() for issue in self.issues],
        }


async def acertify_model_provider(
    provider: ModelProviderAdapter | AsyncModelProviderAdapter,
    *,
    live_probe: bool = False,
    timeout_seconds: int = 30,
) -> ModelProviderCertificationReport:
    """Certify the stable provider contract, optionally exercising advertised features."""

    info = provider.info
    issues: list[ModelProviderCertificationIssue] = []
    if not str(info.provider_id or "").strip():
        issues.append(
            ModelProviderCertificationIssue("PROVIDER_ID_MISSING", "provider_id is required")
        )
    if not str(info.model_id or "").strip():
        issues.append(ModelProviderCertificationIssue("MODEL_ID_MISSING", "model_id is required"))
    if not isinstance(provider, (ModelProviderAdapter, AsyncModelProviderAdapter)):
        issues.append(
            ModelProviderCertificationIssue(
                "PROVIDER_PROTOCOL_INVALID",
                "provider must implement invoke() or ainvoke() with ModelInvocation",
            )
        )
    capabilities = info.capabilities
    if capabilities.supports_native_tools is not None and (
        capabilities.supports_native_tools != info.supports_native_tools
    ):
        issues.append(
            ModelProviderCertificationIssue(
                "CAPABILITY_DECLARATION_CONFLICT",
                "info.supports_native_tools conflicts with capabilities.supports_native_tools",
            )
        )
    if not live_probe or any(issue.severity == "error" for issue in issues):
        return ModelProviderCertificationReport(
            provider_id=info.provider_id,
            model_id=info.model_id,
            live_probe=live_probe,
            issues=tuple(issues),
        )

    deltas: list[str] = []
    try:
        text_turn = await _ainvoke_provider(
            provider,
            ModelInvocation(
                messages=[{"role": "user", "content": "Reply with the word ready."}],
                request_timeout=timeout_seconds,
                max_output_tokens=32,
            ),
            on_text_delta=deltas.append,
        )
        _validate_turn(text_turn, issues)
        if info.supports_streaming and not deltas:
            issues.append(
                ModelProviderCertificationIssue(
                    "STREAMING_NOT_OBSERVED",
                    "provider declared streaming but the probe emitted no deltas",
                    severity="warning",
                )
            )
    except Exception as exc:
        issues.append(
            ModelProviderCertificationIssue(
                "TEXT_PROBE_FAILED",
                f"text invocation failed: {type(exc).__name__}: {exc}",
            )
        )

    if info.supports_native_tools:
        await _probe_native_tools(provider, timeout_seconds, issues)
    if ResponseFormat.JSON_SCHEMA in capabilities.supported_response_formats:
        await _probe_json_schema(provider, timeout_seconds, issues)
    return ModelProviderCertificationReport(
        provider_id=info.provider_id,
        model_id=info.model_id,
        live_probe=True,
        issues=tuple(issues),
    )


def certify_model_provider(
    provider: ModelProviderAdapter | AsyncModelProviderAdapter,
    *,
    live_probe: bool = False,
    timeout_seconds: int = 30,
) -> ModelProviderCertificationReport:
    return asyncio.run(
        acertify_model_provider(
            provider,
            live_probe=live_probe,
            timeout_seconds=timeout_seconds,
        )
    )


async def _probe_native_tools(
    provider: ModelProviderAdapter | AsyncModelProviderAdapter,
    timeout_seconds: int,
    issues: list[ModelProviderCertificationIssue],
) -> None:
    try:
        turn = await _ainvoke_provider(
            provider,
            ModelInvocation(
                messages=[{"role": "user", "content": "Call certification_echo."}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "certification_echo",
                            "description": "Certification probe",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                                "required": ["value"],
                            },
                        },
                    }
                ],
                tool_choice={
                    "type": "function",
                    "function": {"name": "certification_echo"},
                },
                request_timeout=timeout_seconds,
                max_output_tokens=64,
            ),
        )
        if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != "certification_echo":
            raise ValueError("forced tool call was not returned exactly once")
    except Exception as exc:
        issues.append(
            ModelProviderCertificationIssue(
                "NATIVE_TOOL_PROBE_FAILED",
                f"native tool invocation failed: {type(exc).__name__}: {exc}",
            )
        )


async def _probe_json_schema(
    provider: ModelProviderAdapter | AsyncModelProviderAdapter,
    timeout_seconds: int,
    issues: list[ModelProviderCertificationIssue],
) -> None:
    schema = {
        "type": "object",
        "properties": {"ready": {"type": "boolean"}},
        "required": ["ready"],
        "additionalProperties": False,
    }
    try:
        turn = await _ainvoke_provider(
            provider,
            ModelInvocation(
                messages=[{"role": "user", "content": "Return readiness as JSON."}],
                response_contract=ResponseContract(name="certification", schema=schema),
                request_timeout=timeout_seconds,
                max_output_tokens=64,
            ),
        )
        payload = json.loads(turn.content)
        Draft202012Validator(schema).validate(payload)
    except Exception as exc:
        issues.append(
            ModelProviderCertificationIssue(
                "JSON_SCHEMA_PROBE_FAILED",
                f"JSON Schema invocation failed: {type(exc).__name__}: {exc}",
            )
        )


def _validate_turn(
    turn: ModelTurn,
    issues: list[ModelProviderCertificationIssue],
) -> None:
    if not isinstance(turn, ModelTurn):
        issues.append(
            ModelProviderCertificationIssue(
                "TURN_TYPE_INVALID",
                "provider did not return ModelTurn",
            )
        )
    elif not turn.content and not turn.tool_calls:
        issues.append(
            ModelProviderCertificationIssue(
                "TURN_EMPTY",
                "provider returned neither content nor tool calls",
            )
        )
