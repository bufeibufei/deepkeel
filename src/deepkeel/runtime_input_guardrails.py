from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from deepkeel.contracts import DataProvenance
from deepkeel.guardrails import (
    GuardrailAction,
    GuardrailAudit,
    GuardrailRequest,
    GuardrailRunner,
    GuardrailStage,
)
from deepkeel.runtime_api import RuntimeRequest
from deepkeel.type_narrowing import as_dict


@dataclass(frozen=True, slots=True)
class InputGuardrailOutcome:
    request: RuntimeRequest
    audits: tuple[GuardrailAudit, ...] = ()
    error: str = ""


async def guard_runtime_input(
    runner: GuardrailRunner,
    request: RuntimeRequest,
) -> InputGuardrailOutcome:
    if not runner.has_stage(GuardrailStage.INPUT):
        return InputGuardrailOutcome(request=request)
    fingerprint = _request_fingerprint(request)
    result = await runner.arun(
        GuardrailRequest(
            stage=GuardrailStage.INPUT,
            operation_id=f"input:{fingerprint}",
            run_id=request.run_id,
            thread_id=request.thread_id,
            turn_id=request.turn_id,
            user_id=str(request.runtime_scope.user_id or "local-device"),
            tenant_id=request.runtime_scope.tenant_id,
            skill_id=str(as_dict(request.skill_activation).get("skill_id") or ""),
            payload={
                "question": request.question,
                "input_parts": [part.model_dump(mode="json") for part in request.input_parts],
            },
            provenance=(
                DataProvenance(
                    origin="user",
                    source_id=request.turn_id or fingerprint,
                    trust_level="untrusted",
                ),
            ),
            metadata={"namespace": request.runtime_scope.namespace},
        )
    )
    decision = result.decision
    if decision.action == GuardrailAction.BLOCK:
        return InputGuardrailOutcome(
            request=request,
            audits=result.audits,
            error=decision.reason or "input blocked by runtime guardrail",
        )
    if decision.action == GuardrailAction.REQUIRE_APPROVAL:
        return InputGuardrailOutcome(
            request=request,
            audits=result.audits,
            error=(
                decision.reason
                or "input approval is required before a run may be created"
            ),
        )
    patch = dict(decision.payload_patch)
    changes: dict[str, Any] = {}
    if "question" in patch:
        changes["question"] = str(patch["question"])
    if isinstance(patch.get("input_parts"), list):
        changes["input_parts"] = patch["input_parts"]
    guarded_request = (
        RuntimeRequest.model_validate({**request.model_dump(mode="json"), **changes})
        if changes
        else request
    )
    return InputGuardrailOutcome(request=guarded_request, audits=result.audits)


def _request_fingerprint(request: RuntimeRequest) -> str:
    value = (
        f"{request.runtime_scope.storage_key}|{request.run_id}|{request.thread_id}|"
        f"{request.turn_id}|{request.question}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


__all__ = ["InputGuardrailOutcome", "guard_runtime_input"]
