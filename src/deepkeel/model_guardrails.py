from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig

from deepkeel.contracts import DataProvenance, Observation
from deepkeel.graph_workflow import _emit
from deepkeel.guardrails import GuardrailAudit, GuardrailRequest, GuardrailStage
from deepkeel.turn_context import TurnExecutionContext
from deepkeel.type_narrowing import as_dict


def model_guardrail_request(
    stage: GuardrailStage,
    state: Mapping[str, Any],
    turn_context: TurnExecutionContext,
    *,
    payload: Mapping[str, Any],
    operation_suffix: str = "",
) -> GuardrailRequest:
    skill = as_dict(state.get("skill_activation"))
    metadata = as_dict(state.get("metadata"))
    tool_metadata = turn_context.tool_context.metadata
    package_ids = tuple(
        str(value)
        for value in tool_metadata.get("capability_package_ids", ())
        if str(value).strip()
    )
    return GuardrailRequest(
        stage=stage,
        operation_id=(
            f"{state.get('run_id')}:{state.get('turn_id')}:"
            f"model:{int(state.get('step_count') or 0)}:{stage.value}{operation_suffix}"
        ),
        run_id=str(state.get("run_id") or ""),
        thread_id=str(state.get("thread_id") or ""),
        turn_id=str(state.get("turn_id") or ""),
        user_id=turn_context.tool_context.user_id,
        tenant_id=str(tool_metadata.get("tenant_id") or ""),
        package_ids=package_ids,
        skill_id=str(skill.get("skill_id") or ""),
        payload=payload,
        provenance=_state_provenance(state, stage),
        metadata={
            "governance_scope": dict(metadata.get("governance_scope") or {}),
        },
    )


def emit_model_guardrail_audits(
    state: dict[str, Any],
    config: RunnableConfig,
    audits: tuple[GuardrailAudit, ...],
) -> None:
    for audit in audits:
        _emit(
            state,
            config,
            "guardrail.evaluated",
            "Guardrail evaluated",
            f"{audit.stage.value}: {audit.action.value}",
            {
                "guardrail_id": audit.guardrail_id,
                "stage": audit.stage.value,
                "operation_id": audit.operation_id,
                "status": audit.status,
                "action": audit.action.value,
                "duration_ms": audit.duration_ms,
                "replayed": audit.replayed,
                "required": audit.required,
                "reason": audit.reason,
                "error": audit.error,
                "diagnostics": dict(audit.diagnostics),
                "visible": False,
            },
        )


def _state_provenance(
    state: Mapping[str, Any],
    stage: GuardrailStage,
) -> tuple[DataProvenance, ...]:
    values: list[DataProvenance] = []
    if stage in {GuardrailStage.MODEL_INPUT, GuardrailStage.INPUT}:
        for item in state.get("messages") or []:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role") or "")
            if role == "user":
                values.append(
                    DataProvenance(
                        origin="user",
                        source_id=str(item.get("id") or ""),
                        trust_level="untrusted",
                    )
                )
        for item in state.get("observations") or []:
            try:
                values.append(Observation.model_validate(item).provenance)
            except (TypeError, ValueError):
                continue
    else:
        values.append(
            DataProvenance(
                origin="model",
                source_id=str(state.get("run_id") or ""),
                trust_level="untrusted",
            )
        )
    return tuple(values)


__all__ = ["emit_model_guardrail_audits", "model_guardrail_request"]
