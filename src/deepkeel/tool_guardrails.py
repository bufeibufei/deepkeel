from __future__ import annotations

from typing import Any, Mapping

from pydantic import ValidationError

from deepkeel.contracts import (
    Artifact,
    DataProvenance,
    Observation,
    PendingAction,
    ToolCall,
    ToolResult,
)
from deepkeel.guardrails import (
    GuardrailAction,
    GuardrailAudit,
    GuardrailDecision,
    GuardrailRequest,
    GuardrailRunner,
    GuardrailStage,
)
from deepkeel.tool_execution import ToolExecutionContext
from deepkeel.type_narrowing import as_dict


async def guard_tool_input(
    runner: GuardrailRunner | None,
    call: ToolCall,
    context: ToolExecutionContext,
) -> tuple[ToolCall, ToolResult | None]:
    if runner is None or not runner.has_stage(GuardrailStage.TOOL_INPUT):
        return call, None
    result = await runner.arun(
        _request(
            GuardrailStage.TOOL_INPUT,
            call,
            context,
            payload={
                "tool_name": call.name,
                "arguments": dict(call.arguments),
                "read_only": call.read_only,
            },
            provenance=(
                DataProvenance(
                    origin="model",
                    source_id=call.id,
                    trust_level="untrusted",
                ),
            ),
        )
    )
    emit_guardrail_audits(context, result.audits)
    decision = result.decision
    if decision.action == GuardrailAction.BLOCK:
        return call, _blocked_result(call, decision, stage=GuardrailStage.TOOL_INPUT)
    if decision.action == GuardrailAction.REQUIRE_APPROVAL:
        return call, _approval_result(
            call,
            context,
            decision,
            stage=GuardrailStage.TOOL_INPUT,
        )
    patch = as_dict(decision.payload_patch.get("arguments"))
    if patch:
        call = call.model_copy(update={"arguments": {**call.arguments, **patch}})
    return call, None


async def guard_tool_output(
    runner: GuardrailRunner | None,
    call: ToolCall,
    context: ToolExecutionContext,
    result: ToolResult,
) -> ToolResult:
    result = _attach_tool_provenance(call, result)
    if runner is None:
        return result
    if runner.has_stage(GuardrailStage.TOOL_OUTPUT):
        outcome = await runner.arun(
            _request(
                GuardrailStage.TOOL_OUTPUT,
                call,
                context,
                payload={"result": result.model_dump(mode="json")},
                provenance=_result_provenance(result),
            )
        )
        emit_guardrail_audits(context, outcome.audits)
        result = _apply_result_decision(
            call,
            context,
            result,
            outcome.decision,
            stage=GuardrailStage.TOOL_OUTPUT,
        )
        if result.status in {"failed", "requires_user_action"} and result.metadata.get(
            "guardrail_terminal"
        ):
            return result
    if not result.artifacts or not runner.has_stage(GuardrailStage.ARTIFACT_OUTPUT):
        return result
    guarded_artifacts: list[Artifact] = []
    for artifact in result.artifacts:
        outcome = await runner.arun(
            _request(
                GuardrailStage.ARTIFACT_OUTPUT,
                call,
                context,
                payload={"artifact": artifact.model_dump(mode="json")},
                provenance=(artifact.provenance,),
                operation_suffix=f":artifact:{artifact.id}",
            )
        )
        emit_guardrail_audits(context, outcome.audits)
        decision = outcome.decision
        if decision.action == GuardrailAction.BLOCK:
            return _blocked_result(call, decision, stage=GuardrailStage.ARTIFACT_OUTPUT)
        if decision.action == GuardrailAction.REQUIRE_APPROVAL:
            return _approval_result(
                call,
                context,
                decision,
                stage=GuardrailStage.ARTIFACT_OUTPUT,
            )
        patch = as_dict(decision.payload_patch.get("artifact"))
        try:
            guarded_artifacts.append(
                Artifact.model_validate({**artifact.model_dump(mode="json"), **patch})
            )
        except ValidationError as exc:
            return _invalid_transform_result(
                call,
                stage=GuardrailStage.ARTIFACT_OUTPUT,
                error=str(exc),
            )
    return result.model_copy(update={"artifacts": guarded_artifacts})


def emit_guardrail_audits(
    context: ToolExecutionContext,
    audits: tuple[GuardrailAudit, ...],
) -> None:
    sink = context.metadata.get("event_sink")
    if not callable(sink):
        return
    for audit in audits:
        sink(
            {
                "event_type": "guardrail.evaluated",
                "title": "Guardrail evaluated",
                "summary": f"{audit.stage.value}: {audit.action.value}",
                "payload": {
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
                },
                "visibility": "debug",
            }
        )


def _request(
    stage: GuardrailStage,
    call: ToolCall,
    context: ToolExecutionContext,
    *,
    payload: Mapping[str, Any],
    provenance: tuple[DataProvenance, ...],
    operation_suffix: str = "",
) -> GuardrailRequest:
    skill = as_dict(context.metadata.get("skill_activation"))
    package_ids = tuple(
        str(value)
        for value in context.metadata.get("capability_package_ids", ())
        if str(value).strip()
    )
    return GuardrailRequest(
        stage=stage,
        operation_id=(
            f"{context.run_id}:{context.turn_id}:tool:{call.id}:{stage.value}{operation_suffix}"
        ),
        run_id=context.run_id,
        thread_id=context.thread_id,
        turn_id=context.turn_id,
        user_id=context.user_id,
        tenant_id=str(context.metadata.get("tenant_id") or ""),
        package_ids=package_ids,
        skill_id=str(skill.get("skill_id") or ""),
        tool_name=call.name,
        payload=payload,
        provenance=provenance,
        metadata={
            "governance_scope": context.metadata.get("governance_scope") or {},
        },
    )


def _attach_tool_provenance(call: ToolCall, result: ToolResult) -> ToolResult:
    default = DataProvenance(
        origin="tool",
        source_id=call.name,
        trust_level="external",
        parent_ids=[call.id],
    )
    observation = result.observation
    if observation is not None and observation.provenance.origin == "unknown":
        observation = observation.model_copy(update={"provenance": default})
    artifacts = [
        artifact.model_copy(update={"provenance": default})
        if artifact.provenance.origin == "unknown"
        else artifact
        for artifact in result.artifacts
    ]
    return result.model_copy(update={"observation": observation, "artifacts": artifacts})


def _result_provenance(result: ToolResult) -> tuple[DataProvenance, ...]:
    values: list[DataProvenance] = []
    if result.observation is not None:
        values.append(result.observation.provenance)
    values.extend(artifact.provenance for artifact in result.artifacts)
    return tuple(values)


def _apply_result_decision(
    call: ToolCall,
    context: ToolExecutionContext,
    result: ToolResult,
    decision: GuardrailDecision,
    *,
    stage: GuardrailStage,
) -> ToolResult:
    if decision.action == GuardrailAction.BLOCK:
        return _blocked_result(call, decision, stage=stage, prior=result)
    if decision.action == GuardrailAction.REQUIRE_APPROVAL:
        return _approval_result(call, context, decision, stage=stage, prior=result)
    patch = as_dict(decision.payload_patch.get("result"))
    if not patch:
        return result
    try:
        transformed = ToolResult.model_validate(
            {**result.model_dump(mode="json"), **patch}
        )
    except ValidationError as exc:
        return _invalid_transform_result(call, stage=stage, error=str(exc), prior=result)
    return transformed.model_copy(update={"call": call})


def _blocked_result(
    call: ToolCall,
    decision: GuardrailDecision,
    *,
    stage: GuardrailStage,
    prior: ToolResult | None = None,
) -> ToolResult:
    metadata = dict(prior.metadata) if prior is not None else {}
    metadata.update(
        {
            "guardrail_terminal": True,
            "guardrail": {**decision.as_dict(), "stage": stage.value},
            "executed": stage != GuardrailStage.TOOL_INPUT,
        }
    )
    return ToolResult(
        call=call,
        status="failed",
        outcome="degraded",
        summary=decision.reason or "Tool data was blocked by a runtime guardrail.",
        error=decision.reason or "guardrail blocked tool data",
        metadata=metadata,
    )


def _approval_result(
    call: ToolCall,
    context: ToolExecutionContext,
    decision: GuardrailDecision,
    *,
    stage: GuardrailStage,
    prior: ToolResult | None = None,
) -> ToolResult:
    pending = PendingAction(
        id=f"guardrail-confirmation:{stage.value}:{call.id}",
        run_id=context.run_id,
        tool_call_id=call.id,
        action_type="confirm_guarded_operation",
        title=decision.approval_title or "Confirm protected operation",
        prompt=(
            decision.approval_prompt
            or decision.reason
            or "Please confirm before this protected operation continues."
        ),
        payload={
            "source": "guardrail",
            "stage": stage.value,
            "tool_name": call.name,
            "guardrail_code": decision.code,
        },
    )
    metadata = dict(prior.metadata) if prior is not None else {}
    metadata.update(
        {
            "guardrail_terminal": True,
            "guardrail": {**decision.as_dict(), "stage": stage.value},
            "executed": stage != GuardrailStage.TOOL_INPUT,
        }
    )
    return ToolResult(
        call=call,
        status="requires_user_action",
        outcome="partial",
        summary=pending.prompt,
        pending_action=pending,
        metadata=metadata,
    )


def _invalid_transform_result(
    call: ToolCall,
    *,
    stage: GuardrailStage,
    error: str,
    prior: ToolResult | None = None,
) -> ToolResult:
    decision = GuardrailDecision(
        action=GuardrailAction.BLOCK,
        reason="A guardrail returned an invalid transformed payload.",
        code="GUARDRAIL_TRANSFORM_INVALID",
        diagnostics={"validation_error": error},
    )
    return _blocked_result(call, decision, stage=stage, prior=prior)


__all__ = ["emit_guardrail_audits", "guard_tool_input", "guard_tool_output"]
