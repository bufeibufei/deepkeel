from __future__ import annotations

import time
from typing import Any, Callable

from deepkeel.async_ports import run_sync_adapter
from deepkeel.budget import ELAPSED_SECONDS, BudgetRequest
from deepkeel.graph import HARNESS_GRAPH_CONTRACT_VERSION
from deepkeel.hooks import HookAudit
from deepkeel.runtime_api import RuntimeResult
from deepkeel.runtime_lifecycle import run_settlement_lifecycle_hooks
from deepkeel.runtime_online_evaluation import submit_runtime_online_evaluation
from deepkeel.runtime_policy import _budget_limits
from deepkeel.runtime_results import project_harness_result
from deepkeel.scope import RuntimeScope
from deepkeel.telemetry import TelemetryRecord
from deepkeel.type_narrowing import as_dict


async def _asettle_runtime_budget(
    runtime: Any,
    *,
    operational_run_id: str,
    turn_id: str,
    elapsed_seconds: float,
    model_policy: dict[str, Any],
) -> tuple[Any, Any]:
    elapsed = await run_sync_adapter(
        runtime.budget_ledger.consume,
        BudgetRequest(
            run_id=operational_run_id,
            metric=ELAPSED_SECONDS,
            amount=max(0.0, elapsed_seconds),
            limit=_budget_limits(model_policy).get(ELAPSED_SECONDS),
            operation_id=f"runtime-elapsed:{turn_id}",
            metadata={"turn_id": turn_id},
        ),
    )
    snapshot = await run_sync_adapter(runtime.budget_ledger.snapshot, operational_run_id)
    return elapsed, snapshot


async def project_and_settle_runtime_result(
    *,
    runtime: Any,
    state: dict[str, Any],
    question: str,
    context_bundle: dict[str, Any],
    short_context: dict[str, Any],
    skill_activation: dict[str, Any],
    model_policy: dict[str, Any],
    previous_diagnostics: dict[str, Any],
    context_window_diagnostics: dict[str, Any],
    emitter: Any,
    lifecycle_audits: list[HookAudit],
    package_ids: tuple[str, ...],
    recovery_source: str,
    checkpoint_authority: str,
    checkpoint_load_errors: list[str],
    run_started_monotonic: float,
    run_id: str,
    conversation_thread_id: str,
    turn_id: str,
    user_id: str,
    runtime_scope: RuntimeScope,
    active_graph_thread_id: str,
    session: Any,
    execution_fence: Any,
    emit_hook_audits: Callable[..., None],
) -> RuntimeResult:
    operational_run_id = runtime_scope.qualify_identity(run_id)
    await emitter.flush()
    result = project_harness_result(
        state,
        question=question,
        context_bundle=context_bundle,
        short_context=short_context,
        skill_activation=skill_activation,
        streamed_events=emitter.events,
        user_id=user_id,
        answer_delta_streamed=emitter.answer_delta_streamed,
        observation_kinds={
            spec.name: str(spec.observation_contract.get("primary_kind") or "")
            for spec in runtime.tool_registry.list_tools()
        },
        task_kinds={spec.name: spec.task_kind for spec in runtime.tool_registry.list_tools()},
        max_steps=runtime.max_steps,
        previous_diagnostics=previous_diagnostics,
        capability_manifest=runtime._capability_manifest(),
        reference_projector=runtime.reference_projector,
    )
    diagnostics = result.diagnostics
    if recovery_source:
        recovery = as_dict(diagnostics.get("recovery"))
        recovery["checkpoint_source"] = recovery_source
        diagnostics["recovery"] = recovery
    diagnostics["context_window"] = context_window_diagnostics
    diagnostics["hooks"] = {
        "executions": [runtime._hook_audit_payload(audit) for audit in lifecycle_audits],
    }
    execution_contract = {
        "graph_contract_version": HARNESS_GRAPH_CONTRACT_VERSION,
        "graph_reused": runtime.reuse_compiled_graph,
        "graph_durability": runtime.graph_durability,
        "graph_compile_count": runtime.graph_compile_count,
        "tool_catalog_version": runtime.tool_registry.catalog_version(),
        "tool_view_mode": runtime.tool_view_mode,
        "tool_view": as_dict(as_dict(state.get("metadata")).get("tool_view")),
        "checkpoint_authority": checkpoint_authority,
        "checkpoint_policy": "runtime_boundaries",
        "graph_checkpoint_role": "engine_recovery_only",
        "runtime_generation_id": (
            runtime.runtime_generation.generation_id
            if runtime.runtime_generation is not None
            else ""
        ),
    }
    if checkpoint_load_errors:
        execution_contract["checkpoint_load_errors"] = list(checkpoint_load_errors)
    diagnostics["execution_contract"] = execution_contract
    result.checkpoint["execution_contract"] = execution_contract
    elapsed_budget, budget_snapshot = await _asettle_runtime_budget(
        runtime,
        operational_run_id=operational_run_id,
        turn_id=turn_id,
        elapsed_seconds=time.monotonic() - run_started_monotonic,
        model_policy=model_policy,
    )
    diagnostics["budget"] = {
        "elapsed": elapsed_budget.as_dict(),
        "snapshot": budget_snapshot.as_dict(),
    }
    result.checkpoint["budget_state"] = budget_snapshot.as_dict()
    lifecycle_audits.extend(
        await run_settlement_lifecycle_hooks(
            hook_runner=runtime.hook_runner,
            emit=emitter,
            emit_audits=emit_hook_audits,
            run_id=run_id,
            thread_id=conversation_thread_id,
            turn_id=turn_id,
            package_ids=package_ids,
            skill_id=str(skill_activation.get("skill_id") or ""),
            user_id=user_id,
            status=result.status.value,
            stop_reason=result.stop_reason,
            artifact_ids=[artifact.id for artifact in result.artifacts],
        )
    )
    diagnostics["hooks"]["executions"] = [
        runtime._hook_audit_payload(audit) for audit in lifecycle_audits
    ]
    online_eval_diagnostics = submit_runtime_online_evaluation(
        runtime=runtime,
        result=result,
        runtime_scope=runtime_scope,
    )
    if online_eval_diagnostics is not None:
        diagnostics["online_eval"] = online_eval_diagnostics
    try:
        runtime.telemetry.record(
            TelemetryRecord(
                event_name="runtime.settled",
                run_id=run_id,
                thread_id=conversation_thread_id,
                turn_id=turn_id,
                tenant_id=runtime_scope.tenant_id,
                user_id=runtime_scope.user_id,
                namespace=runtime_scope.namespace,
                status=result.status.value,
                attributes={
                    "status": result.status.value,
                    "stop_reason": result.stop_reason,
                    "skill_id": str(skill_activation.get("skill_id") or ""),
                    "recovery_source": recovery_source,
                },
            )
        )
    except Exception as exc:
        emitter.record_telemetry_error(exc)
    if emitter.telemetry_error_count:
        diagnostics["telemetry"] = {
            "status": "degraded",
            "error_count": emitter.telemetry_error_count,
            "last_error": emitter.telemetry_last_error,
        }
    await emitter.flush()
    terminal = result.status.value in {"completed", "failed", "canceled"}
    await runtime._persist_runtime_snapshot(
        result,
        run_id=run_id,
        thread_id=conversation_thread_id,
        session=session,
        user_id=user_id,
        context_bundle=context_bundle,
        execution_fence=execution_fence,
        scope=runtime_scope,
    )
    if terminal:
        if runtime._host_owns_terminal_settlement():
            recovery = as_dict(diagnostics.get("recovery"))
            recovery["checkpoint_cleanup"] = {
                "status": "deferred",
                "reason": "host_settlement_required",
            }
            diagnostics["recovery"] = recovery
        else:
            await runtime._acleanup_run(
                result,
                run_id=run_id,
                session=session,
                user_id=user_id,
                scope=runtime_scope,
                graph_thread_ids=[active_graph_thread_id],
            )
        await runtime._record_checkpoint_cleanup_event(
            result,
            run_id=run_id,
            thread_id=conversation_thread_id,
            turn_id=turn_id,
            scope=runtime_scope,
            event_sink=emitter.event_sink,
            fallback_sequence=emitter.sequence,
        )
    return result
