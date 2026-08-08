from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from deepkeel.contracts import ToolCall, ToolResult
from deepkeel.scope import scoped_adapter_operation
from deepkeel.tool_execution import ToolExecutionClaim, ToolExecutionContext
from deepkeel.tool_execution_support import (
    _failed_result,
    _replayed_result,
    _with_runtime_metrics,
)


@dataclass(frozen=True, slots=True)
class ToolClaimResolution:
    claim: ToolExecutionClaim | None = None
    result: ToolResult | None = None


async def resolve_tool_claim(
    execution_store: Any,
    call: ToolCall,
    context: ToolExecutionContext,
    *,
    lease_seconds: float,
    max_attempts: int,
    reexecution_safe: bool,
    started_at: float,
) -> ToolClaimResolution:
    try:
        operation = scoped_adapter_operation(execution_store, "claim", context.scope)
        kwargs: dict[str, Any] = {
            "run_id": context.run_id,
            "call": call,
            "lease_seconds": lease_seconds,
            "max_attempts": max_attempts,
            "reexecution_safe": reexecution_safe,
        }
        if getattr(operation, "__name__", "") == "claim_scoped":
            kwargs["scope"] = context.scope
        claim = await operation(**kwargs)
    except Exception as exc:
        failed = _failed_result(call, f"tool execution claim failed: {exc}", retryable=True)
        return ToolClaimResolution(
            result=_with_runtime_metrics(
                failed, started_at, phase="idempotent_claim", executed=False
            )
        )
    result = await _claim_status_result(
        execution_store,
        claim,
        call,
        reexecution_safe=reexecution_safe,
        started_at=started_at,
    )
    return ToolClaimResolution(claim=claim, result=result)


async def settle_tool_claim(
    execution_store: Any,
    claim: ToolExecutionClaim,
    result: ToolResult,
) -> Exception | None:
    error: Exception | None = None
    for attempt in range(3):
        try:
            await execution_store.settle(claim, result)
            return None
        except Exception as exc:
            error = exc
            if attempt < 2:
                await asyncio.sleep(0.05 * (attempt + 1))
    return error


async def _claim_status_result(
    execution_store: Any,
    claim: ToolExecutionClaim,
    call: ToolCall,
    *,
    reexecution_safe: bool,
    started_at: float,
) -> ToolResult | None:
    if claim.status == "replay":
        try:
            persisted = await execution_store.replay(claim)
        except Exception as exc:
            result = _failed_result(
                call,
                "Tool execution recovery failed. Retry the operation.",
                retryable=True,
            )
            result.metadata["replay_error"] = str(exc)
        else:
            result = _replayed_result(persisted, call)
        return _with_runtime_metrics(result, started_at, phase="idempotent_replay", executed=False)
    if claim.status not in {"busy", "blocked", "corrupt", "uncertain", "exhausted"}:
        return None
    result = _claim_failure_result(claim, call, reexecution_safe=reexecution_safe)
    return _with_runtime_metrics(
        result,
        started_at,
        phase=f"idempotent_{claim.status}",
        executed=False,
    )


def _claim_failure_result(
    claim: ToolExecutionClaim,
    call: ToolCall,
    *,
    reexecution_safe: bool,
) -> ToolResult:
    if claim.status == "busy":
        result = _failed_result(call, "tool execution is already in progress", retryable=True)
        result.metadata = {
            "idempotency_busy": True,
            "retry_after_seconds": claim.retry_after_seconds,
            "claim_owner": claim.claim_owner,
        }
        return result
    if claim.status == "blocked":
        result = _failed_result(call, f"agent run is already {claim.terminal_status or 'terminal'}")
        result.metadata = {
            "terminal_run_blocked": True,
            "terminal_status": claim.terminal_status,
        }
        return result
    messages = {
        "corrupt": "The tool execution record is corrupt. The run ended safely; start again.",
        "uncertain": "The prior tool result is uncertain. The run ended safely to avoid duplication.",
        "exhausted": "Tool recovery attempts were exhausted and the run ended safely.",
    }
    result = _failed_result(call, messages[claim.status])
    result.metadata = {
        "durable_execution_status": claim.status,
        "durable_execution_detail": claim.detail,
        "tool_invocation_id": claim.record_id,
        "attempt_count": claim.attempt_count,
        "reexecution_safe": reexecution_safe,
    }
    return result
