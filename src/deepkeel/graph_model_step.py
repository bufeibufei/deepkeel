from __future__ import annotations

from typing import Any, Iterable

from deepkeel.contracts import ToolCall
from deepkeel.model_invocations import ModelTurn
from deepkeel.model_routing import ModelStepContext
from deepkeel.type_narrowing import as_dict


def build_model_step_context(
    state: dict[str, Any],
    *,
    available_roles: Iterable[str],
    forced_tool_name: str,
    deadline_monotonic: float | None,
) -> ModelStepContext:
    return ModelStepContext(
        run_id=str(state.get("run_id") or ""),
        user_id=str(state.get("user_id") or ""),
        thread_id=str(state.get("thread_id") or ""),
        turn_id=str(state.get("turn_id") or ""),
        step_index=int(state.get("step_count") or 0),
        message_count=len(state.get("messages") or []),
        observation_count=len(state.get("observations") or []),
        tool_result_count=len(state.get("tool_results") or []),
        available_roles=tuple(str(role) for role in available_roles),
        observation_sources=tuple(
            str(item.get("source") or "")
            for item in (state.get("observations") or [])
            if isinstance(item, dict)
        ),
        tool_result_names=tuple(
            str(item.get("name") or "")
            for item in (state.get("tool_results") or [])
            if isinstance(item, dict)
        ),
        model_policy=dict(state.get("model_policy") or {}),
        skill_activation=dict(state.get("skill_activation") or {}),
        policy_phase=str(state.get("policy_phase") or ""),
        forced_tool_name=forced_tool_name,
        governance_scope=dict(as_dict(state.get("metadata")).get("governance_scope") or {}),
        deadline_monotonic=deadline_monotonic,
        operational_run_id=str(
            as_dict(state.get("metadata")).get("operational_run_id") or ""
        ),
    )


def build_model_metrics(
    turn: ModelTurn,
    route_payload: dict[str, Any],
    *,
    latency_ms: int,
    first_token_latency_ms: int | None,
    delta_count: int,
    delta_chars: int,
    forced_tool_name: str,
) -> dict[str, Any]:
    return {
        "model_id": turn.model_id,
        "model_role": turn.model_role,
        "provider_id": str(route_payload.get("provider_id") or ""),
        "finish_reason": turn.finish_reason,
        "latency_ms": latency_ms,
        "first_token_latency_ms": first_token_latency_ms,
        "delta_count": delta_count,
        "delta_chars": delta_chars,
        "content_chars": len(turn.content),
        "tool_call_count": len(turn.tool_calls),
        "route_reason": str(route_payload.get("reason") or ""),
        "router_id": str(route_payload.get("router_id") or ""),
        "policy": as_dict(route_payload.get("policy")),
        "budget": as_dict(route_payload.get("budget")),
        "budget_metrics": as_dict(route_payload.get("budget_metrics")),
        "usage": as_dict(route_payload.get("usage")),
        "max_output_tokens": route_payload.get("max_output_tokens"),
        "forced_tool_name": forced_tool_name,
    }


def partition_model_tool_calls(
    reported_tool_calls: list[ToolCall],
    *,
    workflow_is_finalizing: bool,
    exposed_tool_names: set[str],
    registered_tool_names: set[str],
) -> tuple[list[ToolCall], list[ToolCall]]:
    accepted = [
        call
        for call in reported_tool_calls
        if not workflow_is_finalizing
        or call.name in exposed_tool_names
        or call.name not in registered_tool_names
    ]
    rejected = [
        call
        for call in reported_tool_calls
        if workflow_is_finalizing
        and call.name in registered_tool_names
        and call.name not in exposed_tool_names
    ]
    return accepted, rejected
