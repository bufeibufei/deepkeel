from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepkeel.type_narrowing import as_dict, as_dict_list, as_list


def project_observation(
    item: dict[str, Any],
    observation_kinds: dict[str, str],
) -> dict[str, Any]:
    projected = dict(item)
    source = str(projected.get("source") or projected.get("tool_name") or "")
    data = as_dict(projected.get("data"))
    kind = str(
        observation_kinds.get(source)
        or data.get("kind")
        or data.get("observation_kind")
        or (source.split(".", 1)[0] if "." in source else source)
        or "tool"
    )
    projected.setdefault("tool_name", source)
    projected.setdefault("kind", kind)
    return projected


def trace_from_events(
    events: list[dict[str, Any]],
    *,
    observation_kinds: dict[str, str],
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        action = str(event.get("event_type") or "")
        if action == "answer.delta":
            continue
        payload = as_dict(event.get("payload"))
        item: dict[str, Any] = {
            "index": int(event.get("sequence") or index + 1),
            "action": action,
            "summary": str(event.get("summary") or ""),
            "created_at": str(event.get("created_at") or ""),
        }
        if action in {"model.completed", "model.failed"}:
            item.update(_model_trace_fields(action, payload))
        elif action == "model.route.selected":
            item.update(_route_trace_fields(payload))
        elif _is_tool_result_action(action):
            item.update(_tool_trace_fields(payload, observation_kinds))
        trace.append(item)
    return trace


def _model_trace_fields(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    latency_ms = int(payload.get("latency_ms") or 0)
    return {
        "model_role": str(payload.get("model_role") or ""),
        "model_id": str(payload.get("model_id") or ""),
        "status": "failed" if action == "model.failed" else "completed",
        "latency_ms": latency_ms,
        "answer_stream": {
            "first_token_latency_ms": payload.get("first_token_latency_ms"),
            "latency_ms": latency_ms,
            "delta_count": int(payload.get("delta_count") or 0),
            "delta_chars": int(payload.get("delta_chars") or 0),
        },
        **_route_fields(payload),
    }


def _route_trace_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_role": str(payload.get("role") or ""),
        "model_id": str(payload.get("model_id") or ""),
        **_route_fields(payload),
        "budget_metrics": as_dict(payload.get("budget_metrics")),
        "usage": as_dict(payload.get("usage")),
        "max_output_tokens": payload.get("max_output_tokens"),
    }


def _route_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_reason": str(payload.get("reason") or payload.get("route_reason") or ""),
        "router_id": str(payload.get("router_id") or ""),
        "policy": as_dict(payload.get("policy")),
        "budget": as_dict(payload.get("budget")),
        "invocation": as_dict(payload.get("invocation")),
    }


def _is_tool_result_action(action: str) -> bool:
    return (action.startswith("tool.call.") and action != "tool.call.started") or action == "run.waiting_async"


def _tool_trace_fields(
    payload: dict[str, Any],
    observation_kinds: dict[str, str],
) -> dict[str, Any]:
    tool_result = as_dict(payload.get("tool_result"))
    metadata = as_dict(tool_result.get("metadata"))
    metrics = as_dict(tool_result.get("runtime_metrics") or metadata.get("runtime_metrics"))
    governance = as_dict(metadata.get("governance"))
    tool_call = as_dict(payload.get("tool_call"))
    observation = tool_result.get("observation")
    return {
        "latency_ms": int(metrics.get("latency_ms") or payload.get("latency_ms") or 0),
        "status": str(tool_result.get("status") or ""),
        "outcome": str(tool_result.get("outcome") or ""),
        "diagnostics": as_dict(tool_result.get("diagnostics")),
        "tool_calls": [
            {"tool_name": str(tool_result.get("name") or tool_call.get("name") or "")}
        ],
        "observations": [project_observation(observation, observation_kinds)]
        if isinstance(observation, dict)
        else [],
        "policy": as_dict(governance.get("policy")),
        "budget": as_dict(governance.get("budget")),
        "artifact_types": sorted(
            {
                str(artifact.get("artifact_type") or "")
                for artifact in tool_result.get("artifacts") or []
                if isinstance(artifact, dict) and str(artifact.get("artifact_type") or "")
            }
        ),
        "artifact_contract_failed": bool(metadata.get("artifact_contract_failed")),
    }


@dataclass(frozen=True, slots=True)
class _DiagnosticRows:
    trace: list[dict[str, Any]]
    model: list[dict[str, Any]]
    routes: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    confirmations: list[dict[str, Any]]


def runtime_diagnostics(
    state: dict[str, Any],
    runtime: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    context_snapshot: dict[str, Any],
    skill_activation: dict[str, Any],
    max_steps: int,
    previous_diagnostics: dict[str, Any] | None = None,
    capability_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous = as_dict(previous_diagnostics)
    rows = _diagnostic_rows(runtime)
    context_selection = as_dict(context_snapshot.get("context_selection"))
    budget_policy = as_dict(as_dict(state.get("model_policy")).get("budget"))
    return {
        "schema_version": "harness-runtime-diagnostics-v1",
        "context": {
            "context_version": str(context_snapshot.get("schema_version") or ""),
            "selection_strategy": str(context_selection.get("strategy") or "harness_context_v1"),
        },
        "capabilities": dict(capability_manifest or {}),
        "loop": {
            "status": str(runtime.get("status") or ""),
            "stop_reason": str(runtime.get("stop_reason") or ""),
            "step_count": int(runtime.get("step_count") or 0),
            "max_steps": int(max_steps),
            "max_tool_calls_per_step": 0,
            "max_elapsed_seconds": float(budget_policy.get("max_elapsed_seconds") or 0),
        },
        "counts": _diagnostic_counts(state, events, rows, previous),
        "recovery": {
            **as_dict(previous.get("recovery")),
            "resume_mode": str(as_dict(runtime.get("pending_action")).get("resume_mode") or ""),
            "pending_action": runtime.get("pending_action") or {},
        },
        "governance": _governance_diagnostics(state, rows, previous),
        "skill": _skill_diagnostics(state, runtime, skill_activation, rows, previous),
        "replay": {"prompt_hashes": [], "model_output_hashes": []},
        "timings": _timing_diagnostics(rows, previous),
    }


def _diagnostic_rows(runtime: dict[str, Any]) -> _DiagnosticRows:
    trace = as_dict_list(runtime.get("trace"))
    return _DiagnosticRows(
        trace=trace,
        model=[item for item in trace if item.get("action") in {"model.completed", "model.failed"}],
        routes=[item for item in trace if item.get("action") == "model.route.selected"],
        tools=[item for item in trace if _is_tool_result_action(str(item.get("action") or ""))],
        confirmations=[item for item in trace if str(item.get("action") or "").startswith("policy.")],
    )


def _skill_diagnostics(
    state: dict[str, Any],
    runtime: dict[str, Any],
    activation: dict[str, Any],
    rows: _DiagnosticRows,
    previous: dict[str, Any],
) -> dict[str, Any]:
    skill = dict(activation)
    if not skill:
        return skill
    completed = {
        str((item.get("tool_calls") or [{}])[0].get("tool_name") or "")
        for item in rows.tools
        if str(item.get("status") or "") in {"ok", "succeeded", "completed"}
    }
    for source in (
        as_dict(previous.get("skill")).get("completed_tools"),
        state.get("completed_tools"),
        as_dict(state.get("skill_activation")).get("completed_tools"),
    ):
        completed.update(str(name) for name in source or [] if str(name))
    missing = as_dict(state.get("missing_requirements"))
    output_contract = as_dict(skill.get("output_contract"))
    skill.update(
        {
            "completed_tools": sorted(name for name in completed if name),
            "completion_outcome": str(runtime.get("status") or ""),
            "policy_violation_count": len(skill.get("policy_violations") or []),
            "policy_phase": str(state.get("policy_phase") or skill.get("policy_phase") or ""),
            "missing_requirements": {
                "tools": list(missing.get("tools") or []),
                "artifacts": list(missing.get("artifacts") or []),
            },
            "repair_count": int(state.get("repair_count") or skill.get("repair_count") or 0),
            "required_artifact": str(
                output_contract.get("requires_artifact") or skill.get("required_artifact") or ""
            ),
        }
    )
    return skill


def _diagnostic_counts(
    state: dict[str, Any],
    events: list[dict[str, Any]],
    rows: _DiagnosticRows,
    previous: dict[str, Any],
) -> dict[str, int]:
    counts = as_dict(previous.get("counts"))
    return {
        "trace_steps": int(counts.get("trace_steps") or 0) + len(rows.trace),
        "model_calls": int(counts.get("model_calls") or 0) + len(rows.model),
        "tool_calls": int(counts.get("tool_calls") or 0) + len(rows.tools),
        "degraded_tool_results": int(counts.get("degraded_tool_results") or 0)
        + sum(str(item.get("outcome") or "") == "degraded" for item in rows.tools),
        "skipped_tool_results": int(counts.get("skipped_tool_results") or 0)
        + sum(str(item.get("outcome") or "") == "skipped" for item in rows.tools),
        "tool_results": max(int(counts.get("tool_results") or 0), len(state.get("tool_results") or [])),
        "observations": max(int(counts.get("observations") or 0), len(state.get("observations") or [])),
        "events": int(counts.get("events") or 0) + len(events),
    }


def _governance_diagnostics(
    state: dict[str, Any],
    rows: _DiagnosticRows,
    previous: dict[str, Any],
) -> dict[str, Any]:
    governance = as_dict(previous.get("governance"))
    return {
        "model_routes": as_list(governance.get("model_routes"))
        + [_model_route_row(item) for item in rows.routes],
        "budget": as_dict(state.get("budget_state")),
        "tool_policies": as_list(governance.get("tool_policies"))
        + [_tool_policy_row(item) for item in rows.tools if item.get("policy") or item.get("budget")],
        "confirmations": as_list(governance.get("confirmations"))
        + [
            {
                "step": item.get("index"),
                "action": str(item.get("action") or ""),
                "summary": str(item.get("summary") or ""),
            }
            for item in rows.confirmations
        ],
    }


def _model_route_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": item.get("index"),
        "role": str(item.get("model_role") or ""),
        "model_id": str(item.get("model_id") or ""),
        "reason": str(item.get("route_reason") or ""),
        "router_id": str(item.get("router_id") or ""),
        "policy": as_dict(item.get("policy")),
        "budget": as_dict(item.get("budget")),
        "budget_metrics": as_dict(item.get("budget_metrics")),
        "usage": as_dict(item.get("usage")),
        "max_output_tokens": item.get("max_output_tokens"),
    }


def _tool_policy_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": item.get("index"),
        "tool_name": str((item.get("tool_calls") or [{}])[0].get("tool_name") or ""),
        "status": str(item.get("status") or ""),
        "outcome": str(item.get("outcome") or ""),
        "policy": as_dict(item.get("policy")),
        "budget": as_dict(item.get("budget")),
    }


def _timing_diagnostics(rows: _DiagnosticRows, previous: dict[str, Any]) -> dict[str, Any]:
    timings = as_dict(previous.get("timings"))
    return {
        "model_calls": as_list(timings.get("model_calls"))
        + [
            {
                "step": item.get("index"),
                "role": str(item.get("model_role") or ""),
                "model_id": str(item.get("model_id") or ""),
                "status": "failed" if item.get("action") == "model.failed" else "completed",
                "latency_ms": int(item.get("latency_ms") or 0),
                "answer_stream": as_dict(item.get("answer_stream")),
                "error": str(item.get("summary") or "") if item.get("action") == "model.failed" else "",
            }
            for item in rows.model
        ],
        "tool_calls": as_list(timings.get("tool_calls"))
        + [
            {
                "tool_name": str((item.get("tool_calls") or [{}])[0].get("tool_name") or ""),
                "status": str(item.get("status") or ""),
                "latency_ms": int(item.get("latency_ms") or 0),
                "error": str(item.get("summary") or "")
                if item.get("action") == "tool.call.failed"
                else "",
            }
            for item in rows.tools
        ],
    }
