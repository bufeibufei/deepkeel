from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from deepkeel.contracts import Observation, PendingAction, ToolCall, ToolResult
from deepkeel.policy import PolicyDecision
from deepkeel.tool_execution import ToolExecutionContext
from deepkeel.tool_registry import ToolSpec
from deepkeel.type_narrowing import as_dict


def _bind_authoritative_context_arguments(
    call: ToolCall,
    context: ToolExecutionContext,
    spec: ToolSpec,
) -> ToolCall:
    """Replace model-provided arguments with explicitly bound Host context."""

    bindings = spec.runtime_policy.get("context_argument_bindings")
    if not isinstance(bindings, dict) or not bindings:
        return call
    document = {
        **context.context_bundle,
        "run_id": context.run_id,
        "user_id": context.user_id,
        "thread_id": context.thread_id,
        "turn_id": context.turn_id,
    }
    arguments = dict(call.arguments)
    changed = False
    for argument_name, configured_paths in bindings.items():
        paths = configured_paths if isinstance(configured_paths, list) else [configured_paths]
        value = next(
            (
                resolved
                for path in paths
                if isinstance(path, str)
                and (resolved := _context_path_value(document, path)) not in (None, "")
            ),
            None,
        )
        name = str(argument_name).strip()
        if name and value is not None and arguments.get(name) != value:
            arguments[name] = deepcopy(value)
            changed = True
    return call.model_copy(update={"arguments": arguments}) if changed else call


def _context_path_value(document: Mapping[str, Any], path: str) -> Any:
    current: Any = document
    for segment in (part for part in path.split(".") if part):
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def _confirmation_grant(context: ToolExecutionContext, call: ToolCall) -> dict[str, Any]:
    grants = context.metadata.get("confirmation_grants")
    if not isinstance(grants, dict):
        return {}
    grant = grants.get(call.id)
    return grant if isinstance(grant, dict) else {}


def _confirmation_required_result(
    call: ToolCall,
    context: ToolExecutionContext,
    spec: ToolSpec,
    policy: PolicyDecision,
) -> ToolResult:
    label = str(spec.visible_label or call.name)
    pending = PendingAction(
        id=f"{call.id}:policy-confirmation",
        run_id=context.run_id,
        tool_call_id=call.id,
        action_type="policy_confirmation",
        title=f"Confirm {label}",
        prompt=policy.reason,
        handoff_view="policy_confirmation",
        payload={
            "policy_confirmation": True,
            "deferred_tool_call": call.model_dump(mode="json"),
            "policy_decision": policy.as_dict(),
            "tool_name": call.name,
            "tool_label": label,
            "arguments": dict(call.arguments),
            "confirm_tool_name": call.name,
            "confirm_tool_args": {**call.arguments, "confirmed": True},
        },
    )
    observation = Observation(
        id=f"{call.id}:policy-confirmation",
        run_id=context.run_id,
        tool_call_id=call.id,
        source=call.name,
        status="requires_user_action",
        summary=policy.reason,
        data={
            "policy_confirmation": True,
            "tool_name": call.name,
            "tool_label": label,
            "confirm_tool_name": call.name,
            "confirm_tool_args": {**call.arguments, "confirmed": True},
        },
        metadata={"policy": policy.as_dict()},
    )
    return ToolResult(
        call=call,
        status="requires_user_action",
        summary=policy.reason,
        data={
            "policy_confirmation": True,
            "tool_name": call.name,
            "tool_label": label,
            "confirm_tool_name": call.name,
            "confirm_tool_args": {**call.arguments, "confirmed": True},
        },
        observation=observation,
        pending_action=pending,
        metadata={"governance": {"policy": policy.as_dict(), "budget": {}}},
    )


def _copy_context_mapping(value: dict[str, Any]) -> dict[str, Any]:
    try:
        return deepcopy(value)
    except (TypeError, ValueError):
        return dict(value)


def _normalize_async_result(result: ToolResult, spec: ToolSpec) -> ToolResult:
    if not spec.async_tool or result.status != "succeeded":
        return result
    if bool(result.metadata.get("completed_inline")):
        return result
    observation = result.observation
    if observation is not None:
        observation = observation.model_copy(update={"status": "pending"})
    metadata = dict(result.metadata)
    metadata["typed_async"] = True
    return result.model_copy(
        update={
            "status": "waiting_async",
            "observation": observation,
            "metadata": metadata,
        }
    )


def _keep_single_suspending_result(results: list[ToolResult]) -> list[ToolResult]:
    pending_seen = False
    normalized: list[ToolResult] = []
    for result in results:
        if result.status not in {"requires_user_action", "waiting_async"}:
            normalized.append(result)
            continue
        if not pending_seen:
            pending_seen = True
            normalized.append(result)
            continue
        rejected_call = result.call or ToolCall(
            id=result.tool_call_id,
            name=result.name,
        )
        rejected = _failed_result(
            rejected_call,
            "multiple suspending tools must be executed sequentially",
        )
        rejected.metadata = {
            **result.metadata,
            "suspension_rejected": True,
            "original_status": result.status,
        }
        normalized.append(rejected)
    return normalized


def normalize_arguments(arguments: dict[str, Any], spec: ToolSpec) -> dict[str, Any]:
    normalized = dict(arguments or {})
    contract = spec.argument_contract if isinstance(spec.argument_contract, dict) else {}
    aliases = as_dict(contract.get("aliases"))
    for canonical, candidates in aliases.items():
        if _has_value(normalized, str(canonical)):
            continue
        for candidate in candidates if isinstance(candidates, list) else []:
            value = _nested_value(normalized, str(candidate))
            if _present(value):
                normalized[str(canonical)] = value
                if "." not in str(candidate) and str(candidate) != str(canonical):
                    normalized.pop(str(candidate), None)
                break
    coercions = as_dict(contract.get("coerce"))
    for field_name, target_type in coercions.items():
        if field_name not in normalized:
            continue
        normalized[field_name] = _coerce(normalized[field_name], str(target_type))
    return normalized


def _failed_result(call: ToolCall, error: str, *, retryable: bool = False) -> ToolResult:
    return ToolResult(
        call=call,
        tool_call_id=call.id,
        name=call.name,
        status="failed",
        summary=error,
        error=error,
        retryable=retryable,
    )


def _argument_schema_error(arguments: dict[str, Any], spec: ToolSpec) -> str:
    if spec.runtime_policy.get("argument_validation_authority") == "handler":
        return ""
    schema = spec.formal_parameters_schema()
    if not schema:
        return ""
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
    except SchemaError as exc:
        return f"tool parameters schema is invalid: {exc.message}"
    errors = sorted(
        validator.iter_errors(arguments),
        key=lambda item: ".".join(str(part) for part in item.absolute_path),
    )
    if not errors:
        return ""
    error = errors[0]
    path = ".".join(str(item) for item in error.absolute_path)
    return f"{path}: {error.message}" if path else error.message


def _invalid_arguments_result(
    call: ToolCall,
    context: ToolExecutionContext,
    spec: ToolSpec,
    error: str,
) -> ToolResult:
    if str(spec.runtime_policy.get("invalid_arguments_mode") or "") != "skip":
        result = _failed_result(call, f"tool arguments do not match schema: {error}")
        result.metadata = {
            "error_code": "TOOL_ARGUMENT_SCHEMA_INVALID",
            "schema_validation_error": error,
            "executed": False,
        }
        return result
    summary = "Tool arguments violated the contract; the call was skipped and control returned to the agent."
    data = {
        "status": "skipped",
        "reason_code": "invalid_tool_arguments",
        "fallback": "continue_with_parent_agent",
    }
    observation = Observation(
        id=f"{call.id}:invalid-arguments",
        run_id=context.run_id,
        tool_call_id=call.id,
        source=call.name,
        status="succeeded",
        outcome="skipped",
        summary=summary,
        data=data,
        metadata={
            "error_code": "TOOL_ARGUMENT_SCHEMA_INVALID",
            "schema_validation_error": error,
        },
    )
    return ToolResult(
        call=call,
        status="succeeded",
        outcome="skipped",
        summary=summary,
        data=data,
        observation=observation,
        metadata={
            "visible": False,
            "error_code": "TOOL_ARGUMENT_SCHEMA_INVALID",
            "schema_validation_error": error,
            "executed": False,
        },
    )


def _with_runtime_metrics(
    result: ToolResult,
    started_at: float,
    *,
    phase: str,
    executed: bool = True,
) -> ToolResult:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    metrics = as_dict(metadata.get("runtime_metrics"))
    result.metadata = {
        **metadata,
        "runtime_metrics": {
            **metrics,
            "latency_ms": max(0, int((time.perf_counter() - started_at) * 1000)),
            "phase": phase,
            "executed": executed,
        },
    }
    return result


def _replayed_result(cached: ToolResult, call: ToolCall) -> ToolResult:
    metadata = dict(cached.metadata)
    metadata["idempotent_replay"] = True
    observation = cached.observation
    if observation is not None:
        observation = observation.model_copy(
            update={
                "id": f"{call.id}:observation",
                "tool_call_id": call.id,
            }
        )
    pending_action = cached.pending_action
    if pending_action is not None:
        pending_action = pending_action.model_copy(
            update={
                "id": f"{call.id}:action",
                "tool_call_id": call.id,
            }
        )
    return cached.model_copy(
        deep=True,
        update={
            "call": call,
            "tool_call_id": call.id,
            "observation": observation,
            "pending_action": pending_action,
            "metadata": metadata,
        },
    )


def _nested_value(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _has_value(payload: dict[str, Any], path: str) -> bool:
    return _present(_nested_value(payload, path))


def _present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None and value != [] and value != {}


def _coerce(value: Any, target_type: str) -> Any:
    if target_type in {"string", "str"}:
        return str(value)
    if target_type in {"int", "integer"}:
        return int(value)
    if target_type in {"float", "number"}:
        return float(value)
    if target_type in {"bool", "boolean"}:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return bool(value)
    return value


def _positive_number(value: Any, default: float) -> float:
    try:
        return max(0.001, float(value))
    except (TypeError, ValueError):
        return default


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _tool_reexecution_safe(spec: ToolSpec) -> bool:
    explicit = spec.runtime_policy.get("idempotency_reexecution_safe")
    if isinstance(explicit, bool):
        return explicit
    side_effect = str(spec.runtime_policy.get("side_effect") or "").strip().lower()
    return bool(spec.read_only or side_effect == "none")
