from __future__ import annotations

from typing import Any, Callable

from deepkeel.capability_manifest import RuntimeGeneration
from deepkeel.hooks import HookAudit
from deepkeel.persistence import CheckpointCompatibilityError
from deepkeel.type_narrowing import as_dict


EventSink = Callable[[dict[str, Any]], None]


def optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def ensure_resume_generation_compatible(
    current: RuntimeGeneration | None,
    durable_state: dict[str, Any],
) -> None:
    runtime = as_dict(durable_state.get("runtime"))
    diagnostics = as_dict(runtime.get("diagnostics"))
    capabilities = as_dict(diagnostics.get("capabilities"))
    generation_payload = capabilities.get("generation")
    if not isinstance(generation_payload, dict) or not generation_payload:
        return
    try:
        previous = RuntimeGeneration.model_validate(generation_payload)
    except Exception as exc:
        raise CheckpointCompatibilityError("persisted runtime generation is invalid") from exc
    if current is None:
        raise CheckpointCompatibilityError(
            f"runtime generation {previous.generation_id} is not installed"
        )
    if current.generation_id == previous.generation_id:
        return
    issues = current.resume_compatibility_issues(previous)
    if issues:
        raise CheckpointCompatibilityError(
            "runtime generation is not resume compatible: " + "; ".join(issues)
        )


def hook_audit_dict(audit: HookAudit) -> dict[str, Any]:
    return {
        "hook_id": audit.hook_id,
        "hook_point": audit.point.value,
        "operation_id": audit.operation_id,
        "status": audit.status,
        "duration_ms": audit.duration_ms,
        "replayed": audit.replayed,
        "required": audit.required,
        "error": audit.error,
        "diagnostics": dict(audit.diagnostics),
    }


def emit_runtime_hook_audits(
    emit: EventSink,
    audits: tuple[HookAudit, ...],
) -> None:
    for audit in audits:
        emit(
            {
                "event_type": "hook.executed",
                "title": "Lifecycle hook",
                "summary": f"{audit.point.value}: {audit.status}",
                "payload": {**hook_audit_dict(audit), "visible": False},
                "visibility": "debug",
            }
        )
