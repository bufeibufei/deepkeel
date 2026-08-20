from __future__ import annotations

from typing import Any, Callable

from deepkeel.guardrails import GuardrailAudit
from deepkeel.type_narrowing import as_dict


EventEmitter = Callable[[dict[str, Any]], None]


def emit_input_guardrail_audits(
    emit: EventEmitter,
    audits: tuple[GuardrailAudit, ...],
) -> None:
    for audit in audits:
        emit(
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
                    "visible": False,
                },
            }
        )


def emit_memory_recall_events(
    emit: EventEmitter,
    bundle: dict[str, Any],
    short: dict[str, Any],
) -> None:
    recall = as_dict(bundle.get("memory_recall"))
    if not recall or short.get("resume") or short.get("recover_interrupted"):
        return
    emit(
        {
            "event_type": "memory.recall.decided",
            "title": "Memory recall policy evaluated",
            "summary": str(recall.get("reason") or recall.get("status") or ""),
            "payload": dict(recall),
            "visible": False,
        }
    )
    status = str(recall.get("status") or "")
    if status in {"completed", "failed", "skipped"}:
        emit(
            {
                "event_type": f"memory.recall.{status}",
                "title": f"Memory recall {status}",
                "summary": str(recall.get("reason") or status),
                "payload": dict(recall),
                "visible": False,
            }
        )


__all__ = ["emit_input_guardrail_audits", "emit_memory_recall_events"]
