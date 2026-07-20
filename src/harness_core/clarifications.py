from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from harness_core.contracts import Observation, PendingAction, ToolCall, ToolResult
from harness_core.type_narrowing import as_dict


class ClarificationRequest(BaseModel):
    """A recoverable request for text input, not a tool execution failure."""

    issue_type: Literal["missing_input", "invalid_input", "constraint_too_broad"]
    prompt: str
    missing_fields: list[str] = Field(default_factory=list)
    invalid_fields: list[str] = Field(default_factory=list)
    accepted_formats: dict[str, str] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "state": "waiting_user_input",
            "interaction_mode": "text_input",
            "clarification": self.model_dump(mode="json"),
        }


def clarification_for_missing_arguments(call: ToolCall, spec: Any) -> ClarificationRequest | None:
    arguments = call.arguments if isinstance(call.arguments, dict) else {}
    missing = [name for name in spec.required_args if not _has_value(arguments, name)]
    missing_groups = [
        [name for name in group if name]
        for group in spec.required_arg_groups
        if group and not any(_has_value(arguments, name) for name in group)
    ]
    if not missing and not missing_groups:
        return None

    contract = spec.argument_contract if isinstance(spec.argument_contract, dict) else {}
    clarification = as_dict(contract.get("clarification"))
    labels = as_dict(clarification.get("field_labels"))
    formats = as_dict(clarification.get("accepted_formats"))
    display_fields = [str(labels.get(name) or name) for name in missing]
    display_fields.extend(" / ".join(str(labels.get(name) or name) for name in group) for group in missing_groups)
    prompt = str(clarification.get("prompt") or "").strip()
    if not prompt:
        prompt = f"Please provide: {'; '.join(display_fields)}."
    return ClarificationRequest(
        issue_type="missing_input",
        prompt=prompt,
        missing_fields=[*missing, *(group[0] for group in missing_groups if group)],
        accepted_formats={str(key): str(value) for key, value in formats.items()},
        details={"required_any_groups": missing_groups},
    )


def clarification_from_validation_error(
    error: Exception,
    *,
    fallback_prompt: str,
    field_labels: dict[str, str] | None = None,
    constraint_markers: tuple[str, ...] = (),
    constraint_fields: tuple[str, ...] = (),
) -> ClarificationRequest:
    labels = field_labels or {}
    missing_fields: list[str] = []
    invalid_fields: list[str] = []
    issue_type: Literal["missing_input", "invalid_input", "constraint_too_broad"] = "invalid_input"
    detail = str(error)

    if isinstance(error, ValidationError):
        for item in error.errors():
            field = ".".join(str(part) for part in item.get("loc") or [])
            if not field:
                continue
            if str(item.get("type") or "").lower() == "missing":
                missing_fields.append(field)
            else:
                invalid_fields.append(field)
    lowered = detail.lower()
    normalized_markers = tuple(
        marker.lower().strip() for marker in constraint_markers if marker.strip()
    )
    if normalized_markers and any(marker in lowered for marker in normalized_markers):
        issue_type = "constraint_too_broad"
        invalid_fields = list(constraint_fields)
    elif missing_fields:
        issue_type = "missing_input"

    named_fields = missing_fields or invalid_fields
    if named_fields and fallback_prompt.endswith("："):
        fallback_prompt += "; ".join(str(labels.get(name) or name) for name in named_fields)
    return ClarificationRequest(
        issue_type=issue_type,
        prompt=fallback_prompt,
        missing_fields=missing_fields,
        invalid_fields=invalid_fields,
        details={"validation_error": detail},
    )


def clarification_tool_result(
    call: ToolCall,
    *,
    run_id: str,
    request: ClarificationRequest,
    visible_label: str = "Additional information required",
) -> ToolResult:
    payload = request.payload()
    pending_action = PendingAction(
        id=f"clarification-{uuid4()}",
        run_id=run_id,
        tool_call_id=call.id,
        action_type="clarification",
        title=visible_label,
        prompt=request.prompt,
        payload=payload,
    )
    return ToolResult(
        call=call,
        status="requires_user_action",
        summary=request.prompt,
        data=payload,
        observation=Observation(
            id=f"{call.id}:clarification",
            run_id=run_id,
            tool_call_id=call.id,
            source=call.name,
            status="requires_user_action",
            summary=request.prompt,
            data=payload,
            metadata={"interaction_mode": "text_input", "issue_type": request.issue_type},
        ),
        pending_action=pending_action,
        metadata={
            "interaction_mode": "text_input",
            "issue_type": request.issue_type,
            "visible_label": visible_label,
        },
    )


def _has_value(arguments: dict[str, Any], path: str) -> bool:
    value: Any = arguments
    for part in str(path).split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True
