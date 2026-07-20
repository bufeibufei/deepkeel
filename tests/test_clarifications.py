from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from harness_core.extension_sdk import (
    ToolSpec,
    clarification_from_validation_error,
    clarification_tool_result,
)
from harness_core.runtime_sdk import ToolCall
from harness_core.clarifications import clarification_for_missing_arguments


def test_missing_argument_clarification_uses_labels_groups_and_formats() -> None:
    spec = ToolSpec(
        name="report.build",
        parameters_schema={"type": "object", "properties": {}},
        required_args=["subject.name"],
        required_arg_groups=[["date.start", "period"]],
        argument_contract={
            "clarification": {
                "prompt": "Complete the report request.",
                "field_labels": {
                    "subject.name": "subject",
                    "date.start": "start date",
                    "period": "date range",
                },
                "accepted_formats": {"date.start": "YYYY-MM-DD"},
            }
        },
    )
    request = clarification_for_missing_arguments(
        ToolCall(id="call-1", name="report.build", arguments={"subject": {"name": " "}}),
        spec,
    )

    assert request is not None
    assert request.issue_type == "missing_input"
    assert request.prompt == "Complete the report request."
    assert request.missing_fields == ["subject.name", "date.start"]
    assert request.accepted_formats == {"date.start": "YYYY-MM-DD"}
    assert request.details["required_any_groups"] == [["date.start", "period"]]

    complete = clarification_for_missing_arguments(
        ToolCall(
            id="call-2",
            name="report.build",
            arguments={"subject": {"name": "Quarterly"}, "period": "Q3"},
        ),
        spec,
    )
    assert complete is None


class InputContract(BaseModel):
    name: str
    count: int = Field(gt=0)


def test_validation_error_clarification_classifies_missing_invalid_and_broad_constraints() -> None:
    try:
        InputContract.model_validate({"count": 0})
    except ValidationError as exc:
        request = clarification_from_validation_error(
            exc,
            fallback_prompt="Correct the input.",
            field_labels={"name": "report name"},
        )
    assert request.issue_type == "missing_input"
    assert request.missing_fields == ["name"]
    assert request.invalid_fields == ["count"]

    broad = clarification_from_validation_error(
        ValueError("range exceeds 90 days"),
        fallback_prompt="Narrow the range.",
        constraint_markers=("exceeds 90 days",),
        constraint_fields=("start_date", "end_date"),
    )
    assert broad.issue_type == "constraint_too_broad"
    assert broad.invalid_fields == ["start_date", "end_date"]


def test_clarification_tool_result_is_a_text_input_handoff() -> None:
    request = clarification_from_validation_error(
        ValueError("invalid date"),
        fallback_prompt="Provide a valid date.",
    )
    result = clarification_tool_result(
        ToolCall(id="call-3", name="calendar.plan"),
        run_id="run-3",
        request=request,
        visible_label="Date required",
    )

    assert result.status == "requires_user_action"
    assert result.pending_action is not None
    assert result.pending_action.action_type == "clarification"
    assert result.pending_action.payload["interaction_mode"] == "text_input"
    assert result.observation is not None
    assert result.observation.metadata["issue_type"] == "invalid_input"
