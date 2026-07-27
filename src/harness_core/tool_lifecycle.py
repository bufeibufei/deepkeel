from __future__ import annotations

from harness_core.contracts import ToolResult


COMPLETED_WORKFLOW_TRANSITION_STATUSES = frozenset(
    {"succeeded", "requires_user_action", "waiting_async"}
)


def completes_workflow_transition(result: ToolResult) -> bool:
    """Whether a tool satisfied the current Skill workflow transition."""

    return result.status in COMPLETED_WORKFLOW_TRANSITION_STATUSES
