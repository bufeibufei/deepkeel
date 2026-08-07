from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from deepkeel.skills import SkillPolicy
from deepkeel.type_narrowing import as_dict


SKILL_CONTRACT_VIOLATION = "SKILL_CONTRACT_VIOLATION"
_SATISFIED_TOOL_STATUSES = frozenset(
    {
        "succeeded",
        "ok",
        "completed",
    }
)
_INCOMPLETE_ARTIFACT_STATUSES = frozenset(
    {"queued", "pending", "running", "failed", "error", "canceled", "cancelled"}
)


@dataclass(frozen=True, slots=True)
class WorkflowCompletionDecision:
    allowed: bool
    missing_tools: tuple[str, ...] = ()
    missing_tool_groups: tuple[tuple[str, ...], ...] = ()
    missing_artifacts: tuple[str, ...] = ()

    def diagnostics(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "tools": list(self.missing_tools),
            "artifacts": list(self.missing_artifacts),
        }
        if self.missing_tool_groups:
            diagnostics["tool_groups"] = [list(group) for group in self.missing_tool_groups]
        return diagnostics


def evaluate_workflow_completion(
    skill: SkillPolicy,
    state: Mapping[str, Any] | None,
) -> WorkflowCompletionDecision:
    """Evaluate a Workflow Skill completion contract without side effects."""
    if not skill.active or not skill.durable:
        return WorkflowCompletionDecision(allowed=True)

    current = state if isinstance(state, Mapping) else {}
    completed_tools = _completed_tool_names(current)
    artifact_types = {
        str(item.get("artifact_type") or "").strip()
        for item in _mapping_items(current.get("artifacts"))
        if str(item.get("artifact_type") or "").strip() and _artifact_is_complete(item)
    }
    missing_tools = tuple(sorted(skill.required_tools - completed_tools))
    missing_tool_groups = tuple(
        tuple(sorted(group))
        for group in skill.required_tool_groups
        if not group.intersection(completed_tools)
    )
    missing_artifacts = tuple(sorted(skill.required_artifacts - artifact_types))
    return WorkflowCompletionDecision(
        allowed=not missing_tools and not missing_tool_groups and not missing_artifacts,
        missing_tools=missing_tools,
        missing_tool_groups=missing_tool_groups,
        missing_artifacts=missing_artifacts,
    )


def workflow_repair_prompt(decision: WorkflowCompletionDecision) -> str:
    missing_sections = []
    if decision.missing_tools:
        missing_sections.append(f"required tools: {', '.join(decision.missing_tools)}")
    if decision.missing_tool_groups:
        missing_sections.extend(
            f"one of required tools: {', '.join(group)}"
            for group in decision.missing_tool_groups
        )
    if decision.missing_artifacts:
        missing_sections.append(f"required artifacts: {', '.join(decision.missing_artifacts)}")
    missing = "; ".join(missing_sections) or "unknown workflow requirements"
    return (
        "Workflow Skill completion policy repair. The previous final answer cannot be accepted because "
        f"the following requirements are missing: {missing}. Continue the workflow and satisfy every "
        "missing requirement before giving another final answer. This is the only policy repair attempt."
    )


def workflow_violation_message(decision: WorkflowCompletionDecision) -> str:
    missing_sections = []
    if decision.missing_tools:
        missing_sections.append(f"tools={','.join(decision.missing_tools)}")
    if decision.missing_tool_groups:
        missing_sections.extend(
            f"one_of={','.join(group)}"
            for group in decision.missing_tool_groups
        )
    if decision.missing_artifacts:
        missing_sections.append(f"artifacts={','.join(decision.missing_artifacts)}")
    details = "; ".join(missing_sections) or "unspecified requirements"
    return f"Workflow Skill completion contract was not satisfied after policy repair: {details}."


def _completed_tool_names(state: Mapping[str, Any]) -> set[str]:
    completed = {
        str(name).strip()
        for name in state.get("completed_tools", [])
        if str(name).strip()
    }
    skill = state.get("skill_activation")
    if isinstance(skill, Mapping):
        completed.update(
            str(name).strip()
            for name in skill.get("completed_tools", [])
            if str(name).strip()
        )
    for result in _mapping_items(state.get("tool_results")):
        if str(result.get("status") or "").strip() not in _SATISFIED_TOOL_STATUSES:
            continue
        name = str(result.get("name") or result.get("tool_name") or "").strip()
        if name:
            completed.add(name)
    failed_resume_sources = {
        str(item.get("source") or "").strip()
        for item in _mapping_items(state.get("observations"))
        if str(item.get("status") or "").strip() == "failed"
        and str(as_dict(item.get("metadata")).get("resume_source") or "").strip()
    }
    completed.difference_update(failed_resume_sources)
    return completed


def _mapping_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _artifact_is_complete(artifact: Mapping[str, Any]) -> bool:
    data = as_dict(artifact.get("data"))
    status = str(
        artifact.get("status") or data.get("status") or data.get("report_status") or ""
    ).strip().lower()
    return status not in _INCOMPLETE_ARTIFACT_STATUSES
