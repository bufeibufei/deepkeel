from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PENDING_ACTION_SCHEMA_VERSION = "pending-action-v2"


@dataclass(frozen=True, slots=True)
class HandoffSpec:
    action_kind: str
    noun: str
    title: str
    summary: str
    primary_label: str
    cancel_label: str
    handoff_view: str = ""
    secondary_label: str = ""
    aria_label: str = "User action pending"
    completion_artifact_type: str = ""

    def presentation(self) -> dict[str, str]:
        return {
            "kind": self.action_kind,
            "noun": self.noun,
            "title": self.title,
            "summary": self.summary,
            "primary_label": self.primary_label,
            "secondary_label": self.secondary_label,
            "cancel_label": self.cancel_label,
            "aria_label": self.aria_label,
            "completion_artifact_type": self.completion_artifact_type,
        }


class HandoffRegistry:
    """Business-neutral registry populated by capability packs."""

    def __init__(self) -> None:
        self._specs: dict[str, HandoffSpec] = {}

    def register(self, tool_name: str, spec: HandoffSpec) -> None:
        tool = str(tool_name or "").strip()
        if not tool:
            raise ValueError("tool_name is required")
        self._specs[tool] = spec

    def resolve(self, tool_name: str) -> HandoffSpec | None:
        return self._specs.get(str(tool_name or "").strip())

    def items(self) -> tuple[tuple[str, HandoffSpec], ...]:
        return tuple(self._specs.items())


def handoff_spec(
    tool_name: str,
    payload: dict[str, Any] | None = None,
    *,
    registry: HandoffRegistry | None = None,
) -> HandoffSpec:
    tool = str(tool_name or "").strip()
    configured = registry.resolve(tool) if registry is not None else None
    if configured is not None:
        return configured
    source = payload if isinstance(payload, dict) else {}
    title = str(source.get("title") or "Continue current action").strip()
    summary = str(source.get("summary") or "The agent will resume after the action is complete").strip()
    return HandoffSpec(
        action_kind="tool_handoff",
        noun="action",
        title=title,
        summary=summary,
        primary_label="Continue",
        cancel_label="Cancel",
        handoff_view=_handoff_view(source),
    )


def standardize_pending_action_payload(
    *,
    tool_name: str,
    action_type: str,
    payload: dict[str, Any] | None,
    registry: HandoffRegistry | None = None,
) -> tuple[str, dict[str, Any]]:
    source = dict(payload) if isinstance(payload, dict) else {}
    nested = source.get("pending_action") if isinstance(source.get("pending_action"), dict) else {}
    explicit_type = str(nested.get("action_type") or action_type or "").strip().lower()
    if explicit_type == "clarification":
        return "clarification", source

    spec = handoff_spec(tool_name, source, registry=registry)
    handoff_view = spec.handoff_view or _handoff_view(source)
    contract = {
        "schema_version": PENDING_ACTION_SCHEMA_VERSION,
        "action_kind": spec.action_kind,
        "presentation": spec.presentation(),
        "handoff": {
            "view": handoff_view,
            "tool_name": str(tool_name or ""),
        },
        "resume": {"method": "tool_handoff"},
        "cancel": {"method": "run_cancel"},
        "completion": {
            "artifact_type": spec.completion_artifact_type,
        },
    }
    return "tool_handoff", {**source, **contract}


def _handoff_view(payload: dict[str, Any]) -> str:
    pending = payload.get("pending_action") if isinstance(payload.get("pending_action"), dict) else {}
    result = payload.get("tool_result") if isinstance(payload.get("tool_result"), dict) else {}
    return str(
        pending.get("handoff_view")
        or result.get("handoff_view")
        or payload.get("handoff_view")
        or ""
    ).strip()
