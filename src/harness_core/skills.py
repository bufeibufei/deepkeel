from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from harness_core.type_narrowing import as_dict


@dataclass(frozen=True, slots=True)
class DelegationPolicy:
    enabled: bool = False
    allowed_agents: frozenset[str] = field(default_factory=frozenset)
    max_tasks: int = 1
    max_concurrency: int = 1
    max_model_calls: int | None = None
    max_tool_calls: int | None = None

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any] | None) -> "DelegationPolicy":
        raw = snapshot if isinstance(snapshot, dict) else {}
        configured = raw.get("delegation_policy")
        policy = configured if isinstance(configured, dict) else {}
        enabled = bool(policy.get("enabled"))
        return cls(
            enabled=enabled,
            allowed_agents=frozenset(
                str(agent_id).strip()
                for agent_id in policy.get("allowed_agents", [])
                if str(agent_id or "").strip()
            ),
            max_tasks=_bounded_int(policy.get("max_tasks"), default=1, minimum=1, maximum=3),
            max_concurrency=_bounded_int(
                policy.get("max_concurrency"), default=1, minimum=1, maximum=3
            ),
            max_model_calls=_optional_bounded_int(
                policy.get("max_model_calls"), minimum=1, maximum=24
            ),
            max_tool_calls=_optional_bounded_int(
                policy.get("max_tool_calls"), minimum=0, maximum=24
            ),
        )

    def allows_agent(self, agent_id: str) -> bool:
        return self.enabled and str(agent_id or "") in self.allowed_agents

    def runtime_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "allowed_agents": sorted(self.allowed_agents),
            "max_tasks": self.max_tasks,
            "max_concurrency": self.max_concurrency,
            "max_model_calls": self.max_model_calls,
            "max_tool_calls": self.max_tool_calls,
        }


@dataclass(frozen=True, slots=True)
class SkillPolicy:
    skill_id: str = ""
    version: str = ""
    kind: Literal["prompt", "workflow"] = "prompt"
    label: str = ""
    invocation_id: str = ""
    source: str = ""
    explicit: bool = False
    phase: str = ""
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    required_tools: frozenset[str] = field(default_factory=frozenset)
    required_tool_groups: tuple[frozenset[str], ...] = ()
    required_artifacts: frozenset[str] = field(default_factory=frozenset)
    prompt_instructions: str = ""
    completion_policy: dict[str, Any] = field(default_factory=dict)
    retry_policy: dict[str, Any] = field(default_factory=dict)
    ui_handoff: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any] | None) -> "SkillPolicy":
        raw = snapshot if isinstance(snapshot, dict) else {}
        kind = str(raw.get("kind") or "prompt")
        return cls(
            skill_id=str(raw.get("skill_id") or ""),
            version=str(raw.get("version") or ""),
            kind="workflow" if kind == "workflow" else "prompt",
            label=str(raw.get("label") or ""),
            invocation_id=str(raw.get("invocation_id") or ""),
            source=str(raw.get("source") or ""),
            explicit=bool(raw.get("explicit")),
            phase=str(raw.get("phase") or ""),
            allowed_tools=frozenset(str(name) for name in raw.get("allowed_tools", []) if name),
            required_tools=frozenset(str(name) for name in raw.get("required_tools", []) if name),
            required_tool_groups=_required_tool_groups(raw),
            required_artifacts=_required_artifacts(raw),
            prompt_instructions=str(raw.get("prompt_instructions") or ""),
            completion_policy=dict(raw.get("completion_policy") or {}),
            retry_policy=dict(raw.get("retry_policy") or {}),
            ui_handoff=dict(raw.get("ui_handoff") or {}),
            extensions=dict(raw),
        )

    @property
    def active(self) -> bool:
        return bool(self.skill_id)

    @property
    def durable(self) -> bool:
        return self.kind == "workflow"

    def allows_tool(self, tool_name: str) -> bool:
        return not self.allowed_tools or tool_name in self.allowed_tools

    @property
    def policy_repair_limit(self) -> int:
        raw_limit = self.retry_policy.get(
            "policy_repair_attempts",
            self.completion_policy.get("policy_repair_attempts", 1),
        )
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            limit = 1
        return min(1, max(0, limit))

    def runtime_snapshot(self) -> dict[str, Any]:
        if not self.active:
            return {}
        return {
            **self.extensions,
            "skill_id": self.skill_id,
            "version": self.version,
            "kind": self.kind,
            "label": self.label,
            "invocation_id": self.invocation_id,
            "source": self.source,
            "explicit": self.explicit,
            "phase": self.phase,
            "allowed_tools": sorted(self.allowed_tools),
            "required_tools": sorted(self.required_tools),
            "required_tool_groups": [sorted(group) for group in self.required_tool_groups],
            "required_artifacts": sorted(self.required_artifacts),
            "prompt_instructions": self.prompt_instructions,
            "completion_policy": dict(self.completion_policy),
            "retry_policy": dict(self.retry_policy),
            "ui_handoff": dict(self.ui_handoff),
        }

    @property
    def delegation(self) -> DelegationPolicy:
        return DelegationPolicy.from_snapshot(self.extensions)


def _required_artifacts(raw: dict[str, Any]) -> frozenset[str]:
    completion = as_dict(raw.get("completion_policy"))
    output = as_dict(raw.get("output_contract"))
    values = [
        raw.get("required_artifacts"),
        completion.get("required_artifact"),
        completion.get("required_artifacts"),
        output.get("requires_artifact"),
        output.get("required_artifacts"),
    ]
    names: set[str] = set()
    for value in values:
        candidates = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
        names.update(str(item).strip() for item in candidates if str(item or "").strip())
    return frozenset(names)


def _required_tool_groups(raw: dict[str, Any]) -> tuple[frozenset[str], ...]:
    groups = raw.get("required_tool_groups")
    if not isinstance(groups, (list, tuple)):
        completion = as_dict(raw.get("completion_policy"))
        required_any = completion.get("required_transition_any")
        groups = [required_any] if isinstance(required_any, (list, tuple, set, frozenset)) else []
    normalized: list[frozenset[str]] = []
    for group in groups:
        if not isinstance(group, (list, tuple, set, frozenset)):
            continue
        names = frozenset(str(name).strip() for name in group if str(name or "").strip())
        if names:
            normalized.append(names)
    return tuple(normalized)


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


def _optional_bounded_int(value: Any, *, minimum: int, maximum: int) -> int | None:
    if value in (None, ""):
        return None
    return _bounded_int(value, default=minimum, minimum=minimum, maximum=maximum)
