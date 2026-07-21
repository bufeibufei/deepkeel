from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness_core.skills import SkillPolicy
from harness_core.tool_registry import ToolRegistry, ToolSpec
from harness_core.turn_context import ToolViewMode


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    name: str
    description: str
    exposure_mode: str
    tags: tuple[str, ...]

    @classmethod
    def from_spec(cls, spec: ToolSpec) -> "ToolDescriptor":
        exposure_mode = spec.exposure_mode
        if str(spec.runtime_policy.get("model_exposure") or "") == "skill_only":
            exposure_mode = "skill_only"
        return cls(
            name=spec.name,
            description=spec.description,
            exposure_mode=exposure_mode,
            tags=tuple(sorted(str(tag) for tag in spec.discovery_tags if str(tag).strip())),
        )


@dataclass(frozen=True, slots=True)
class ToolView:
    mode: ToolViewMode
    catalog_version: str
    allowed_names: frozenset[str]
    exposed_names: frozenset[str]
    proposed_names: frozenset[str]
    fail_open: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "catalog_version": self.catalog_version,
            "allowed_names": sorted(self.allowed_names),
            "exposed_names": sorted(self.exposed_names),
            "proposed_names": sorted(self.proposed_names),
            "fail_open": self.fail_open,
        }


def resolve_tool_view(
    *,
    registry: ToolRegistry,
    allowed_names: set[str] | None,
    skill: SkillPolicy,
    mode: ToolViewMode,
) -> ToolView:
    installed = {spec.name for spec in registry.list_tools()}
    allowed = installed if allowed_names is None else installed & set(allowed_names)
    proposed: set[str] = set()
    for spec in registry.list_tools():
        if spec.name not in allowed:
            continue
        descriptor = ToolDescriptor.from_spec(spec)
        if descriptor.exposure_mode == "baseline":
            proposed.add(spec.name)
        elif skill.active and descriptor.exposure_mode in {"skill_entry", "skill_only"}:
            proposed.add(spec.name)
    proposed.update(skill.required_tools & allowed)
    for group in skill.required_tool_groups:
        proposed.update(group & allowed)

    fail_open = False
    if mode == "legacy":
        exposed = allowed
    elif mode == "shadow":
        exposed = allowed
    else:
        exposed = proposed
        # Until semantic discovery is configured, never strand a turn that has
        # executable tools but no discoverable entry point.
        if allowed and not exposed:
            exposed = allowed
            fail_open = True
    return ToolView(
        mode=mode,
        catalog_version=registry.catalog_version(),
        allowed_names=frozenset(allowed),
        exposed_names=frozenset(exposed),
        proposed_names=frozenset(proposed),
        fail_open=fail_open,
    )
