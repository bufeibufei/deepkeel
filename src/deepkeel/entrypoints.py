from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentEntrypointSpec(BaseModel):
    """A user-facing root Agent assembled from an installed runtime generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    version: str = Field(default="1.0.0", min_length=1)
    label: str = Field(min_length=1)
    description: str = ""
    package_allowlist: tuple[str, ...] = ()
    skill_allowlist: tuple[str, ...] = ()
    tool_allowlist: tuple[str, ...] = ()
    subagent_allowlist: tuple[str, ...] = ()
    include_dependencies: bool = True
    system_prompt: str = ""
    model_policy: dict[str, Any] = Field(default_factory=dict)
    context_policy: dict[str, Any] = Field(default_factory=dict)
    memory_policy: dict[str, Any] = Field(default_factory=dict)
    handoff_policy: dict[str, Any] = Field(default_factory=dict)
    entry_modes: tuple[str, ...] = ("direct",)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize(self) -> "AgentEntrypointSpec":
        for field_name in (
            "package_allowlist",
            "skill_allowlist",
            "tool_allowlist",
            "subagent_allowlist",
            "entry_modes",
        ):
            normalized = tuple(
                dict.fromkeys(
                    str(value).strip() for value in getattr(self, field_name) if str(value).strip()
                )
            )
            object.__setattr__(self, field_name, normalized)
        object.__setattr__(self, "id", self.id.strip())
        object.__setattr__(self, "version", self.version.strip())
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "system_prompt", self.system_prompt.strip())
        return self

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "label": self.label,
            "description": self.description,
            "entry_modes": list(self.entry_modes),
            "metadata": dict(self.metadata),
        }


class CapabilityView(BaseModel):
    """Immutable effective capability scope for one root Agent conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entrypoint_id: str = "default"
    restricted: bool = False
    entrypoint_version: str = ""
    label: str = ""
    generation_id: str = ""
    package_ids: tuple[str, ...] = ()
    skill_ids: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    subagent_ids: tuple[str, ...] = ()
    context_contributor_ids: tuple[str, ...] = ()
    hook_ids: tuple[str, ...] = ()
    artifact_types: tuple[str, ...] = ()
    memory_namespaces: tuple[str, ...] = ()
    permission_scopes: tuple[str, ...] = ()
    system_prompt: str = ""
    model_policy: dict[str, Any] = Field(default_factory=dict)
    context_policy: dict[str, Any] = Field(default_factory=dict)
    memory_policy: dict[str, Any] = Field(default_factory=dict)
    handoff_policy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    scope_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def allows_skill(self, skill_id: str) -> bool:
        normalized = str(skill_id or "").strip()
        return not self.restricted or not normalized or normalized in self.skill_ids

    def allows_tool(self, tool_name: str) -> bool:
        return not self.restricted or str(tool_name or "").strip() in self.tool_names

    def allows_subagent(self, subagent_id: str) -> bool:
        return not self.restricted or str(subagent_id or "").strip() in self.subagent_ids


def resolve_capability_view(
    *,
    entrypoint_id: str,
    entrypoint_version: str = "",
    runtime_generation: Any,
    contributions: Sequence[Any],
    catalog: Any,
    installed_tool_names: Iterable[str] = (),
) -> CapabilityView:
    """Resolve an entrypoint against one immutable installed runtime generation."""

    normalized_id = str(entrypoint_id or "").strip()
    entrypoints = getattr(catalog, "agent_entrypoints", {})
    manifests = {
        str(getattr(manifest, "id", "")): manifest
        for manifest in tuple(getattr(runtime_generation, "packages", ()) or ())
        if str(getattr(manifest, "id", ""))
    }
    contributions_by_package = {
        str(getattr(item, "package_id", "")): item
        for item in contributions
        if str(getattr(item, "package_id", ""))
    }
    generation_id = str(getattr(runtime_generation, "generation_id", "") or "")

    if not normalized_id or normalized_id == "default":
        spec = None
        package_ids = set(contributions_by_package)
        normalized_id = "default"
    else:
        try:
            candidate = entrypoints[normalized_id]
        except KeyError as exc:
            raise ValueError(f"unknown Agent entrypoint: {normalized_id}") from exc
        if not isinstance(candidate, AgentEntrypointSpec):
            raise TypeError(f"Agent entrypoint {normalized_id} has an invalid specification")
        spec = candidate
        expected_version = str(entrypoint_version or "").strip()
        if expected_version and expected_version != spec.version:
            raise ValueError(
                f"Agent entrypoint {normalized_id} version mismatch: "
                f"expected {expected_version}, installed {spec.version}"
            )
        owners = {
            package_id
            for package_id, contribution in contributions_by_package.items()
            if normalized_id in tuple(getattr(contribution, "agent_entrypoints", ()) or ())
        }
        if len(owners) != 1:
            raise ValueError(
                f"Agent entrypoint {normalized_id} must have exactly one owning package"
            )
        owner = next(iter(owners))
        package_ids = set(spec.package_allowlist or (owner,))
        package_ids.add(owner)
        if spec.include_dependencies:
            package_ids = _include_transitive_dependencies(package_ids, manifests)

    missing_packages = sorted(package_ids - set(contributions_by_package))
    if missing_packages:
        raise ValueError(
            f"Agent entrypoint {normalized_id} references unavailable packages: "
            + ", ".join(missing_packages)
        )

    selected = tuple(contributions_by_package[item] for item in sorted(package_ids))
    skill_ids = _contribution_union(selected, "skills")
    tool_names = _contribution_union(selected, "tools")
    subagent_ids = _contribution_union(selected, "subagents")
    context_contributor_ids = _contribution_union(selected, "context_contributors")
    hook_ids = _contribution_union(selected, "hooks")
    artifact_types = _contribution_union(selected, "artifact_types")

    installed_tools = {str(name).strip() for name in installed_tool_names if str(name).strip()}
    if spec is None:
        tool_names.update(installed_tools)
    tool_names.update(
        name
        for name in installed_tools
        if name in {"runtime.discover_tools", "runtime.create_plan"}
    )
    if spec is not None:
        skill_ids = _apply_allowlist("skills", spec.skill_allowlist, skill_ids, normalized_id)
        tool_names = _apply_allowlist("tools", spec.tool_allowlist, tool_names, normalized_id)
        subagent_ids = _apply_allowlist(
            "subagents", spec.subagent_allowlist, subagent_ids, normalized_id
        )

    memory_namespaces = {
        str(namespace)
        for package_id in package_ids
        for namespace in tuple(getattr(manifests.get(package_id), "memory_namespaces", ()) or ())
        if str(namespace).strip()
    }
    permission_scopes = {
        str(scope)
        for package_id in package_ids
        for scope in tuple(getattr(manifests.get(package_id), "permissions", ()) or ())
        if str(scope).strip()
    }
    payload: dict[str, Any] = {
        "entrypoint_id": normalized_id,
        "restricted": spec is not None,
        "entrypoint_version": spec.version if spec is not None else "",
        "label": spec.label if spec is not None else "",
        "generation_id": generation_id,
        "package_ids": tuple(sorted(package_ids)),
        "skill_ids": tuple(sorted(skill_ids)),
        "tool_names": tuple(sorted(tool_names)),
        "subagent_ids": tuple(sorted(subagent_ids)),
        "context_contributor_ids": tuple(sorted(context_contributor_ids)),
        "hook_ids": tuple(sorted(hook_ids)),
        "artifact_types": tuple(sorted(artifact_types)),
        "memory_namespaces": tuple(sorted(memory_namespaces)),
        "permission_scopes": tuple(sorted(permission_scopes)),
        "system_prompt": spec.system_prompt if spec is not None else "",
        "model_policy": dict(spec.model_policy) if spec is not None else {},
        "context_policy": dict(spec.context_policy) if spec is not None else {},
        "memory_policy": dict(spec.memory_policy) if spec is not None else {},
        "handoff_policy": dict(spec.handoff_policy) if spec is not None else {},
        "metadata": dict(spec.metadata) if spec is not None else {},
    }
    payload["scope_hash"] = _scope_hash(payload)
    return CapabilityView.model_validate(payload)


def narrow_capability_view(
    parent: CapabilityView,
    *,
    skill_ids: Iterable[str] | None = None,
    tool_names: Iterable[str] | None = None,
    subagent_ids: Iterable[str] | None = None,
) -> CapabilityView:
    """Create a child scope while proving it cannot exceed its parent scope."""

    updates: dict[str, Any] = {}
    for field_name, requested in (
        ("skill_ids", skill_ids),
        ("tool_names", tool_names),
        ("subagent_ids", subagent_ids),
    ):
        if requested is None:
            continue
        parent_values = set(getattr(parent, field_name))
        normalized = {str(item).strip() for item in requested if str(item).strip()}
        unexpected = sorted(normalized - parent_values)
        if unexpected:
            raise ValueError(
                f"child CapabilityView cannot add {field_name}: " + ", ".join(unexpected)
            )
        updates[field_name] = tuple(sorted(normalized))
    payload = parent.model_dump(mode="python")
    payload.update(updates)
    payload["scope_hash"] = ""
    payload["scope_hash"] = _scope_hash(payload)
    return CapabilityView.model_validate(payload)


def merge_model_policy(
    entrypoint_policy: Mapping[str, Any],
    request_policy: Mapping[str, Any],
) -> dict[str, Any]:
    merged = _deep_merge(dict(entrypoint_policy), dict(request_policy))
    return merged


def _restrict_tool_names(
    capability_view: Mapping[str, Any],
    names: Iterable[str],
) -> set[str]:
    scoped = {str(name).strip() for name in names if str(name).strip()}
    if not bool(capability_view.get("restricted")):
        return scoped
    allowed = {
        str(name).strip()
        for name in capability_view.get("tool_names", ())
        if str(name).strip()
    }
    scoped.intersection_update(allowed)
    if not _restrict_subagent_ids(capability_view, ()):
        scoped.discard("agent.delegate")
    return scoped


def _restrict_subagent_ids(
    capability_view: Mapping[str, Any],
    subagent_ids: Iterable[str],
) -> set[str]:
    requested = {
        str(agent_id).strip() for agent_id in subagent_ids if str(agent_id).strip()
    }
    if not bool(capability_view.get("restricted")):
        return requested
    allowed = {
        str(agent_id).strip()
        for agent_id in capability_view.get("subagent_ids", ())
        if str(agent_id).strip()
    }
    return requested & allowed if requested else allowed


def _entrypoint_identity(view: CapabilityView) -> dict[str, str]:
    return {
        "id": view.entrypoint_id,
        "version": view.entrypoint_version,
        "label": view.label,
        "scope_hash": view.scope_hash,
    }


def _entrypoint_resolved_event(view: CapabilityView) -> dict[str, Any]:
    return {
        "event_type": "agent.entrypoint.resolved",
        "title": "Agent entrypoint resolved",
        "summary": view.label or "default",
        "payload": {
            **_entrypoint_identity(view),
            "package_ids": list(view.package_ids),
            "visible": False,
        },
        "visible": False,
    }


def _compose_entrypoint_prompt(base_prompt: str, view: CapabilityView) -> str:
    return "\n\n".join(
        part for part in (base_prompt, view.system_prompt) if part.strip()
    )


def _validate_entrypoint_skill(view: CapabilityView, skill_id: str) -> None:
    if view.allows_skill(skill_id):
        return
    raise ValueError(
        f"Skill {skill_id!r} is outside Agent entrypoint {view.entrypoint_id!r}"
    )


def _include_transitive_dependencies(
    package_ids: set[str],
    manifests: Mapping[str, Any],
) -> set[str]:
    resolved = set(package_ids)
    pending = list(package_ids)
    while pending:
        package_id = pending.pop()
        manifest = manifests.get(package_id)
        dependencies = dict(getattr(manifest, "dependencies", {}) or {})
        for dependency_id in dependencies:
            normalized = str(dependency_id).strip()
            if normalized and normalized not in resolved:
                resolved.add(normalized)
                pending.append(normalized)
    return resolved


def _contribution_union(contributions: Sequence[Any], field_name: str) -> set[str]:
    return {
        str(value).strip()
        for contribution in contributions
        for value in tuple(getattr(contribution, field_name, ()) or ())
        if str(value).strip()
    }


def _apply_allowlist(
    kind: str,
    allowlist: tuple[str, ...],
    available: set[str],
    entrypoint_id: str,
) -> set[str]:
    if not allowlist:
        return available
    requested = set(allowlist)
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(
            f"Agent entrypoint {entrypoint_id} references out-of-scope {kind}: "
            + ", ".join(unknown)
        )
    return requested


def _scope_hash(payload: Mapping[str, Any]) -> str:
    canonical = {
        key: value for key, value in payload.items() if key not in {"scope_hash", "system_prompt"}
    }
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


__all__ = [
    "AgentEntrypointSpec",
    "CapabilityView",
    "merge_model_policy",
    "narrow_capability_view",
    "resolve_capability_view",
]
