from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


_WORKFLOW_WAITING_STATUSES = {"waiting_user_input", "waiting_user_action"}
_WORKFLOW_RUNNING_STATUSES = {"reasoning", "executing_tools", "task_running"}
_WORKFLOW_TERMINAL_STATUSES = {"completed", "failed", "canceled", "cancelled"}


class WorkflowCompletionPolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_transition: str = ""
    required_transition_any: list[str] = Field(default_factory=list)
    required_artifact: str = ""
    required_artifacts: list[str] = Field(default_factory=list)
    waiting_statuses: list[str] = Field(default_factory=list)
    running_statuses: list[str] = Field(default_factory=list)
    terminal_statuses: list[str] = Field(default_factory=list)
    allow_model_clarification: bool = False
    clarification_strategy: Literal["model", "tool_contract"] = "model"
    policy_repair_attempts: int = Field(default=1, ge=0, le=1)

    @field_validator(
        "required_transition_any",
        "required_artifacts",
        "waiting_statuses",
        "running_statuses",
        "terminal_statuses",
    )
    @classmethod
    def unique_non_blank_values(cls, values: list[str]) -> list[str]:
        normalized = [str(value or "").strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("workflow policy list values must not be blank")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_status_contract(self) -> "WorkflowCompletionPolicySpec":
        _reject_unknown_statuses("waiting_statuses", self.waiting_statuses, _WORKFLOW_WAITING_STATUSES)
        _reject_unknown_statuses("running_statuses", self.running_statuses, _WORKFLOW_RUNNING_STATUSES)
        _reject_unknown_statuses("terminal_statuses", self.terminal_statuses, _WORKFLOW_TERMINAL_STATUSES)
        if self.allow_model_clarification and "waiting_user_input" not in self.waiting_statuses:
            raise ValueError(
                "allow_model_clarification requires waiting_user_input in waiting_statuses"
            )
        if self.clarification_strategy == "tool_contract" and self.allow_model_clarification:
            raise ValueError(
                "tool_contract clarification cannot allow model-only clarification"
            )
        return self

    def transition_tools(self) -> set[str]:
        return {
            name
            for name in [self.required_transition, *self.required_transition_any]
            if name
        }


class DelegationPolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    allowed_agents: list[str] = Field(default_factory=list)
    max_tasks: int = Field(default=1, ge=1, le=3)
    max_concurrency: int = Field(default=1, ge=1, le=3)
    max_model_calls: int | None = Field(default=None, ge=1, le=24)
    max_tool_calls: int | None = Field(default=None, ge=0, le=24)

    @field_validator("allowed_agents")
    @classmethod
    def unique_agent_ids(cls, values: list[str]) -> list[str]:
        normalized = [str(value or "").strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("delegation allowed_agents must not contain blank ids")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_enabled_policy(self) -> "DelegationPolicySpec":
        if self.enabled and not self.allowed_agents:
            raise ValueError("enabled delegation_policy must declare allowed_agents")
        if self.max_concurrency > self.max_tasks:
            raise ValueError("delegation max_concurrency cannot exceed max_tasks")
        return self


class SkillPackageManifest(BaseModel):
    schema_version: str = "harness-skill-package-v2"
    package_id: str = Field(min_length=1)
    capability_pack: str = Field(min_length=1)
    order: int = Field(default=100, ge=0)
    entry_tool: str = Field(min_length=1)
    entry_tools: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    artifact_types: list[str] = Field(default_factory=list)
    eval_case_ids: list[str] = Field(default_factory=list)
    state_schema_version: str = "1"
    resume_compatible_versions: list[str] = Field(default_factory=list)
    skill_spec: dict[str, Any]

    @field_validator(
        "required_tools",
        "entry_tools",
        "artifact_types",
        "eval_case_ids",
        "resume_compatible_versions",
    )
    @classmethod
    def unique_non_blank_values(cls, values: list[str]) -> list[str]:
        normalized = [str(value or "").strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("package list values must not be blank")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_skill_contract(self) -> "SkillPackageManifest":
        if not self.skill_id or not self.version:
            raise ValueError("skill_spec must include id and version")
        if self.entry_tool not in self.entry_tools:
            self.entry_tools.insert(0, self.entry_tool)
        allowed_tools = set(self.skill_spec.get("allowed_tools") or [])
        declared_required = set(self.skill_spec.get("required_tools") or [])
        disallowed_required = declared_required - allowed_tools
        if disallowed_required:
            raise ValueError("required_tools must also be allowed by skill_spec")
        missing_entries = set(self.entry_tools) - allowed_tools
        if missing_entries:
            raise ValueError("entry_tools must be allowed by skill_spec")
        if not set(self.required_tools).issubset(declared_required):
            raise ValueError("package required_tools must be required by skill_spec")
        if str(self.skill_spec.get("kind") or "prompt") == "workflow":
            completion = WorkflowCompletionPolicySpec.model_validate(
                self.skill_spec.get("completion_policy") or {}
            )
            transition_tools = completion.transition_tools()
            disallowed_transitions = transition_tools - allowed_tools
            if disallowed_transitions:
                raise ValueError(
                    "completion policy transitions must be allowed by skill_spec"
                )
            required_groups = {
                frozenset(str(name).strip() for name in group if str(name or "").strip())
                for group in self.skill_spec.get("required_tool_groups") or []
                if isinstance(group, (list, tuple, set, frozenset))
            }
            if completion.required_transition and completion.required_transition not in declared_required:
                raise ValueError(
                    "required_transition must be declared as a required_tool"
                )
            if completion.required_transition_any and frozenset(
                completion.required_transition_any
            ) not in required_groups:
                raise ValueError(
                    "required_transition_any must match a required_tool_group"
                )
        output = self.skill_spec.get("output_contract") if isinstance(self.skill_spec.get("output_contract"), dict) else {}
        required_artifact = str(output.get("requires_artifact") or "")
        if self.artifact_types and required_artifact not in self.artifact_types:
            raise ValueError("skill_spec artifact contract is not declared by package")
        if self.version not in self.resume_compatible_versions:
            self.resume_compatible_versions.append(self.version)
        delegation = self.skill_spec.get("delegation_policy")
        if delegation is not None:
            DelegationPolicySpec.model_validate(delegation)
        return self

    @computed_field
    @property
    def skill_id(self) -> str:
        return str(self.skill_spec.get("id") or "").strip()

    @computed_field
    @property
    def version(self) -> str:
        return str(self.skill_spec.get("version") or "").strip()

    @computed_field
    @property
    def digest(self) -> str:
        canonical = json.dumps(self.skill_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def runtime_spec(self) -> dict[str, Any]:
        return {
            **self.skill_spec,
            "package": {
                "schema_version": self.schema_version,
                "package_id": self.package_id,
                "capability_pack": self.capability_pack,
                "entry_tools": list(self.entry_tools),
                "state_schema_version": self.state_schema_version,
                "resume_compatible_versions": list(self.resume_compatible_versions),
                "digest": self.digest,
            },
        }


def builtin_skill_package_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "capabilities" / "skill_packages"


def load_skill_packages(directory: str | Path | None = None) -> list[SkillPackageManifest]:
    root = Path(directory) if directory is not None else builtin_skill_package_dir()
    if not root.exists():
        return []
    manifests = [
        SkillPackageManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(root.glob("*.json"))
    ]
    skill_ids = [manifest.skill_id for manifest in manifests]
    package_ids = [manifest.package_id for manifest in manifests]
    if len(skill_ids) != len(set(skill_ids)):
        raise ValueError("duplicate skill package manifest")
    if len(package_ids) != len(set(package_ids)):
        raise ValueError("duplicate skill package id")
    return sorted(manifests, key=lambda item: (item.order, item.skill_id))


def validate_skill_packages(
    manifests: Iterable[SkillPackageManifest],
    *,
    skill_registry: Any,
    tool_registry: Any | None = None,
) -> list[str]:
    issues: list[str] = []
    available_tools = {tool.name for tool in tool_registry.list_tools()} if tool_registry is not None else None
    for manifest in manifests:
        try:
            skill = skill_registry.get(manifest.skill_id)
        except KeyError:
            issues.append(f"{manifest.skill_id}: skill is not registered")
            continue
        if str(skill.version) != manifest.version:
            issues.append(f"{manifest.skill_id}: version mismatch ({skill.version} != {manifest.version})")
        missing_entries = sorted(set(manifest.entry_tools) - set(skill.allowed_tools))
        if missing_entries:
            issues.append(f"{manifest.skill_id}: entry tools are not allowed: {', '.join(missing_entries)}")
        missing_required = sorted(set(manifest.required_tools) - set(skill.required_tools))
        if missing_required:
            issues.append(f"{manifest.skill_id}: required tools drifted: {', '.join(missing_required)}")
        artifact_type = str(skill.output_contract.get("requires_artifact") or "")
        if manifest.artifact_types and artifact_type not in manifest.artifact_types:
            issues.append(f"{manifest.skill_id}: artifact contract drifted ({artifact_type or 'missing'})")
        package = skill.package if isinstance(getattr(skill, "package", None), dict) else {}
        if package.get("digest") != manifest.digest:
            issues.append(f"{manifest.skill_id}: runtime spec is not compiled from the package")
        if available_tools is not None:
            missing_tools = sorted(set(manifest.required_tools + manifest.entry_tools) - available_tools)
            if missing_tools:
                issues.append(f"{manifest.skill_id}: tools are not registered: {', '.join(missing_tools)}")
    return issues


def _reject_unknown_statuses(field_name: str, values: list[str], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unsupported statuses: {', '.join(unknown)}")
