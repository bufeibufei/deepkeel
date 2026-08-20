from __future__ import annotations

import hashlib
import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepkeel.version import (
    DEEPKEEL_CONTRACT_VERSION,
    DEEPKEEL_VERSION,
)


_LEGACY_PACKAGE_VERSION_BRIDGES = {
    # The distribution/import rename is a v4 SDK break, while durable capability
    # generations continue to use the unchanged harness-core-v3 contract.
    DEEPKEEL_CONTRACT_VERSION: ("3.35.1",),
}


class CapabilityBudgetSpec(BaseModel):
    """Portable upper bounds contributed by one Capability Package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_model_calls: int = Field(default=0, ge=0)
    max_tool_calls: int = Field(default=0, ge=0)
    max_input_tokens_total: int = Field(default=0, ge=0)
    max_output_tokens_total: int = Field(default=0, ge=0)
    max_model_retries: int = Field(default=0, ge=0)
    max_parallel_tools: int = Field(default=0, ge=0)
    max_elapsed_seconds: float = Field(default=0.0, ge=0)
    roles: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_roles(self) -> "CapabilityBudgetSpec":
        normalized: dict[str, dict[str, float]] = {}
        for role, limits in self.roles.items():
            role_name = str(role or "").strip()
            if not role_name:
                raise ValueError("budget role names must not be blank")
            normalized_limits: dict[str, float] = {}
            for name, value in limits.items():
                metric = str(name or "").strip()
                numeric = float(value)
                if not metric or numeric < 0:
                    raise ValueError("budget role limits must be named and non-negative")
                normalized_limits[metric] = numeric
            normalized[role_name] = normalized_limits
        object.__setattr__(self, "roles", normalized)
        return self

    def limits(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.model_dump(mode="json").items()
            if value not in (0, 0.0, {}, None)
        }


class CapabilityManifest(BaseModel):
    """Declarative, versioned package boundary consumed by the runtime control plane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "harness-capability-manifest-v1"
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    core_contract: str = DEEPKEEL_CONTRACT_VERSION
    core_version: str = f">={DEEPKEEL_VERSION}"
    entrypoint: str = Field(min_length=1)
    dependencies: dict[str, str] = Field(default_factory=dict)
    skills: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    artifact_types: tuple[str, ...] = ()
    subagents: tuple[str, ...] = ()
    handoffs: tuple[str, ...] = ()
    hooks: tuple[str, ...] = ()
    context_contributors: tuple[str, ...] = ()
    agent_entrypoints: tuple[str, ...] = ()
    mcp_servers: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    tool_permissions: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    memory_namespaces: tuple[str, ...] = ()
    ui_surfaces: tuple[str, ...] = ()
    budget: CapabilityBudgetSpec = Field(default_factory=CapabilityBudgetSpec)
    state_schema_version: str = Field(default="1", min_length=1)
    resume_compatible_versions: tuple[str, ...] = ()
    state_migrations: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_and_validate(self) -> "CapabilityManifest":
        if self.core_contract != DEEPKEEL_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported core contract {self.core_contract!r}; "
                f"expected {DEEPKEEL_CONTRACT_VERSION!r}"
            )
        if not core_version_satisfies(self.core_contract, self.core_version):
            raise ValueError(
                f"core {DEEPKEEL_VERSION} does not satisfy {self.core_version!r}"
            )
        for field_name in (
            "skills",
            "tools",
            "artifact_types",
            "subagents",
            "handoffs",
            "hooks",
            "context_contributors",
            "agent_entrypoints",
            "mcp_servers",
            "resources",
            "permissions",
            "memory_namespaces",
            "ui_surfaces",
            "resume_compatible_versions",
        ):
            values = tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value in getattr(self, field_name)
                    if str(value).strip()
                )
            )
            object.__setattr__(self, field_name, values)
        normalized_dependencies = {
            str(package_id).strip(): str(version_range).strip() or "*"
            for package_id, version_range in self.dependencies.items()
            if str(package_id).strip()
        }
        object.__setattr__(self, "dependencies", normalized_dependencies)
        normalized_tool_permissions = {
            str(tool_name).strip(): tuple(
                dict.fromkeys(
                    str(scope).strip()
                    for scope in scopes
                    if str(scope).strip()
                )
            )
            for tool_name, scopes in self.tool_permissions.items()
            if str(tool_name).strip()
        }
        unknown_tools = sorted(set(normalized_tool_permissions) - set(self.tools))
        if unknown_tools:
            raise ValueError(
                "tool permission mappings reference undeclared tools: "
                + ", ".join(unknown_tools)
            )
        undeclared_permissions = sorted(
            {
                scope
                for scopes in normalized_tool_permissions.values()
                for scope in scopes
            }
            - set(self.permissions)
        )
        if undeclared_permissions:
            raise ValueError(
                "tool permission mappings reference undeclared permissions: "
                + ", ".join(undeclared_permissions)
            )
        object.__setattr__(self, "tool_permissions", normalized_tool_permissions)
        compatible_versions = tuple(
            dict.fromkeys((*self.resume_compatible_versions, self.version))
        )
        object.__setattr__(
            self,
            "resume_compatible_versions",
            compatible_versions,
        )
        normalized_migrations = {
            str(schema).strip(): str(migration).strip()
            for schema, migration in self.state_migrations.items()
            if str(schema).strip() and str(migration).strip()
        }
        if len(normalized_migrations) != len(self.state_migrations):
            raise ValueError("state migration schema and handler ids must not be blank")
        object.__setattr__(self, "state_migrations", normalized_migrations)
        return self

    def can_resume_from(self, previous: "CapabilityManifest") -> bool:
        if previous.id != self.id:
            return False
        if previous.version not in self.resume_compatible_versions:
            return False
        return (
            previous.state_schema_version == self.state_schema_version
            or previous.state_schema_version in self.state_migrations
        )


class RuntimeGeneration(BaseModel):
    """Immutable snapshot frozen for all runs started by one runtime generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "harness-runtime-generation-v1"
    generation_id: str
    core_version: str = DEEPKEEL_VERSION
    core_contract: str = DEEPKEEL_CONTRACT_VERSION
    catalog_version: str = ""
    packages: tuple[CapabilityManifest, ...]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        manifests: tuple[CapabilityManifest, ...],
        *,
        catalog_version: str = "",
    ) -> "RuntimeGeneration":
        validate_manifest_set(manifests)
        encoded = json.dumps(
            {
                "core_version": DEEPKEEL_VERSION,
                "catalog_version": catalog_version,
                "packages": [
                    manifest.model_dump(mode="json")
                    for manifest in sorted(manifests, key=lambda item: item.id)
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
        return cls(
            generation_id=f"generation-{digest}",
            catalog_version=catalog_version,
            packages=tuple(sorted(manifests, key=lambda item: item.id)),
        )

    def package_versions(self) -> dict[str, str]:
        return {manifest.id: manifest.version for manifest in self.packages}

    def resume_compatibility_issues(
        self,
        previous: "RuntimeGeneration",
    ) -> tuple[str, ...]:
        current = {manifest.id: manifest for manifest in self.packages}
        issues: list[str] = []
        for old_manifest in previous.packages:
            new_manifest = current.get(old_manifest.id)
            if new_manifest is None:
                issues.append(f"{old_manifest.id}: package is no longer active")
            elif not new_manifest.can_resume_from(old_manifest):
                issues.append(
                    f"{old_manifest.id}: {old_manifest.version}/"
                    f"{old_manifest.state_schema_version} cannot resume on "
                    f"{new_manifest.version}/{new_manifest.state_schema_version}"
                )
        return tuple(issues)

    def supports_resume_from(self, previous: "RuntimeGeneration") -> bool:
        return not self.resume_compatibility_issues(previous)


class RuntimeGenerationManager:
    """Atomic activation and rollback for immutable package/catalog snapshots."""

    def __init__(self) -> None:
        self._generations: dict[str, RuntimeGeneration] = {}
        self._current_id = ""

    @property
    def current(self) -> RuntimeGeneration | None:
        return self._generations.get(self._current_id)

    def activate(
        self,
        manifests: tuple[CapabilityManifest, ...],
        *,
        catalog_version: str = "",
    ) -> RuntimeGeneration:
        generation = RuntimeGeneration.create(
            manifests,
            catalog_version=catalog_version,
        )
        self._generations[generation.generation_id] = generation
        self._current_id = generation.generation_id
        return generation

    def rollback(self, generation_id: str) -> RuntimeGeneration:
        try:
            generation = self._generations[generation_id]
        except KeyError as exc:
            raise KeyError(f"runtime generation is not available: {generation_id}") from exc
        self._current_id = generation_id
        return generation

    def get(self, generation_id: str) -> RuntimeGeneration | None:
        return self._generations.get(generation_id)


def load_capability_manifest(path: str | Path) -> CapabilityManifest:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
    elif source.suffix.lower() in {".yaml", ".yml"}:
        try:
            yaml = importlib.import_module("yaml")
        except ImportError as exc:
            raise RuntimeError(
                "YAML capability manifests require the optional PyYAML dependency"
            ) from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported capability manifest format: {source.suffix}")
    if not isinstance(payload, Mapping):
        raise ValueError("capability manifest must contain an object")
    return CapabilityManifest.model_validate(dict(payload))


def validate_manifest_set(manifests: tuple[CapabilityManifest, ...]) -> None:
    by_id: dict[str, CapabilityManifest] = {}
    ownership: dict[tuple[str, str], str] = {}
    issues: list[str] = []
    for manifest in manifests:
        if manifest.id in by_id:
            issues.append(f"duplicate package id: {manifest.id}")
        by_id[manifest.id] = manifest
        for kind in (
            "skills",
            "tools",
            "artifact_types",
            "subagents",
            "handoffs",
            "hooks",
            "context_contributors",
            "agent_entrypoints",
            "resources",
        ):
            for capability_id in getattr(manifest, kind):
                key = (kind, capability_id)
                owner = ownership.get(key)
                if owner and owner != manifest.id:
                    issues.append(
                        f"{kind} {capability_id!r} is declared by both "
                        f"{owner!r} and {manifest.id!r}"
                    )
                ownership[key] = manifest.id
    for manifest in manifests:
        for dependency_id, version_range in manifest.dependencies.items():
            dependency = by_id.get(dependency_id)
            if dependency is None:
                issues.append(
                    f"{manifest.id} requires missing package {dependency_id}"
                )
            elif not version_satisfies(dependency.version, version_range):
                issues.append(
                    f"{manifest.id} requires {dependency_id}{version_range}, "
                    f"installed {dependency.version}"
                )
        unknown_handoffs = sorted(set(manifest.handoffs) - set(manifest.tools))
        if unknown_handoffs:
            issues.append(
                f"{manifest.id} handoffs reference undeclared tools: "
                + ", ".join(unknown_handoffs)
            )
    if issues:
        raise ValueError("invalid capability manifest set: " + "; ".join(issues))


def version_satisfies(version: str, constraint: str) -> bool:
    normalized = str(constraint or "*").strip()
    if normalized in {"", "*"}:
        return True
    clauses = tuple(item.strip() for item in normalized.split(",") if item.strip())
    specifier = ",".join(
        clause
        if clause.startswith(("~=", "==", "!=", "<=", ">=", "<", ">", "==="))
        else f"=={clause}"
        for clause in clauses
    )
    try:
        return Version(str(version).strip()) in SpecifierSet(specifier)
    except (InvalidSpecifier, InvalidVersion):
        return False


def core_version_satisfies(contract: str, constraint: str) -> bool:
    """Accept the current package or an explicit same-contract upgrade bridge."""

    if version_satisfies(DEEPKEEL_VERSION, constraint):
        return True
    return any(
        version_satisfies(version, constraint)
        for version in _LEGACY_PACKAGE_VERSION_BRIDGES.get(contract, ())
    )


__all__ = [
    "CapabilityBudgetSpec",
    "CapabilityManifest",
    "RuntimeGeneration",
    "RuntimeGenerationManager",
    "load_capability_manifest",
    "validate_manifest_set",
    "core_version_satisfies",
    "version_satisfies",
]
