from __future__ import annotations

import hashlib
import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness_core.version import (
    HARNESS_CORE_CONTRACT_VERSION,
    HARNESS_CORE_VERSION,
)


class CapabilityManifest(BaseModel):
    """Declarative, versioned package boundary consumed by the runtime control plane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "harness-capability-manifest-v1"
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    core_contract: str = HARNESS_CORE_CONTRACT_VERSION
    core_version: str = f">={HARNESS_CORE_VERSION}"
    entrypoint: str = Field(min_length=1)
    dependencies: dict[str, str] = Field(default_factory=dict)
    skills: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    subagents: tuple[str, ...] = ()
    handoffs: tuple[str, ...] = ()
    hooks: tuple[str, ...] = ()
    mcp_servers: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    memory_namespaces: tuple[str, ...] = ()
    ui_surfaces: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_and_validate(self) -> "CapabilityManifest":
        if self.core_contract != HARNESS_CORE_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported core contract {self.core_contract!r}; "
                f"expected {HARNESS_CORE_CONTRACT_VERSION!r}"
            )
        if not version_satisfies(HARNESS_CORE_VERSION, self.core_version):
            raise ValueError(
                f"core {HARNESS_CORE_VERSION} does not satisfy {self.core_version!r}"
            )
        for field_name in (
            "skills",
            "tools",
            "subagents",
            "handoffs",
            "hooks",
            "mcp_servers",
            "permissions",
            "memory_namespaces",
            "ui_surfaces",
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
        return self


class RuntimeGeneration(BaseModel):
    """Immutable snapshot frozen for all runs started by one runtime generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "harness-runtime-generation-v1"
    generation_id: str
    core_version: str = HARNESS_CORE_VERSION
    core_contract: str = HARNESS_CORE_CONTRACT_VERSION
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
                "core_version": HARNESS_CORE_VERSION,
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
        for kind in ("skills", "tools", "subagents", "handoffs", "hooks"):
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
    actual = _version_tuple(version)
    for clause in (item.strip() for item in normalized.split(",") if item.strip()):
        operator = "=="
        expected_text = clause
        for candidate in (">=", "<=", "==", ">", "<"):
            if clause.startswith(candidate):
                operator = candidate
                expected_text = clause[len(candidate) :].strip()
                break
        expected = _version_tuple(expected_text)
        matches = {
            ">=": actual >= expected,
            "<=": actual <= expected,
            "==": actual == expected,
            ">": actual > expected,
            "<": actual < expected,
        }[operator]
        if not matches:
            return False
    return True


def _version_tuple(value: str) -> tuple[int, int, int]:
    numbers: list[int] = []
    for part in str(value or "0").split(".")[:3]:
        digits = "".join(char for char in part if char.isdigit())
        numbers.append(int(digits or 0))
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers)  # type: ignore[return-value]


__all__ = [
    "CapabilityManifest",
    "RuntimeGeneration",
    "RuntimeGenerationManager",
    "load_capability_manifest",
    "validate_manifest_set",
    "version_satisfies",
]
