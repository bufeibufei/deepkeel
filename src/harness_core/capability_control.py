from __future__ import annotations

from datetime import UTC, datetime
from threading import RLock
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from harness_core.capability_manifest import (
    CapabilityManifest,
    RuntimeGeneration,
    validate_manifest_set,
    version_satisfies,
)


class CapabilityPackageConflict(RuntimeError):
    """Raised when a package catalog changed during a control-plane update."""


class CapabilityPackageRecord(BaseModel):
    """Installed package metadata independent from a process-local pack object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: CapabilityManifest
    enabled: bool = True
    history: tuple[CapabilityManifest, ...] = ()
    installed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CapabilityPackageSnapshot(BaseModel):
    """Optimistically versioned package catalog used to build one runtime generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int = Field(default=0, ge=0)
    packages: tuple[CapabilityPackageRecord, ...] = ()

    def get(self, package_id: str) -> CapabilityPackageRecord | None:
        normalized = str(package_id or "").strip()
        return next(
            (
                record
                for record in self.packages
                if record.manifest.id == normalized
            ),
            None,
        )

    def active_manifests(self) -> tuple[CapabilityManifest, ...]:
        return tuple(
            record.manifest
            for record in sorted(
                self.packages,
                key=lambda item: item.manifest.id,
            )
            if record.enabled
        )

    def runtime_generation(self, *, catalog_version: str = "") -> RuntimeGeneration:
        return RuntimeGeneration.create(
            self.active_manifests(),
            catalog_version=catalog_version or f"package-revision-{self.revision}",
        )


class CapabilityPackageStore(Protocol):
    """Persistence port for package lifecycle state."""

    def load(self) -> CapabilityPackageSnapshot: ...

    def save(
        self,
        snapshot: CapabilityPackageSnapshot,
        *,
        expected_revision: int,
    ) -> CapabilityPackageSnapshot: ...


class InMemoryCapabilityPackageStore:
    """Thread-safe reference adapter with optimistic concurrency semantics."""

    def __init__(self) -> None:
        self._snapshot = CapabilityPackageSnapshot()
        self._lock = RLock()

    def load(self) -> CapabilityPackageSnapshot:
        with self._lock:
            return self._snapshot.model_copy(deep=True)

    def save(
        self,
        snapshot: CapabilityPackageSnapshot,
        *,
        expected_revision: int,
    ) -> CapabilityPackageSnapshot:
        with self._lock:
            if self._snapshot.revision != expected_revision:
                raise CapabilityPackageConflict(
                    "capability package catalog changed "
                    f"({self._snapshot.revision} != {expected_revision})"
                )
            stored = snapshot.model_copy(
                update={"revision": expected_revision + 1},
                deep=True,
            )
            self._snapshot = stored
            return stored.model_copy(deep=True)


class CapabilityPackageManager:
    """Lifecycle service for install, activation, upgrade, and rollback."""

    def __init__(self, store: CapabilityPackageStore) -> None:
        self.store = store

    def inspect(self) -> CapabilityPackageSnapshot:
        return self.store.load()

    def install(
        self,
        manifest: CapabilityManifest,
        *,
        enabled: bool = True,
    ) -> CapabilityPackageSnapshot:
        snapshot = self.store.load()
        if snapshot.get(manifest.id) is not None:
            raise ValueError(f"capability package is already installed: {manifest.id}")
        record = CapabilityPackageRecord(manifest=manifest, enabled=enabled)
        return self._save(snapshot, (*snapshot.packages, record))

    def enable(self, package_id: str) -> CapabilityPackageSnapshot:
        return self._replace_enabled(package_id, enabled=True)

    def disable(self, package_id: str) -> CapabilityPackageSnapshot:
        return self._replace_enabled(package_id, enabled=False)

    def upgrade(self, manifest: CapabilityManifest) -> CapabilityPackageSnapshot:
        snapshot = self.store.load()
        current = self._require(snapshot, manifest.id)
        if manifest.version == current.manifest.version:
            raise ValueError(
                f"capability package already uses version {manifest.version}: {manifest.id}"
            )
        if not version_satisfies(manifest.version, f">{current.manifest.version}"):
            raise ValueError(
                f"capability package upgrade must increase version "
                f"({current.manifest.version} -> {manifest.version})"
            )
        replacement = current.model_copy(
            update={
                "manifest": manifest,
                "history": (*current.history, current.manifest),
                "updated_at": datetime.now(UTC),
            }
        )
        return self._save(snapshot, self._replace(snapshot, replacement))

    def rollback(
        self,
        package_id: str,
        *,
        version: str = "",
    ) -> CapabilityPackageSnapshot:
        snapshot = self.store.load()
        current = self._require(snapshot, package_id)
        candidates = list(current.history)
        if version:
            selected = next(
                (item for item in candidates if item.version == version),
                None,
            )
            if selected is None:
                raise KeyError(
                    f"capability package version is not available: {package_id}@{version}"
                )
        elif candidates:
            selected = candidates[-1]
        else:
            raise KeyError(f"capability package has no rollback version: {package_id}")
        remaining = tuple(item for item in candidates if item is not selected)
        replacement = current.model_copy(
            update={
                "manifest": selected,
                "history": (*remaining, current.manifest),
                "updated_at": datetime.now(UTC),
            }
        )
        return self._save(snapshot, self._replace(snapshot, replacement))

    def uninstall(self, package_id: str) -> CapabilityPackageSnapshot:
        snapshot = self.store.load()
        self._require(snapshot, package_id)
        remaining = tuple(
            record
            for record in snapshot.packages
            if record.manifest.id != package_id
        )
        return self._save(snapshot, remaining)

    def generation(self, *, catalog_version: str = "") -> RuntimeGeneration:
        return self.store.load().runtime_generation(catalog_version=catalog_version)

    def _replace_enabled(
        self,
        package_id: str,
        *,
        enabled: bool,
    ) -> CapabilityPackageSnapshot:
        snapshot = self.store.load()
        current = self._require(snapshot, package_id)
        if current.enabled == enabled:
            return snapshot
        replacement = current.model_copy(
            update={"enabled": enabled, "updated_at": datetime.now(UTC)}
        )
        return self._save(snapshot, self._replace(snapshot, replacement))

    def _save(
        self,
        snapshot: CapabilityPackageSnapshot,
        packages: tuple[CapabilityPackageRecord, ...],
    ) -> CapabilityPackageSnapshot:
        normalized = tuple(
            sorted(packages, key=lambda item: item.manifest.id)
        )
        validate_manifest_set(
            tuple(record.manifest for record in normalized if record.enabled)
        )
        candidate = snapshot.model_copy(update={"packages": normalized})
        return self.store.save(candidate, expected_revision=snapshot.revision)

    @staticmethod
    def _replace(
        snapshot: CapabilityPackageSnapshot,
        replacement: CapabilityPackageRecord,
    ) -> tuple[CapabilityPackageRecord, ...]:
        return tuple(
            replacement
            if record.manifest.id == replacement.manifest.id
            else record
            for record in snapshot.packages
        )

    @staticmethod
    def _require(
        snapshot: CapabilityPackageSnapshot,
        package_id: str,
    ) -> CapabilityPackageRecord:
        record = snapshot.get(package_id)
        if record is None:
            raise KeyError(f"capability package is not installed: {package_id}")
        return record


__all__ = [
    "CapabilityPackageConflict",
    "CapabilityPackageManager",
    "CapabilityPackageRecord",
    "CapabilityPackageSnapshot",
    "CapabilityPackageStore",
    "InMemoryCapabilityPackageStore",
]
