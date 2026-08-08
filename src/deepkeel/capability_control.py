from __future__ import annotations

from datetime import UTC, datetime
from threading import RLock
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from deepkeel.capability_manifest import (
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
    generations: tuple[RuntimeGeneration, ...] = ()
    active_generation_id: str = ""

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
        if not catalog_version and self.active_generation_id:
            generation = self.get_generation(self.active_generation_id)
            if generation is not None:
                return generation
        return RuntimeGeneration.create(
            self.active_manifests(),
            catalog_version=catalog_version or f"package-revision-{self.revision}",
        )

    def get_generation(self, generation_id: str) -> RuntimeGeneration | None:
        normalized = str(generation_id or "").strip()
        return next(
            (
                generation
                for generation in self.generations
                if generation.generation_id == normalized
            ),
            None,
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

    def reconcile(
        self,
        manifests: tuple[CapabilityManifest, ...],
        *,
        disable_missing: bool = True,
    ) -> CapabilityPackageSnapshot:
        """Atomically replace the active package generation.

        Package ownership can move between manifests during a deployment. A
        sequence of otherwise-valid per-package upgrades may therefore create
        an invalid intermediate generation. Reconciliation validates and saves
        only the final generation while preserving package history.
        """

        desired = {manifest.id: manifest for manifest in manifests}
        if len(desired) != len(manifests):
            raise ValueError("capability package reconciliation contains duplicate ids")
        validate_manifest_set(tuple(desired.values()))

        snapshot = self.store.load()
        now = datetime.now(UTC)
        records: list[CapabilityPackageRecord] = []
        remaining = dict(desired)
        for current in snapshot.packages:
            manifest = remaining.pop(current.manifest.id, None)
            if manifest is None:
                records.append(
                    current.model_copy(
                        update={
                            "enabled": False if disable_missing else current.enabled,
                            "updated_at": now
                            if disable_missing and current.enabled
                            else current.updated_at,
                        }
                    )
                )
                continue
            records.append(self._reconciled_record(current, manifest, now=now))

        records.extend(
            CapabilityPackageRecord(manifest=manifest, enabled=True)
            for manifest in remaining.values()
        )
        normalized = tuple(sorted(records, key=lambda item: item.manifest.id))
        if normalized == snapshot.packages:
            return snapshot
        return self._save(snapshot, normalized)

    def generation(
        self,
        generation_id: str = "",
        *,
        catalog_version: str = "",
    ) -> RuntimeGeneration:
        snapshot = self.store.load()
        if generation_id:
            generation = snapshot.get_generation(generation_id)
            if generation is None:
                raise KeyError(
                    f"runtime generation is not available: {generation_id}"
                )
            return generation
        return snapshot.runtime_generation(catalog_version=catalog_version)

    def resume_compatibility_issues(
        self,
        previous_generation_id: str,
    ) -> tuple[str, ...]:
        previous = self.generation(previous_generation_id)
        current = self.generation()
        return current.resume_compatibility_issues(previous)

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

    @staticmethod
    def _reconciled_record(
        current: CapabilityPackageRecord,
        manifest: CapabilityManifest,
        *,
        now: datetime,
    ) -> CapabilityPackageRecord:
        if current.manifest == manifest:
            if current.enabled:
                return current
            return current.model_copy(update={"enabled": True, "updated_at": now})
        if current.manifest.version == manifest.version:
            raise ValueError(
                "capability package content changed without a version bump: "
                f"{manifest.id}@{manifest.version}"
            )
        if version_satisfies(manifest.version, f">{current.manifest.version}"):
            history = (*current.history, current.manifest)
        else:
            selected = next(
                (item for item in current.history if item == manifest),
                None,
            )
            if selected is None:
                raise ValueError(
                    "capability package reconciliation cannot select unavailable version: "
                    f"{manifest.id}@{manifest.version}"
                )
            history = (
                *(item for item in current.history if item is not selected),
                current.manifest,
            )
        return current.model_copy(
            update={
                "manifest": manifest,
                "enabled": True,
                "history": history,
                "updated_at": now,
            }
        )

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
        next_revision = snapshot.revision + 1
        generation = RuntimeGeneration.create(
            tuple(record.manifest for record in normalized if record.enabled),
            catalog_version=f"package-revision-{next_revision}",
        )
        generations = (
            snapshot.generations
            if snapshot.get_generation(generation.generation_id) is not None
            else (*snapshot.generations, generation)
        )
        candidate = snapshot.model_copy(
            update={
                "packages": normalized,
                "generations": generations,
                "active_generation_id": generation.generation_id,
            }
        )
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
