from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable


class StateMigrationError(ValueError):
    """Raised when versioned runtime state cannot reach its target schema."""


StateMigrationHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class StateMigration:
    state_kind: str
    from_version: str
    to_version: str
    handler: StateMigrationHandler


class StateMigrationRegistry:
    """Explicit, deterministic migration chains for persisted Core contracts."""

    def __init__(self) -> None:
        self._migrations: dict[tuple[str, str], StateMigration] = {}

    def register(
        self,
        state_kind: str,
        from_version: str,
        to_version: str,
        handler: StateMigrationHandler,
    ) -> None:
        kind = _required(state_kind, "state_kind")
        source = _required(from_version, "from_version")
        target = _required(to_version, "to_version")
        if source == target:
            raise ValueError("migration versions must differ")
        key = (kind, source)
        if key in self._migrations:
            raise ValueError(f"migration already registered for {kind}:{source}")
        self._migrations[key] = StateMigration(kind, source, target, handler)

    def migrate(
        self,
        state_kind: str,
        payload: dict[str, Any],
        *,
        target_version: str,
    ) -> dict[str, Any]:
        kind = _required(state_kind, "state_kind")
        target = _required(target_version, "target_version")
        current = copy.deepcopy(payload)
        version = str(current.get("schema_version") or "").strip()
        if not version:
            raise StateMigrationError(f"{kind} schema_version is missing")
        visited: set[str] = set()
        while version != target:
            if version in visited:
                raise StateMigrationError(f"migration cycle detected for {kind}:{version}")
            visited.add(version)
            migration = self._migrations.get((kind, version))
            if migration is None:
                raise StateMigrationError(
                    f"no migration path for {kind} from {version} to {target}"
                )
            migrated = migration.handler(copy.deepcopy(current))
            if not isinstance(migrated, dict):
                raise StateMigrationError(
                    f"migration {kind}:{version} must return a mapping"
                )
            migrated_version = str(migrated.get("schema_version") or "").strip()
            if migrated_version != migration.to_version:
                raise StateMigrationError(
                    f"migration {kind}:{version} must set schema_version "
                    f"to {migration.to_version}"
                )
            current = migrated
            version = migrated_version
        return current

    def path(self, state_kind: str, from_version: str, target_version: str) -> tuple[str, ...]:
        kind = _required(state_kind, "state_kind")
        version = _required(from_version, "from_version")
        target = _required(target_version, "target_version")
        path = [version]
        visited: set[str] = set()
        while version != target:
            if version in visited:
                raise StateMigrationError(f"migration cycle detected for {kind}:{version}")
            visited.add(version)
            migration = self._migrations.get((kind, version))
            if migration is None:
                raise StateMigrationError(
                    f"no migration path for {kind} from {version} to {target}"
                )
            version = migration.to_version
            path.append(version)
        return tuple(path)


def default_state_migrations() -> StateMigrationRegistry:
    """Migration history shipped by Core for its persisted public contracts."""

    registry = StateMigrationRegistry()
    registry.register(
        "checkpoint",
        "harness-checkpoint-v1",
        "harness-checkpoint-v2",
        _upgrade_checkpoint_v1,
    )
    registry.register(
        "durable_checkpoint",
        "harness-durable-checkpoint-v1",
        "harness-durable-checkpoint-v2",
        _upgrade_durable_checkpoint_v1,
    )
    registry.register(
        "runtime",
        "harness-runtime-v1",
        "harness-runtime-v2",
        _upgrade_runtime_v1,
    )
    return registry


def _upgrade_checkpoint_v1(state: dict[str, Any]) -> dict[str, Any]:
    return {
        **state,
        "schema_version": "harness-checkpoint-v2",
        "messages": list(state.get("messages") or []),
        "observations": list(state.get("observations") or []),
        "artifacts": list(state.get("artifacts") or []),
        "metadata": dict(state.get("metadata") or {}),
        "budget_state": dict(state.get("budget_state") or {}),
    }


def _upgrade_durable_checkpoint_v1(state: dict[str, Any]) -> dict[str, Any]:
    return {**state, "schema_version": "harness-durable-checkpoint-v2"}


def _upgrade_runtime_v1(state: dict[str, Any]) -> dict[str, Any]:
    return {**state, "schema_version": "harness-runtime-v2"}


def _required(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized
