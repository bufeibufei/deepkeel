from __future__ import annotations

import pytest

from harness_core.extension_sdk import (
    CapabilityManifest,
    CapabilityPackageConflict,
    CapabilityPackageManager,
    CapabilityPackageSnapshot,
    InMemoryCapabilityPackageStore,
)


def _manifest(
    package_id: str,
    version: str = "1.0.0",
    *,
    dependencies: dict[str, str] | None = None,
    tools: tuple[str, ...] = (),
) -> CapabilityManifest:
    return CapabilityManifest(
        id=package_id,
        version=version,
        core_version="*",
        entrypoint=f"{package_id}:Pack",
        dependencies=dependencies or {},
        tools=tools,
    )


def test_package_lifecycle_builds_immutable_runtime_generations() -> None:
    manager = CapabilityPackageManager(InMemoryCapabilityPackageStore())

    foundation = manager.install(_manifest("demo.foundation", "1.0.0"))
    first_generation = foundation.runtime_generation()
    manager.install(
        _manifest(
            "demo.workflow",
            dependencies={"demo.foundation": ">=1.0.0"},
            tools=("demo.run",),
        )
    )
    second_generation = manager.generation()

    assert foundation.revision == 1
    assert first_generation.package_versions() == {"demo.foundation": "1.0.0"}
    assert second_generation.package_versions() == {
        "demo.foundation": "1.0.0",
        "demo.workflow": "1.0.0",
    }
    assert first_generation.generation_id != second_generation.generation_id


def test_disable_and_uninstall_fail_closed_when_dependents_are_active() -> None:
    manager = CapabilityPackageManager(InMemoryCapabilityPackageStore())
    manager.install(_manifest("demo.foundation"))
    manager.install(
        _manifest(
            "demo.workflow",
            dependencies={"demo.foundation": ">=1.0.0"},
        )
    )

    with pytest.raises(ValueError, match="requires missing package demo.foundation"):
        manager.disable("demo.foundation")
    with pytest.raises(ValueError, match="requires missing package demo.foundation"):
        manager.uninstall("demo.foundation")

    manager.disable("demo.workflow")
    snapshot = manager.disable("demo.foundation")

    assert snapshot.active_manifests() == ()


def test_upgrade_and_rollback_preserve_version_history() -> None:
    manager = CapabilityPackageManager(InMemoryCapabilityPackageStore())
    manager.install(_manifest("demo.workflow", "1.0.0"))

    upgraded = manager.upgrade(_manifest("demo.workflow", "2.0.0"))
    record = upgraded.get("demo.workflow")
    assert record is not None
    assert record.manifest.version == "2.0.0"
    assert [item.version for item in record.history] == ["1.0.0"]

    rolled_back = manager.rollback("demo.workflow")
    record = rolled_back.get("demo.workflow")
    assert record is not None
    assert record.manifest.version == "1.0.0"
    assert [item.version for item in record.history] == ["2.0.0"]


def test_upgrade_rejects_downgrades_and_dependency_breakage() -> None:
    manager = CapabilityPackageManager(InMemoryCapabilityPackageStore())
    manager.install(_manifest("demo.foundation", "2.0.0"))
    manager.install(
        _manifest(
            "demo.workflow",
            dependencies={"demo.foundation": ">=2.0.0"},
        )
    )

    with pytest.raises(ValueError, match="must increase version"):
        manager.upgrade(_manifest("demo.foundation", "1.0.0"))


def test_store_uses_optimistic_concurrency() -> None:
    store = InMemoryCapabilityPackageStore()
    initial = store.load()
    store.save(initial, expected_revision=0)

    with pytest.raises(CapabilityPackageConflict):
        store.save(CapabilityPackageSnapshot(), expected_revision=0)
