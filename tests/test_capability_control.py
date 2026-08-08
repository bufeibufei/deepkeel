from __future__ import annotations

import pytest

from deepkeel.extension_sdk import (
    CapabilityBudgetSpec,
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
    subagents: tuple[str, ...] = (),
) -> CapabilityManifest:
    return CapabilityManifest(
        id=package_id,
        version=version,
        core_version="*",
        entrypoint=f"{package_id}:Pack",
        dependencies=dependencies or {},
        tools=tools,
        subagents=subagents,
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
    assert manager.generation(first_generation.generation_id) == first_generation


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


def test_manifest_budget_and_resume_compatibility_are_portable() -> None:
    previous = _manifest("demo.workflow", "1.0.0")
    compatible = CapabilityManifest(
        id="demo.workflow",
        version="2.0.0",
        core_version="*",
        entrypoint="demo.workflow:Pack",
        budget=CapabilityBudgetSpec(
            max_model_calls=4,
            max_tool_calls=8,
            max_elapsed_seconds=120,
        ),
        state_schema_version="2",
        resume_compatible_versions=("1.0.0",),
        state_migrations={"1": "demo.workflow:migrate_v1_to_v2"},
    )
    incompatible = compatible.model_copy(
        update={
            "resume_compatible_versions": ("2.0.0",),
            "state_migrations": {},
        }
    )

    assert compatible.budget.limits() == {
        "max_model_calls": 4,
        "max_tool_calls": 8,
        "max_elapsed_seconds": 120.0,
    }
    assert compatible.can_resume_from(previous) is True
    assert incompatible.can_resume_from(previous) is False


def test_package_manager_reports_generation_resume_compatibility() -> None:
    manager = CapabilityPackageManager(InMemoryCapabilityPackageStore())
    first = manager.install(_manifest("demo.workflow", "1.0.0"))
    first_generation_id = first.active_generation_id
    manager.upgrade(
        CapabilityManifest(
            id="demo.workflow",
            version="2.0.0",
            core_version="*",
            entrypoint="demo.workflow:Pack",
            resume_compatible_versions=("1.0.0",),
        )
    )

    assert manager.resume_compatibility_issues(first_generation_id) == ()

    incompatible_manager = CapabilityPackageManager(
        InMemoryCapabilityPackageStore()
    )
    first = incompatible_manager.install(_manifest("demo.workflow", "1.0.0"))
    incompatible_manager.upgrade(_manifest("demo.workflow", "2.0.0"))

    assert incompatible_manager.resume_compatibility_issues(
        first.active_generation_id
    ) == (
        "demo.workflow: 1.0.0/1 cannot resume on 2.0.0/1",
    )


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


def test_reconcile_atomically_transfers_capability_ownership() -> None:
    manager = CapabilityPackageManager(InMemoryCapabilityPackageStore())
    manager.install(
        _manifest("demo.orchestration", "1.0.0", subagents=("demo.reviewer",))
    )
    manager.install(_manifest("demo.domain", "1.0.0"))

    snapshot = manager.reconcile(
        (
            _manifest("demo.orchestration", "1.1.0"),
            _manifest("demo.domain", "1.1.0", subagents=("demo.reviewer",)),
        )
    )

    assert snapshot.revision == 3
    assert snapshot.get("demo.orchestration").manifest.subagents == ()
    assert snapshot.get("demo.domain").manifest.subagents == ("demo.reviewer",)
    assert [
        item.version for item in snapshot.get("demo.orchestration").history
    ] == ["1.0.0"]
    assert [item.version for item in snapshot.get("demo.domain").history] == [
        "1.0.0"
    ]


def test_reconcile_rejects_same_version_content_drift() -> None:
    manager = CapabilityPackageManager(InMemoryCapabilityPackageStore())
    manager.install(_manifest("demo.domain", "1.0.0"))

    with pytest.raises(ValueError, match="content changed without a version bump"):
        manager.reconcile(
            (_manifest("demo.domain", "1.0.0", tools=("demo.changed",)),)
        )
