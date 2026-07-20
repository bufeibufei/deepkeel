from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness_core.adapter_sdk import (
    StateMigrationError,
    StateMigrationRegistry,
    default_state_migrations,
)
from harness_core.persistence import (
    CHECKPOINT_SCHEMA_VERSION,
    checkpoint_from_durable_state,
    restore_run_context,
)


def test_registry_applies_a_deterministic_multistep_migration_chain() -> None:
    registry = StateMigrationRegistry()
    registry.register(
        "checkpoint",
        "checkpoint-v0",
        "checkpoint-v1",
        lambda state: {**state, "schema_version": "checkpoint-v1", "first": True},
    )
    registry.register(
        "checkpoint",
        "checkpoint-v1",
        CHECKPOINT_SCHEMA_VERSION,
        lambda state: {**state, "schema_version": CHECKPOINT_SCHEMA_VERSION, "second": True},
    )

    migrated = registry.migrate(
        "checkpoint",
        {"schema_version": "checkpoint-v0"},
        target_version=CHECKPOINT_SCHEMA_VERSION,
    )

    assert registry.path("checkpoint", "checkpoint-v0", CHECKPOINT_SCHEMA_VERSION) == (
        "checkpoint-v0",
        "checkpoint-v1",
        CHECKPOINT_SCHEMA_VERSION,
    )
    assert migrated["first"] is True
    assert migrated["second"] is True


def test_registry_rejects_missing_paths_and_invalid_handler_versions() -> None:
    registry = StateMigrationRegistry()
    with pytest.raises(StateMigrationError, match="no migration path"):
        registry.migrate("checkpoint", {"schema_version": "old"}, target_version="new")

    registry.register("checkpoint", "old", "new", lambda state: state)
    with pytest.raises(StateMigrationError, match="must set schema_version"):
        registry.migrate("checkpoint", {"schema_version": "old"}, target_version="new")


def test_checkpoint_recovery_uses_registered_outer_and_inner_migrations() -> None:
    registry = StateMigrationRegistry()
    registry.register(
        "durable_checkpoint",
        "durable-v1",
        "harness-durable-checkpoint-v2",
        lambda state: {**state, "schema_version": "harness-durable-checkpoint-v2"},
    )
    registry.register(
        "checkpoint",
        "checkpoint-v1",
        CHECKPOINT_SCHEMA_VERSION,
        lambda state: {**state, "schema_version": CHECKPOINT_SCHEMA_VERSION},
    )
    durable = {
        "schema_version": "durable-v1",
        "checkpoint": {
            "schema_version": "checkpoint-v1",
            "run_id": "run-1",
            "messages": [],
            "observations": [],
            "artifacts": [],
            "pending_action": {"action_type": "confirm", "tool_call_id": "call-1"},
        },
    }

    checkpoint = checkpoint_from_durable_state(durable, migrations=registry)
    context = restore_run_context(
        checkpoint=checkpoint,
        resume_payload={"status": "succeeded", "summary": "confirmed"},
        run_id="run-1",
        thread_id="thread-1",
        turn_id="turn-1",
        user_id="user-1",
        migrations=registry,
    )

    assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert context.observations[-1].summary == "confirmed"


def test_shipped_checkpoint_fixture_migrates_and_resumes() -> None:
    fixture = _state_fixture("checkpoint-v1-waiting-user-action.json")
    context = restore_run_context(
        checkpoint=fixture,
        resume_payload={"status": "succeeded", "summary": "confirmed"},
        run_id="fixture-waiting-run",
        thread_id="fixture-thread",
        turn_id="fixture-turn",
        user_id="fixture-user",
        migrations=default_state_migrations(),
    )

    assert context.step_count == 2
    assert context.observations[-1].summary == "confirmed"
    assert context.metadata["source_version"] == "3.1.0"


def test_shipped_durable_fixture_migrates_outer_and_inner_contracts() -> None:
    durable = _state_fixture("durable-v1-task-running.json")
    checkpoint = checkpoint_from_durable_state(
        durable,
        migrations=default_state_migrations(),
    )

    assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["pending_async"]["tool_call_id"] == "call-async-1"
    assert checkpoint["metadata"]["source_version"] == "3.1.0"


def _state_fixture(name: str) -> dict:
    path = Path(__file__).parent / "fixtures" / "state" / name
    return json.loads(path.read_text(encoding="utf-8"))
