from __future__ import annotations

from typing import Any
from uuid import uuid4

from harness_core.contracts import Observation, ToolCall, ToolResult
from harness_core.persistence import DurableCheckpointStore
from harness_core.state_store import (
    RuntimeStateConflict,
    RuntimeStateMutation,
    RuntimeStateStore,
)
from harness_core.tools import ToolExecutionStore


def verify_runtime_state_store_contract(
    store: RuntimeStateStore,
    *,
    run_id: str,
    user_id: str = "",
    session: Any = None,
) -> None:
    """Assert atomic mutation, replay, and optimistic-conflict semantics."""

    first = RuntimeStateMutation(
        mutation_id=f"conformance:{uuid4().hex}",
        run_id=run_id,
        event_type="conformance.started",
        target_status="waiting_user_input",
        checkpoint_state={"phase": "waiting"},
    )
    receipt = store.commit(first, session=session, user_id=user_id)
    replay = store.commit(first, session=session, user_id=user_id)
    assert replay.mutation_id == receipt.mutation_id
    assert replay.version == receipt.version
    assert replay.sequence == receipt.sequence
    assert replay.replayed is True

    second = RuntimeStateMutation(
        mutation_id=f"conformance:{uuid4().hex}",
        run_id=run_id,
        event_type="conformance.completed",
        target_status="completed",
        checkpoint_state={"phase": "completed"},
        expected_version=receipt.version,
        expected_sequence=receipt.sequence,
    )
    settled = store.commit(second, session=session, user_id=user_id)
    assert settled.version > receipt.version
    assert settled.sequence > receipt.sequence

    stale = RuntimeStateMutation(
        mutation_id=f"conformance:{uuid4().hex}",
        run_id=run_id,
        event_type="conformance.stale",
        target_status="failed",
        expected_version=receipt.version,
    )
    try:
        store.commit(stale, session=session, user_id=user_id)
    except RuntimeStateConflict:
        return
    raise AssertionError("runtime state store accepted a stale mutation")


def verify_tool_execution_store_contract(
    store: ToolExecutionStore,
    *,
    run_id: str,
) -> None:
    """Assert exclusive claim, durable settlement, and replay semantics."""

    call_id = f"conformance:{uuid4().hex}"
    call = ToolCall(
        id=call_id,
        name="conformance.echo",
        arguments={"value": "ok"},
        idempotency_key=call_id,
    )
    claim = store.claim(
        run_id=run_id,
        call=call,
        lease_seconds=30,
        max_attempts=2,
        reexecution_safe=True,
    )
    assert claim.status == "claimed"

    concurrent = store.claim(
        run_id=run_id,
        call=call,
        lease_seconds=30,
        max_attempts=2,
        reexecution_safe=True,
    )
    assert concurrent.status == "busy"

    result = ToolResult(
        call=call,
        status="succeeded",
        summary="Conformance execution completed.",
        data={"value": "ok"},
        observation=Observation(
            id=f"{call.id}:observation",
            run_id=run_id,
            tool_call_id=call.id,
            source=call.name,
            status="succeeded",
            summary="Conformance execution completed.",
            data={"value": "ok"},
        ),
    )
    store.settle(claim, result)
    replay_claim = store.claim(
        run_id=run_id,
        call=call,
        lease_seconds=30,
        max_attempts=2,
        reexecution_safe=True,
    )
    assert replay_claim.status == "replay"
    replay = store.replay(replay_claim)
    assert replay.model_dump(mode="json") == result.model_dump(mode="json")


def verify_durable_checkpoint_store_contract(
    store: DurableCheckpointStore,
    *,
    run_id: str,
    user_id: str = "",
    session: Any = None,
) -> None:
    """Assert isolation, defensive copies, listing, and deletion semantics."""

    state = {"schema_version": "harness-durable-checkpoint-v2", "phase": "waiting"}
    store.save(run_id, state, session=session, user_id=user_id)
    state["phase"] = "mutated-after-save"
    assert store.exists(run_id, session=session, user_id=user_id)
    loaded = store.load(run_id, session=session, user_id=user_id)
    assert loaded is not None and loaded["phase"] == "waiting"
    loaded["phase"] = "mutated-after-load"
    reloaded = store.load(run_id, session=session, user_id=user_id)
    assert reloaded is not None and reloaded["phase"] == "waiting"
    assert run_id in store.list_ids(session=session, user_id=user_id, limit=100)
    store.delete(run_id, session=session, user_id=user_id)
    assert not store.exists(run_id, session=session, user_id=user_id)
    assert store.load(run_id, session=session, user_id=user_id) is None
