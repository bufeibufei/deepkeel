from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from deepkeel.budget import BudgetLedger, BudgetRequest
from deepkeel.contracts import Observation, ToolCall, ToolResult
from deepkeel.capability_control import (
    CapabilityPackageConflict,
    CapabilityPackageManager,
    CapabilityPackageStore,
)
from deepkeel.capability_manifest import CapabilityManifest
from deepkeel.context_window import ContextSummaryCache, ContextSummaryRecord
from deepkeel.event_journal import EventJournalConflict, RuntimeEventJournal
from deepkeel.leases import RunLeaseConflict, RunLeaseStore
from deepkeel.persistence import DurableCheckpointStore
from deepkeel.events import envelope_runtime_event
from deepkeel.model import (
    ModelInvocation,
    ModelInvocationConflict,
    ModelInvocationEnvelope,
    ModelInvocationStore,
    ModelTurn,
)
from deepkeel.runtime_api import RuntimeEventEnvelope
from deepkeel.scope import RuntimeScope
from deepkeel.state_store import (
    RUN_SETTLED_EVENT,
    RuntimeStateConflict,
    RuntimeStateMutation,
    RuntimeStateStore,
)
from deepkeel.tools import ToolExecutionStore
from deepkeel.telemetry import TelemetryRecord, TraceQuery, TraceStore


def verify_budget_ledger_contract(
    ledger: BudgetLedger,
    *,
    run_id: str,
) -> None:
    """Assert idempotent accounting, rejection, peak usage, and monotonic restore."""

    first = ledger.consume(
        BudgetRequest(
            run_id=run_id,
            metric="model_calls",
            amount=1,
            limit=2,
            operation_id="model:1",
        )
    )
    replay = ledger.consume(
        BudgetRequest(
            run_id=run_id,
            metric="model_calls",
            amount=1,
            limit=2,
            operation_id="model:1",
        )
    )
    assert first.allowed is True
    assert replay == first
    assert ledger.snapshot(run_id).usage["model_calls"] == 1

    second = ledger.consume(
        BudgetRequest(
            run_id=run_id,
            metric="model_calls",
            amount=1,
            limit=2,
            operation_id="model:2",
        )
    )
    rejected = ledger.consume(
        BudgetRequest(
            run_id=run_id,
            metric="model_calls",
            amount=1,
            limit=2,
            operation_id="model:3",
        )
    )
    assert second.allowed is True and second.used == 2
    assert rejected.allowed is False and rejected.used == 2
    assert ledger.snapshot(run_id).usage["model_calls"] == 2

    ledger.consume(
        BudgetRequest(
            run_id=run_id,
            metric="tool_concurrency",
            amount=3,
            limit=4,
            operation_id="tools:batch:1",
            aggregation="max",
        )
    )
    lower_peak = ledger.consume(
        BudgetRequest(
            run_id=run_id,
            metric="tool_concurrency",
            amount=2,
            limit=4,
            operation_id="tools:batch:2",
            aggregation="max",
        )
    )
    assert lower_peak.allowed is True and lower_peak.used == 3

    ledger.restore(
        run_id,
        {
            "usage": {
                "model_calls": 1,
                "tool_concurrency": 4,
                "input_tokens": 128,
            }
        },
    )
    restored = ledger.snapshot(run_id).usage
    assert restored["model_calls"] == 2
    assert restored["tool_concurrency"] == 4
    assert restored["input_tokens"] == 128


def verify_trace_store_contract(store: TraceStore, *, run_id: str) -> None:
    first = TelemetryRecord(event_name="run.started", run_id=run_id, sequence=1)
    second = TelemetryRecord(event_name="tool.completed", run_id=run_id, sequence=2)
    store.record(first)
    store.record(first)
    store.record(second)

    page = store.query(TraceQuery(run_id=run_id, limit=10))
    assert [record.telemetry_id for record in page.records] == [
        first.telemetry_id,
        second.telemetry_id,
    ]
    assert page.truncated is False
    assert store.query(TraceQuery(run_id=run_id, limit=1)).truncated is True


def verify_context_summary_cache_contract(cache: ContextSummaryCache) -> None:
    record = ContextSummaryRecord(
        cache_key="profile:summary",
        source_fingerprint="fingerprint-v1",
        summary={"focus": "career"},
        summary_version="v1",
    )
    cache.put(record)
    assert cache.get(record.cache_key, record.source_fingerprint) == record
    assert cache.get(record.cache_key, "stale") is None
    cache.invalidate(record.cache_key)
    assert cache.get(record.cache_key, record.source_fingerprint) is None


def verify_capability_package_store_contract(
    store: CapabilityPackageStore,
    *,
    package_id: str = "",
) -> None:
    """Assert optimistic lifecycle persistence and immutable generation replay."""

    resolved_id = str(package_id or f"conformance.package.{uuid4().hex}").strip()
    manager = CapabilityPackageManager(store)
    initial = manager.inspect()
    installed = manager.install(
        CapabilityManifest(
            id=resolved_id,
            version="1.0.0",
            core_version="*",
            entrypoint="conformance:PackV1",
        )
    )
    first_generation_id = installed.active_generation_id
    assert manager.generation(first_generation_id).package_versions()[resolved_id] == "1.0.0"

    manager.disable(resolved_id)
    assert manager.inspect().get(resolved_id) is not None
    assert manager.inspect().get(resolved_id).enabled is False  # type: ignore[union-attr]
    manager.enable(resolved_id)
    manager.upgrade(
        CapabilityManifest(
            id=resolved_id,
            version="2.0.0",
            core_version="*",
            entrypoint="conformance:PackV2",
            resume_compatible_versions=("1.0.0",),
        )
    )
    assert manager.generation().package_versions()[resolved_id] == "2.0.0"
    assert manager.generation(first_generation_id).package_versions()[resolved_id] == "1.0.0"
    assert manager.resume_compatibility_issues(first_generation_id) == ()

    try:
        store.save(initial, expected_revision=initial.revision)
    except CapabilityPackageConflict:
        pass
    else:
        raise AssertionError("capability package store accepted a stale catalog revision")

    manager.rollback(resolved_id, version="1.0.0")
    assert manager.generation().package_versions()[resolved_id] == "1.0.0"
    manager.uninstall(resolved_id)
    assert manager.inspect().get(resolved_id) is None


def verify_runtime_event_projection_contract(
    projector: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Assert that a Host exposes flat canonical events after journal replay."""

    projected = projector(
        {
            "id": "journal-event-1",
            "run_id": "run-projection",
            "sequence": 7,
            "payload": {
                "event_type": "tool.started",
                "title": "Lookup",
                "payload": {
                    "tool_name": "example.lookup",
                    "tool_call": {"id": "call-1", "arguments": {"query": "test"}},
                },
            },
        }
    )
    assert projected["event_type"] == "tool.call.started"
    assert projected["source_event_type"] == "tool.started"
    assert projected["run_id"] == "run-projection"
    assert projected["sequence"] == 7
    assert projected["payload"]["tool_name"] == "example.lookup"
    assert "event_type" not in projected["payload"]

    answer_delta = projector(
        {"event_type": "model.delta", "payload": {"delta": "hello"}}
    )
    assert answer_delta["event_type"] == "answer.delta"
    assert answer_delta["payload"]["delta"] == "hello"


def verify_runtime_event_journal_contract(
    journal: RuntimeEventJournal,
    *,
    run_id: str,
) -> None:
    """Assert stable identities, monotonic cursors, and idempotent replay."""

    first = RuntimeEventEnvelope.model_validate(
        envelope_runtime_event(
            {"event_type": "run.created", "payload": {"phase": "created"}},
            run_id=run_id,
            thread_id=f"{run_id}:thread",
            turn_id=f"{run_id}:turn",
            sequence=1,
        )
    )
    third = RuntimeEventEnvelope.model_validate(
        envelope_runtime_event(
            {"event_type": "agent.reasoning", "payload": {"phase": "reasoning"}},
            run_id=run_id,
            thread_id=f"{run_id}:thread",
            turn_id=f"{run_id}:turn",
            sequence=3,
        )
    )
    journal.append(first)
    journal.append(third)
    assert journal.append(third).event_id == third.event_id
    assert journal.latest_sequence(run_id) == 3
    assert [event.sequence for event in journal.read_after(run_id, after_sequence=1)] == [3]

    conflicting = third.model_copy(update={"summary": "changed"})
    try:
        journal.append(conflicting)
    except EventJournalConflict:
        return
    raise AssertionError("runtime event journal accepted conflicting event content")


def verify_model_invocation_store_contract(
    store: ModelInvocationStore,
    *,
    run_id: str,
) -> None:
    """Assert atomic claim, durable settlement, and exact result replay."""

    envelope = ModelInvocationEnvelope(
        invocation_id=f"{run_id}:model:0:attempt:1",
        run_id=run_id,
        thread_id=f"{run_id}:thread",
        turn_id=f"{run_id}:turn",
        request=ModelInvocation(messages=[{"role": "user", "content": "hello"}]),
    )
    claim = store.claim(envelope, lease_seconds=30)
    assert claim.outcome == "acquired" and claim.claim_token
    assert store.claim(envelope, lease_seconds=30).outcome == "in_progress"

    result = ModelTurn(content="answer", finish_reason="stop")
    store.complete(
        envelope.invocation_id,
        claim_token=claim.claim_token,
        result=result,
    )
    replay = store.claim(envelope, lease_seconds=30)
    assert replay.outcome == "replay" and replay.result == result

    changed = envelope.model_copy(
        update={
            "request": ModelInvocation(
                messages=[{"role": "user", "content": "changed"}]
            )
        }
    )
    try:
        store.claim(changed, lease_seconds=30)
    except ModelInvocationConflict:
        return
    raise AssertionError("model invocation store accepted a changed request")


def verify_run_lease_store_contract(
    store: RunLeaseStore,
    *,
    run_id: str,
) -> None:
    """Assert exclusive ownership, renewal, fencing, and release semantics."""

    lease = store.claim(run_id, owner_id="worker-a", ttl_seconds=30)
    assert lease.run_id == run_id
    assert lease.owner_id == "worker-a"
    assert store.inspect(run_id) is not None
    try:
        store.claim(run_id, owner_id="worker-b", ttl_seconds=30)
    except RunLeaseConflict:
        pass
    else:
        raise AssertionError("run lease store allowed concurrent ownership")
    renewed = store.renew(lease, ttl_seconds=60)
    assert renewed.token == lease.token
    assert renewed.generation == lease.generation
    assert renewed.expires_at > lease.expires_at
    store.release(renewed)
    assert store.inspect(run_id) is None


def verify_runtime_state_store_contract(
    store: RuntimeStateStore,
    *,
    run_id: str,
    user_id: str = "",
    session: Any = None,
) -> None:
    """Assert atomic mutation, replay, and optimistic-conflict semantics."""

    assert str(getattr(store, "terminal_settlement_owner", "")) in {"runtime", "host"}

    first = RuntimeStateMutation(
        mutation_id=f"conformance:{uuid4().hex}",
        run_id=run_id,
        event_type="conformance.started",
        target_status="waiting_user_input",
        checkpoint_state={"phase": "waiting"},
        fence_token="worker-a-token",
        fence_generation=1,
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
        event_type=RUN_SETTLED_EVENT,
        target_status="completed",
        event_payload={"status": "completed"},
        checkpoint_state={"phase": "completed"},
        expected_version=receipt.version,
        expected_sequence=receipt.sequence,
        fence_token="worker-a-token",
        fence_generation=1,
    )
    settled = store.commit(second, session=session, user_id=user_id)
    assert settled.version > receipt.version
    assert settled.sequence > receipt.sequence
    snapshot = store.load_snapshot(run_id, session=session, user_id=user_id)
    assert snapshot.settled is True
    assert snapshot.settlement_status == "completed"
    assert snapshot.fence_generation == 1
    assert snapshot.fence_token == "worker-a-token"
    isolated_user_id = f"{user_id or 'conformance'}:isolated"
    try:
        isolated = store.load_snapshot(
            run_id,
            session=session,
            user_id=isolated_user_id,
        )
    except Exception as exc:
        if not _is_scope_denial(exc):
            raise
    else:
        if isolated.version or isolated.sequence or isolated.settled:
            raise AssertionError("runtime state store leaked a run across user scopes")

    commit_scoped = getattr(store, "commit_scoped", None)
    load_scoped = getattr(store, "load_snapshot_scoped", None)
    if callable(commit_scoped) and callable(load_scoped):
        tenant_run_id = f"{run_id}:tenant-scope"
        tenant_a = RuntimeScope(tenant_id="tenant-a", user_id=user_id or "user-a")
        tenant_b = RuntimeScope(tenant_id="tenant-b", user_id=user_id or "user-a")
        commit_scoped(
            RuntimeStateMutation(
                mutation_id=f"conformance:{uuid4().hex}",
                run_id=tenant_run_id,
                event_type="conformance.scoped",
                target_status="task_running",
            ),
            scope=tenant_a,
            session=session,
        )
        scoped_a = load_scoped(tenant_run_id, scope=tenant_a, session=session)
        scoped_b = load_scoped(tenant_run_id, scope=tenant_b, session=session)
        if not scoped_a.version or scoped_b.version or scoped_b.sequence:
            raise AssertionError("runtime state store leaked a run across tenant scopes")

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
        pass
    else:
        raise AssertionError("runtime state store accepted a stale mutation")

    fence_run_id = f"{run_id}:fence"
    current_owner = RuntimeStateMutation(
        mutation_id=f"conformance:{uuid4().hex}",
        run_id=fence_run_id,
        event_type="conformance.claimed",
        target_status="task_running",
        fence_token="worker-b-token",
        fence_generation=2,
    )
    store.commit(current_owner, session=session, user_id=user_id)
    stale_owner = RuntimeStateMutation(
        mutation_id=f"conformance:{uuid4().hex}",
        run_id=fence_run_id,
        event_type="conformance.stale-owner",
        target_status="task_running",
        fence_token="worker-a-token",
        fence_generation=1,
    )
    try:
        store.commit(stale_owner, session=session, user_id=user_id)
    except RuntimeStateConflict:
        return
    raise AssertionError("runtime state store accepted an obsolete fencing generation")


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
    isolated_user_id = f"{user_id or 'conformance'}:isolated"
    try:
        isolated = store.load(run_id, session=session, user_id=isolated_user_id)
    except Exception as exc:
        if not _is_scope_denial(exc):
            raise
    else:
        if isolated is not None:
            raise AssertionError("durable checkpoint store leaked a run across user scopes")
    assert run_id in store.list_ids(session=session, user_id=user_id, limit=100)
    store.delete(run_id, session=session, user_id=user_id)
    assert not store.exists(run_id, session=session, user_id=user_id)
    assert store.load(run_id, session=session, user_id=user_id) is None


def _is_scope_denial(exc: Exception) -> bool:
    return isinstance(exc, (LookupError, PermissionError)) or int(
        getattr(exc, "status_code", 0) or 0
    ) in {403, 404}
