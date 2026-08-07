from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Iterable, Protocol, TypeVar

from deepkeel.event_journal import RuntimeEventJournal
from deepkeel.leases import RunLease, RunLeaseStore
from deepkeel.persistence import DurableCheckpointStore
from deepkeel.runtime_api import RuntimeEventEnvelope
from deepkeel.scope import RuntimeScope
from deepkeel.state_store import (
    RunStateSnapshot,
    RuntimeStateMutation,
    RuntimeStateReceipt,
    RuntimeStateStore,
)
from deepkeel.telemetry import TracePage, TraceQuery, TraceStore


T = TypeVar("T")


class AsyncRuntimeStateStore(Protocol):
    terminal_settlement_owner: str

    async def commit_scoped(
        self,
        mutation: RuntimeStateMutation,
        *,
        scope: RuntimeScope,
        session: Any = None,
    ) -> RuntimeStateReceipt: ...

    async def load_snapshot_scoped(
        self,
        run_id: str,
        *,
        scope: RuntimeScope,
        session: Any = None,
    ) -> RunStateSnapshot: ...

    async def list_snapshots_scoped(
        self,
        *,
        scope: RuntimeScope,
        session: Any = None,
        statuses: Iterable[str] = (),
        limit: int = 100,
    ) -> tuple[RunStateSnapshot, ...]: ...


class AsyncDurableCheckpointStore(Protocol):
    async def load(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> dict[str, Any] | None: ...

    async def save(
        self,
        run_id: str,
        state: dict[str, Any],
        *,
        session: Any = None,
        user_id: str = "",
    ) -> None: ...

    async def delete(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> None: ...


class AsyncRuntimeEventJournal(Protocol):
    async def append(self, event: RuntimeEventEnvelope) -> RuntimeEventEnvelope: ...

    async def latest_sequence(self, run_id: str) -> int: ...

    async def read_after(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[RuntimeEventEnvelope, ...]: ...


class AsyncTraceStore(Protocol):
    async def query(self, query: TraceQuery) -> TracePage: ...


class AsyncRunLeaseStore(Protocol):
    async def claim(
        self,
        run_id: str,
        *,
        owner_id: str,
        ttl_seconds: float,
    ) -> RunLease: ...

    async def renew(self, lease: RunLease, *, ttl_seconds: float) -> RunLease: ...

    async def release(self, lease: RunLease) -> None: ...

    async def inspect(self, run_id: str) -> RunLease | None: ...


async def run_sync_adapter(
    operation: Callable[..., T],
    /,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Explicitly offload a blocking adapter operation from the event loop."""

    return await asyncio.to_thread(operation, *args, **kwargs)


class AsyncRuntimeStateStoreAdapter:
    """Opt-in bridge for thread-safe synchronous state adapters."""

    def __init__(self, store: RuntimeStateStore) -> None:
        self.store = store
        self.terminal_settlement_owner = store.terminal_settlement_owner

    async def commit_scoped(
        self,
        mutation: RuntimeStateMutation,
        *,
        scope: RuntimeScope,
        session: Any = None,
    ) -> RuntimeStateReceipt:
        scoped = getattr(self.store, "commit_scoped", None)
        if not callable(scoped):
            raise TypeError("state adapter does not implement commit_scoped")
        return await run_sync_adapter(scoped, mutation, scope=scope, session=session)

    async def load_snapshot_scoped(
        self,
        run_id: str,
        *,
        scope: RuntimeScope,
        session: Any = None,
    ) -> RunStateSnapshot:
        scoped = getattr(self.store, "load_snapshot_scoped", None)
        if not callable(scoped):
            raise TypeError("state adapter does not implement load_snapshot_scoped")
        return await run_sync_adapter(scoped, run_id, scope=scope, session=session)

    async def list_snapshots_scoped(
        self,
        *,
        scope: RuntimeScope,
        session: Any = None,
        statuses: Iterable[str] = (),
        limit: int = 100,
    ) -> tuple[RunStateSnapshot, ...]:
        scoped = getattr(self.store, "list_snapshots_scoped", None)
        if not callable(scoped):
            raise TypeError("state adapter does not implement list_snapshots_scoped")
        return await run_sync_adapter(
            scoped,
            scope=scope,
            session=session,
            statuses=statuses,
            limit=limit,
        )


class AsyncDurableCheckpointStoreAdapter:
    def __init__(self, store: DurableCheckpointStore) -> None:
        self.store = store

    async def load(self, run_id: str, **kwargs: Any) -> dict[str, Any] | None:
        return await run_sync_adapter(self.store.load, run_id, **kwargs)

    async def save(self, run_id: str, state: dict[str, Any], **kwargs: Any) -> None:
        await run_sync_adapter(self.store.save, run_id, state, **kwargs)

    async def delete(self, run_id: str, **kwargs: Any) -> None:
        await run_sync_adapter(self.store.delete, run_id, **kwargs)


class AsyncRuntimeEventJournalAdapter:
    def __init__(self, journal: RuntimeEventJournal) -> None:
        self.journal = journal

    async def append(self, event: RuntimeEventEnvelope) -> RuntimeEventEnvelope:
        return await run_sync_adapter(self.journal.append, event)

    async def latest_sequence(self, run_id: str) -> int:
        return await run_sync_adapter(self.journal.latest_sequence, run_id)

    async def read_after(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[RuntimeEventEnvelope, ...]:
        return await run_sync_adapter(
            self.journal.read_after,
            run_id,
            after_sequence=after_sequence,
            limit=limit,
        )


class AsyncTraceStoreAdapter:
    def __init__(self, store: TraceStore) -> None:
        self.store = store

    async def query(self, query: TraceQuery) -> TracePage:
        return await run_sync_adapter(self.store.query, query)


class AsyncRunLeaseStoreAdapter:
    """Opt-in bridge for a thread-safe synchronous lease adapter."""

    def __init__(self, store: RunLeaseStore) -> None:
        self.store = store

    async def claim(
        self,
        run_id: str,
        *,
        owner_id: str,
        ttl_seconds: float,
    ) -> RunLease:
        return await run_sync_adapter(
            self.store.claim,
            run_id,
            owner_id=owner_id,
            ttl_seconds=ttl_seconds,
        )

    async def renew(self, lease: RunLease, *, ttl_seconds: float) -> RunLease:
        return await run_sync_adapter(
            self.store.renew,
            lease,
            ttl_seconds=ttl_seconds,
        )

    async def release(self, lease: RunLease) -> None:
        await run_sync_adapter(self.store.release, lease)

    async def inspect(self, run_id: str) -> RunLease | None:
        return await run_sync_adapter(self.store.inspect, run_id)
