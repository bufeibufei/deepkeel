from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Event, Lock, Thread
from typing import Callable, Protocol
from uuid import uuid4


class RunLeaseConflict(RuntimeError):
    """Raised when another worker still owns a live run lease."""


class RunLeaseLost(RuntimeError):
    """Raised when a worker can no longer renew its execution lease."""


@dataclass(frozen=True, slots=True)
class RunLease:
    run_id: str
    owner_id: str
    token: str
    acquired_at: datetime
    expires_at: datetime
    generation: int = 1

    @property
    def expired(self) -> bool:
        return self.expires_at <= datetime.now(UTC)


class RunLeaseStore(Protocol):
    def claim(
        self,
        run_id: str,
        *,
        owner_id: str,
        ttl_seconds: float,
    ) -> RunLease: ...

    def renew(self, lease: RunLease, *, ttl_seconds: float) -> RunLease: ...

    def release(self, lease: RunLease) -> None: ...

    def inspect(self, run_id: str) -> RunLease | None: ...


class InMemoryRunLeaseStore:
    """Reference lease adapter with takeover and fencing-token semantics."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._leases: dict[str, RunLease] = {}
        self._generations: dict[str, int] = {}
        self._lock = Lock()

    def claim(
        self,
        run_id: str,
        *,
        owner_id: str,
        ttl_seconds: float,
    ) -> RunLease:
        normalized_run_id = str(run_id or "").strip()
        normalized_owner = str(owner_id or "").strip()
        if not normalized_run_id:
            raise ValueError("run_id is required")
        if not normalized_owner:
            raise ValueError("owner_id is required")
        ttl = _validated_ttl(ttl_seconds)
        with self._lock:
            now = self._clock()
            current = self._leases.get(normalized_run_id)
            if current is not None and current.expires_at > now:
                raise RunLeaseConflict(
                    f"run {normalized_run_id} is owned by {current.owner_id}"
                )
            generation = self._generations.get(normalized_run_id, 0) + 1
            lease = RunLease(
                run_id=normalized_run_id,
                owner_id=normalized_owner,
                token=uuid4().hex,
                acquired_at=now,
                expires_at=now + timedelta(seconds=ttl),
                generation=generation,
            )
            self._leases[normalized_run_id] = lease
            self._generations[normalized_run_id] = generation
            return lease

    def renew(self, lease: RunLease, *, ttl_seconds: float) -> RunLease:
        ttl = _validated_ttl(ttl_seconds)
        with self._lock:
            now = self._clock()
            current = self._leases.get(lease.run_id)
            if (
                current is None
                or current.token != lease.token
                or current.generation != lease.generation
                or current.expires_at <= now
            ):
                raise RunLeaseLost(f"run lease was lost for {lease.run_id}")
            renewed = RunLease(
                run_id=current.run_id,
                owner_id=current.owner_id,
                token=current.token,
                acquired_at=current.acquired_at,
                expires_at=now + timedelta(seconds=ttl),
                generation=current.generation,
            )
            self._leases[lease.run_id] = renewed
            return renewed

    def release(self, lease: RunLease) -> None:
        with self._lock:
            current = self._leases.get(lease.run_id)
            if current is None:
                return
            if current.token != lease.token or current.generation != lease.generation:
                raise RunLeaseLost(f"run lease was replaced for {lease.run_id}")
            self._leases.pop(lease.run_id, None)

    def inspect(self, run_id: str) -> RunLease | None:
        with self._lock:
            current = self._leases.get(str(run_id or ""))
            if current is None:
                return None
            if current.expires_at <= self._clock():
                return None
            return current


class RunLeaseGuard:
    """Claims a lease and renews it while one runtime turn is executing."""

    def __init__(
        self,
        store: RunLeaseStore,
        *,
        run_id: str,
        owner_id: str,
        ttl_seconds: float,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.owner_id = owner_id
        self.ttl_seconds = _validated_ttl(ttl_seconds)
        self.lease: RunLease | None = None
        self._stop = Event()
        self._thread: Thread | None = None
        self._lost: BaseException | None = None
        self._lock = Lock()

    def __enter__(self) -> "RunLeaseGuard":
        self.lease = self.store.claim(
            self.run_id,
            owner_id=self.owner_id,
            ttl_seconds=self.ttl_seconds,
        )
        self._thread = Thread(
            target=self._heartbeat,
            name=f"run-lease-{self.run_id[:16]}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.ttl_seconds))
        lease = self.lease
        if lease is not None:
            try:
                self.store.release(lease)
            except RunLeaseLost:
                pass

    def raise_if_lost(self) -> None:
        with self._lock:
            lost = self._lost
        if lost is not None:
            raise RunLeaseLost(f"run lease was lost for {self.run_id}") from lost

    def _heartbeat(self) -> None:
        interval = max(0.1, self.ttl_seconds / 3)
        while not self._stop.wait(interval):
            try:
                lease = self.lease
                if lease is None:
                    return
                self.lease = self.store.renew(lease, ttl_seconds=self.ttl_seconds)
            except BaseException as exc:
                with self._lock:
                    self._lost = exc
                return


def _validated_ttl(ttl_seconds: float) -> float:
    ttl = float(ttl_seconds)
    if ttl <= 0:
        raise ValueError("ttl_seconds must be positive")
    return ttl
