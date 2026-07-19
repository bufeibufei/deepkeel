from __future__ import annotations

from typing import Protocol

from harness_core.failures import RunCanceledError


class RunControl(Protocol):
    """Product-neutral cooperative control port for an active Agent run."""

    def raise_if_cancelled(self, run_id: str, *, force: bool = False) -> None: ...

    def release(self, run_id: str) -> None: ...


class NoopRunControl:
    def raise_if_cancelled(self, run_id: str, *, force: bool = False) -> None:
        del run_id, force

    def release(self, run_id: str) -> None:
        del run_id


class InMemoryRunControl:
    """Small deterministic implementation for embedding and tests."""

    def __init__(self) -> None:
        self._canceled: set[str] = set()

    def cancel(self, run_id: str) -> None:
        self._canceled.add(str(run_id))

    def raise_if_cancelled(self, run_id: str, *, force: bool = False) -> None:
        del force
        if str(run_id) in self._canceled:
            raise RunCanceledError()

    def release(self, run_id: str) -> None:
        self._canceled.discard(str(run_id))
