from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event

import pytest

from harness_core.adapter_sdk import (
    HarnessRuntimeBuilder,
    InMemoryRunLeaseStore,
    RunLeaseConflict,
    RunLeaseLost,
    RuntimePorts,
    verify_run_lease_store_contract,
)
from harness_core.runtime_sdk import RuntimeRequest


class BlockingProvider:
    model = "lease-model"
    model_role = "reasoning"

    def __init__(self, entered: Event, release: Event) -> None:
        self.entered = entered
        self.release = release

    def complete_chat(self, _messages, **_kwargs):
        self.entered.set()
        assert self.release.wait(5)
        return {
            "message": {"role": "assistant", "content": "completed"},
            "finish_reason": "stop",
            "model": self.model,
        }


def test_reference_run_lease_store_passes_contract() -> None:
    verify_run_lease_store_contract(InMemoryRunLeaseStore(), run_id="lease-contract")


def test_expired_lease_can_be_taken_over_with_a_new_fencing_generation() -> None:
    now = datetime(2026, 7, 20, tzinfo=UTC)
    clock = lambda: now
    store = InMemoryRunLeaseStore(clock=clock)
    first = store.claim("run-1", owner_id="worker-a", ttl_seconds=10)
    now += timedelta(seconds=11)
    second = store.claim("run-1", owner_id="worker-b", ttl_seconds=10)

    assert second.generation == first.generation + 1
    assert second.token != first.token
    with pytest.raises(RunLeaseLost):
        store.release(first)


def test_runtime_lease_prevents_two_workers_from_executing_one_run() -> None:
    entered = Event()
    release = Event()
    store = InMemoryRunLeaseStore()
    runtime = (
        HarnessRuntimeBuilder()
        .with_ports(
            RuntimePorts(
                run_lease_store=store,
                run_lease_owner_id="worker-a",
                run_lease_ttl_seconds=3,
            )
        )
        .build()
    )
    request = RuntimeRequest(question="hello", run_id="shared-run")
    provider = BlockingProvider(entered, release)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(runtime.run, request, provider=provider)
        assert entered.wait(2)
        with pytest.raises(RunLeaseConflict):
            runtime.run(request, provider=provider)
        release.set()
        result = first.result(timeout=5)

    assert result.status.value == "completed"
    assert store.inspect("shared-run") is None
