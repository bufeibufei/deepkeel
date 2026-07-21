from __future__ import annotations

from harness_core.runtime import HarnessRuntime
from harness_core.state_store import RunStateSnapshot
from harness_core.tool_registry import ToolRegistry
from harness_core.tools import ToolExecutor


class RuntimeStateStore:
    terminal_settlement_owner = "runtime"

    def load_snapshot(self, run_id, *, session=None, user_id=""):
        del session, user_id
        return RunStateSnapshot(
            run_id=run_id,
            version=2,
            sequence=3,
            status="waiting_user",
            checkpoint_type="runtime",
            checkpoint_state={"source": "runtime-state"},
        )

    def commit(self, mutation, *, session=None, user_id=""):  # pragma: no cover
        raise AssertionError("not used")


class LegacyCheckpointStore:
    def load(self, run_id, *, session=None, user_id=""):
        del run_id, session, user_id
        return {"source": "legacy"}


def test_runtime_state_store_is_the_authoritative_portable_checkpoint() -> None:
    registry = ToolRegistry()
    runtime = HarnessRuntime(
        registry,
        ToolExecutor(registry),
        runtime_state_store=RuntimeStateStore(),
        checkpoint_store=LegacyCheckpointStore(),
    )

    state, authority, errors = runtime._load_authoritative_checkpoint(
        "run-1",
        session=None,
        user_id="user-1",
    )

    assert state == {"source": "runtime-state"}
    assert authority == "runtime_state_store"
    assert errors == []
