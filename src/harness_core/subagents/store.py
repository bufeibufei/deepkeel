from __future__ import annotations

from typing import Any, Protocol

from harness_core.subagents.contracts import DelegationTask, SubAgentResult, SubAgentSpec


class SubAgentRunStore(Protocol):
    """Persistence port for child-run lineage owned outside the runtime core."""

    def parent_accepts_results(self, parent_run_id: str) -> bool: ...

    def create_child(
        self,
        *,
        child_run_id: str,
        root_run_id: str,
        parent_run_id: str,
        delegation_id: str,
        task: DelegationTask,
        spec: SubAgentSpec,
        user_id: str,
        thread_id: str,
    ) -> None: ...

    def settle_child(self, result: SubAgentResult) -> None: ...

    def load_child_result(self, child_run_id: str) -> SubAgentResult | None: ...

    def load_child_checkpoint(self, child_run_id: str) -> dict[str, Any] | None: ...

    def checkpoint_child(
        self,
        child_run_id: str,
        *,
        phase: str,
        state: dict[str, Any],
    ) -> None: ...

    def cancel_requested(self, child_run_id: str, parent_run_id: str) -> bool: ...
