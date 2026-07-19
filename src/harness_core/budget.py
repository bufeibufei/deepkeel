from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol


MODEL_CALLS = "model_calls"
TOOL_CALLS = "tool_calls"


@dataclass(frozen=True, slots=True)
class BudgetRequest:
    run_id: str
    metric: str
    amount: float = 1
    limit: float | None = None
    operation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    allowed: bool
    metric: str
    requested: float
    used: float
    remaining: float | None
    limit: float | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "metric": self.metric,
            "requested": self.requested,
            "used": self.used,
            "remaining": self.remaining,
            "limit": self.limit,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    run_id: str
    usage: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "harness-budget-v1",
            "run_id": self.run_id,
            "usage": dict(self.usage),
        }


class BudgetLedger(Protocol):
    """Usage accounting port shared by model and tool execution."""

    def consume(self, request: BudgetRequest) -> BudgetDecision: ...

    def snapshot(self, run_id: str) -> BudgetSnapshot: ...

    def restore(self, run_id: str, snapshot: dict[str, Any] | BudgetSnapshot | None) -> None: ...

    def clear(self, run_id: str) -> None: ...


class InMemoryBudgetLedger:
    """Thread-safe default ledger; durable adapters can implement the same port."""

    def __init__(self) -> None:
        self._usage: dict[str, dict[str, float]] = {}
        self._decisions: dict[tuple[str, str, str], BudgetDecision] = {}
        self._lock = Lock()

    def consume(self, request: BudgetRequest) -> BudgetDecision:
        amount = max(0.0, float(request.amount))
        limit = None if request.limit is None or float(request.limit) <= 0 else float(request.limit)
        with self._lock:
            decision_key = (request.run_id, request.metric, request.operation_id)
            if request.operation_id and decision_key in self._decisions:
                return self._decisions[decision_key]
            run_usage = self._usage.setdefault(request.run_id, {})
            used = float(run_usage.get(request.metric) or 0.0)
            projected = used + amount
            if limit is not None and projected > limit:
                decision = BudgetDecision(
                    allowed=False,
                    metric=request.metric,
                    requested=amount,
                    used=used,
                    remaining=max(0.0, limit - used),
                    limit=limit,
                    reason=f"{request.metric} budget exceeded",
                )
                if request.operation_id:
                    self._decisions[decision_key] = decision
                return decision
            run_usage[request.metric] = projected
            decision = BudgetDecision(
                allowed=True,
                metric=request.metric,
                requested=amount,
                used=projected,
                remaining=None if limit is None else max(0.0, limit - projected),
                limit=limit,
                reason="budget reserved",
            )
            if request.operation_id:
                self._decisions[decision_key] = decision
            return decision

    def snapshot(self, run_id: str) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(run_id=run_id, usage=dict(self._usage.get(run_id) or {}))

    def restore(self, run_id: str, snapshot: dict[str, Any] | BudgetSnapshot | None) -> None:
        if isinstance(snapshot, BudgetSnapshot):
            usage = snapshot.usage
        elif isinstance(snapshot, dict):
            usage = snapshot.get("usage") if isinstance(snapshot.get("usage"), dict) else {}
        else:
            usage = {}
        with self._lock:
            current = self._usage.setdefault(run_id, {})
            for metric, value in usage.items():
                try:
                    current[str(metric)] = max(float(current.get(str(metric)) or 0.0), float(value))
                except (TypeError, ValueError):
                    continue

    def clear(self, run_id: str) -> None:
        with self._lock:
            self._usage.pop(run_id, None)
            self._decisions = {
                key: value for key, value in self._decisions.items() if key[0] != run_id
            }


class BudgetExceededError(RuntimeError):
    code = "BUDGET_EXCEEDED"

    def __init__(self, decision: BudgetDecision):
        super().__init__(decision.reason)
        self.decision = decision
