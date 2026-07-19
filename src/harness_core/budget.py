from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol


MODEL_CALLS = "model_calls"
TOOL_CALLS = "tool_calls"
INPUT_TOKENS = "input_tokens"
OUTPUT_TOKENS = "output_tokens"
MODEL_RETRIES = "model_retries"
ELAPSED_SECONDS = "elapsed_seconds"
TOOL_CONCURRENCY = "tool_concurrency"


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """Typed limits with optional per-model-role overrides."""

    max_model_calls: int = 0
    max_tool_calls: int = 0
    max_input_tokens_total: int = 0
    max_input_tokens_per_call: int = 0
    max_output_tokens_total: int = 0
    max_output_tokens_per_call: int = 0
    max_model_retries: int = 0
    max_parallel_tools: int = 4
    max_elapsed_seconds: float = 900.0
    max_total_elapsed_seconds: float = 0.0
    max_request_seconds: float = 0.0
    roles: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "BudgetPolicy":
        raw = value if isinstance(value, dict) else {}
        roles = raw.get("roles") if isinstance(raw.get("roles"), dict) else {}
        return cls(
            max_model_calls=_nonnegative_int(raw.get("max_model_calls")),
            max_tool_calls=_nonnegative_int(raw.get("max_tool_calls")),
            max_input_tokens_total=_nonnegative_int(raw.get("max_input_tokens_total")),
            max_input_tokens_per_call=_nonnegative_int(raw.get("max_input_tokens_per_call")),
            max_output_tokens_total=_nonnegative_int(raw.get("max_output_tokens_total")),
            max_output_tokens_per_call=_nonnegative_int(raw.get("max_output_tokens_per_call")),
            max_model_retries=_nonnegative_int(raw.get("max_model_retries")),
            max_parallel_tools=max(1, _nonnegative_int(raw.get("max_parallel_tools"), 4)),
            max_elapsed_seconds=_nonnegative_float(raw.get("max_elapsed_seconds"), 900.0),
            max_total_elapsed_seconds=_nonnegative_float(raw.get("max_total_elapsed_seconds")),
            max_request_seconds=_nonnegative_float(raw.get("max_request_seconds")),
            roles={
                str(role): {
                    str(key): float(item)
                    for key, item in limits.items()
                    if _is_nonnegative_number(item)
                }
                for role, limits in roles.items()
                if isinstance(limits, dict)
            },
        )

    def limit(self, name: str, *, role: str = "") -> float | None:
        role_value = (self.roles.get(role) or {}).get(name)
        value = role_value if role_value is not None else getattr(self, name, 0)
        numeric = float(value or 0)
        return numeric if numeric > 0 else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_model_calls": self.max_model_calls,
            "max_tool_calls": self.max_tool_calls,
            "max_input_tokens_total": self.max_input_tokens_total,
            "max_input_tokens_per_call": self.max_input_tokens_per_call,
            "max_output_tokens_total": self.max_output_tokens_total,
            "max_output_tokens_per_call": self.max_output_tokens_per_call,
            "max_model_retries": self.max_model_retries,
            "max_parallel_tools": self.max_parallel_tools,
            "max_elapsed_seconds": self.max_elapsed_seconds,
            "max_total_elapsed_seconds": self.max_total_elapsed_seconds,
            "max_request_seconds": self.max_request_seconds,
            "roles": {key: dict(item) for key, item in self.roles.items()},
        }


@dataclass(frozen=True, slots=True)
class UsageReport:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    source: str = "estimated"

    @classmethod
    def from_provider(
        cls,
        value: dict[str, Any] | None,
        *,
        estimated_input: int = 0,
        estimated_output: int = 0,
    ) -> "UsageReport":
        raw = value if isinstance(value, dict) else {}
        input_tokens = _first_nonnegative_int(raw, "input_tokens", "prompt_tokens")
        output_tokens = _first_nonnegative_int(raw, "output_tokens", "completion_tokens")
        actual = input_tokens is not None or output_tokens is not None
        resolved_input = input_tokens if input_tokens is not None else max(0, int(estimated_input))
        resolved_output = output_tokens if output_tokens is not None else max(0, int(estimated_output))
        total = _first_nonnegative_int(raw, "total_tokens")
        return cls(
            input_tokens=resolved_input,
            output_tokens=resolved_output,
            total_tokens=total if total is not None else resolved_input + resolved_output,
            source="provider" if actual else "estimated",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class BudgetRequest:
    run_id: str
    metric: str
    amount: float = 1
    limit: float | None = None
    operation_id: str = ""
    aggregation: str = "sum"
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
            projected = max(used, amount) if request.aggregation == "max" else used + amount
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


def preview_budget(snapshot: BudgetSnapshot, request: BudgetRequest) -> BudgetDecision:
    """Non-mutating preflight used before an external side effect."""

    amount = max(0.0, float(request.amount))
    limit = None if request.limit is None or float(request.limit) <= 0 else float(request.limit)
    used = float(snapshot.usage.get(request.metric) or 0.0)
    projected = max(used, amount) if request.aggregation == "max" else used + amount
    allowed = limit is None or projected <= limit
    return BudgetDecision(
        allowed=allowed,
        metric=request.metric,
        requested=amount,
        used=projected if allowed else used,
        remaining=None if limit is None else max(0.0, limit - (projected if allowed else used)),
        limit=limit,
        reason="budget available" if allowed else f"{request.metric} budget exceeded",
    )


def _nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _nonnegative_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return max(0.0, float(default))


def _is_nonnegative_number(value: Any) -> bool:
    try:
        return float(value) >= 0
    except (TypeError, ValueError):
        return False


def _first_nonnegative_int(value: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in value:
            continue
        try:
            return max(0, int(value[key]))
        except (TypeError, ValueError):
            continue
    return None
