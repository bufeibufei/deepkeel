from __future__ import annotations

import asyncio
from copy import deepcopy
import inspect
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any, Awaitable, Callable, Literal, Mapping, Protocol
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from harness_core.budget import (
    TOOL_CONCURRENCY,
    TOOL_CALLS,
    BudgetLedger,
    BudgetRequest,
    InMemoryBudgetLedger,
)
from harness_core.clarifications import clarification_for_missing_arguments, clarification_tool_result
from harness_core.contracts import Observation, PendingAction, ToolCall, ToolResult
from harness_core.control import NoopRunControl, RunControl
from harness_core.deadlines import ensure_time_remaining
from harness_core.failures import RunCanceledError, RunDeadlineExceededError
from harness_core.hooks import (
    HookAction,
    HookAudit,
    HookInvocation,
    HookPoint,
    HookRunner,
)
from harness_core.leases import ExecutionFence
from harness_core.policy import (
    DefaultPolicyEngine,
    PolicyDecision,
    PolicyEngine,
    PolicyRequest,
)
from harness_core.ports import RuntimeSession, SessionFactory
from harness_core.tool_registry import ToolRegistry, ToolSpec
from harness_core.type_narrowing import as_dict


ToolHandler = Callable[
    [ToolCall, "ToolExecutionContext"],
    ToolResult | Awaitable[ToolResult],
]
ToolPreflight = Callable[[ToolCall, "ToolExecutionContext", ToolSpec], str | None]


@dataclass(slots=True)
class ToolExecutionContext:
    run_id: str
    user_id: str
    thread_id: str = ""
    turn_id: str = ""
    session: RuntimeSession | None = None
    session_factory: SessionFactory | None = None
    context_bundle: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    budget_limits: dict[str, float] = field(default_factory=dict)
    deadline_monotonic: float | None = None
    run_control: RunControl = field(default_factory=NoopRunControl)
    execution_fence: ExecutionFence | None = None

    @property
    def fence_token(self) -> str:
        return self.execution_fence.token if self.execution_fence is not None else ""

    @property
    def fence_generation(self) -> int:
        return self.execution_fence.generation if self.execution_fence is not None else 0

    def raise_if_fence_lost(self) -> None:
        if self.execution_fence is not None:
            self.execution_fence.raise_if_lost()

    def fork(self, *, session: RuntimeSession | None = None) -> "ToolExecutionContext":
        return ToolExecutionContext(
            run_id=self.run_id,
            user_id=self.user_id,
            thread_id=self.thread_id,
            turn_id=self.turn_id,
            session=session,
            session_factory=self.session_factory,
            context_bundle=_copy_context_mapping(self.context_bundle),
            metadata=_copy_context_mapping(self.metadata),
            budget_limits=dict(self.budget_limits),
            deadline_monotonic=self.deadline_monotonic,
            run_control=self.run_control,
            execution_fence=self.execution_fence,
        )


@dataclass(frozen=True, slots=True)
class ToolExecutionClaim:
    status: Literal[
        "claimed",
        "replay",
        "busy",
        "blocked",
        "corrupt",
        "uncertain",
        "exhausted",
    ]
    run_id: str
    idempotency_key: str
    record_id: str = ""
    claim_owner: str = ""
    attempt_count: int = 0
    lease_expires_at: datetime | None = None
    retry_after_seconds: float = 0.0
    terminal_status: str = ""
    detail: str = ""


class ToolExecutionStore(Protocol):
    """Durable claim/replay/settle port for idempotent tool execution."""

    def claim(
        self,
        *,
        run_id: str,
        call: ToolCall,
        lease_seconds: float,
        max_attempts: int,
        reexecution_safe: bool = True,
    ) -> ToolExecutionClaim: ...

    def replay(self, claim: ToolExecutionClaim) -> ToolResult: ...

    def settle(self, claim: ToolExecutionClaim, result: ToolResult) -> None: ...


@dataclass(slots=True)
class _MemoryToolExecution:
    record_id: str
    status: str
    claim_owner: str
    lease_expires_at: datetime | None
    attempt_count: int
    result: ToolResult | None = None


class InMemoryToolExecutionStore:
    """Process-local fallback for Harness tests without persistence."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], _MemoryToolExecution] = {}
        self._lock = Lock()

    def claim(
        self,
        *,
        run_id: str,
        call: ToolCall,
        lease_seconds: float,
        max_attempts: int,
        reexecution_safe: bool = True,
    ) -> ToolExecutionClaim:
        now = datetime.now(UTC)
        key = (run_id, call.idempotency_key)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.status == "uncertain":
                return _memory_claim("uncertain", run_id, call.idempotency_key, entry)
            if entry is not None and entry.status == "exhausted":
                return _memory_claim("exhausted", run_id, call.idempotency_key, entry)
            if entry is not None and entry.result is not None:
                retry_failed = (
                    entry.result.status == "failed"
                    and entry.result.retryable
                    and entry.attempt_count < max(1, int(max_attempts))
                )
                if not retry_failed:
                    return _memory_claim("replay", run_id, call.idempotency_key, entry)
            if (
                entry is not None
                and entry.result is None
                and entry.claim_owner
                and entry.lease_expires_at is not None
                and entry.lease_expires_at > now
            ):
                return _memory_claim("busy", run_id, call.idempotency_key, entry)
            if entry is not None and entry.result is None:
                if entry.attempt_count >= max(1, int(max_attempts)):
                    entry.status = "exhausted"
                    entry.claim_owner = ""
                    entry.lease_expires_at = None
                    return _memory_claim("exhausted", run_id, call.idempotency_key, entry)
                if not reexecution_safe:
                    entry.status = "uncertain"
                    entry.claim_owner = ""
                    entry.lease_expires_at = None
                    return _memory_claim("uncertain", run_id, call.idempotency_key, entry)
            owner = uuid4().hex
            lease_expires_at = now + timedelta(seconds=max(0.001, float(lease_seconds)))
            if entry is None:
                entry = _MemoryToolExecution(
                    record_id=uuid4().hex,
                    status="running",
                    claim_owner=owner,
                    lease_expires_at=lease_expires_at,
                    attempt_count=1,
                )
                self._entries[key] = entry
            else:
                entry.status = "running"
                entry.claim_owner = owner
                entry.lease_expires_at = lease_expires_at
                entry.attempt_count += 1
                entry.result = None
            return _memory_claim("claimed", run_id, call.idempotency_key, entry)

    def replay(self, claim: ToolExecutionClaim) -> ToolResult:
        with self._lock:
            entry = self._entries.get((claim.run_id, claim.idempotency_key))
            if entry is None or entry.result is None:
                raise LookupError("tool execution result is not available for replay")
            return entry.result.model_copy(deep=True)

    def settle(self, claim: ToolExecutionClaim, result: ToolResult) -> None:
        with self._lock:
            entry = self._entries.get((claim.run_id, claim.idempotency_key))
            if entry is None or entry.claim_owner != claim.claim_owner:
                raise RuntimeError("tool execution claim is no longer owned")
            entry.status = result.status
            entry.claim_owner = ""
            entry.lease_expires_at = None
            entry.result = result.model_copy(deep=True)


def _memory_claim(
    status: Literal["claimed", "replay", "busy", "uncertain", "exhausted"],
    run_id: str,
    idempotency_key: str,
    entry: _MemoryToolExecution,
) -> ToolExecutionClaim:
    retry_after = 0.0
    if status == "busy" and entry.lease_expires_at is not None:
        retry_after = max(0.0, (entry.lease_expires_at - datetime.now(UTC)).total_seconds())
    return ToolExecutionClaim(
        status=status,
        run_id=run_id,
        idempotency_key=idempotency_key,
        record_id=entry.record_id,
        claim_owner=entry.claim_owner,
        attempt_count=entry.attempt_count,
        lease_expires_at=entry.lease_expires_at,
        retry_after_seconds=retry_after,
    )


def _tool_hook_invocation(
    point: HookPoint,
    call: ToolCall,
    context: ToolExecutionContext,
    *,
    payload: Mapping[str, Any],
) -> HookInvocation:
    skill = as_dict(context.metadata.get("skill_activation"))
    package_ids = tuple(
        str(value)
        for value in context.metadata.get("capability_package_ids", ())
        if str(value).strip()
    )
    return HookInvocation(
        point=point,
        operation_id=(
            f"{context.run_id}:{context.turn_id}:tool:{call.id}:{point.value}"
        ),
        run_id=context.run_id,
        thread_id=context.thread_id,
        turn_id=context.turn_id,
        package_ids=package_ids,
        skill_id=str(skill.get("skill_id") or ""),
        payload={
            "tool_call_id": call.id,
            "tool_name": call.name,
            **dict(payload),
        },
        metadata={
            "tenant_id": str(context.metadata.get("tenant_id") or ""),
            "user_id": context.user_id,
        },
    )


def _emit_hook_audits(
    context: ToolExecutionContext,
    audits: tuple[HookAudit, ...],
) -> None:
    sink = context.metadata.get("event_sink")
    if not callable(sink):
        return
    for audit in audits:
        sink(
            {
                "event_type": "hook.executed",
                "title": "Lifecycle hook",
                "summary": f"{audit.point.value}: {audit.status}",
                "payload": {
                    "hook_id": audit.hook_id,
                    "hook_point": audit.point.value,
                    "operation_id": audit.operation_id,
                    "status": audit.status,
                    "duration_ms": audit.duration_ms,
                    "replayed": audit.replayed,
                    "required": audit.required,
                    "error": audit.error,
                    "diagnostics": dict(audit.diagnostics),
                },
                "visibility": "debug",
            }
        )


def _hook_confirmation_result(
    call: ToolCall,
    context: ToolExecutionContext,
    *,
    title: str,
    message: str,
) -> ToolResult:
    pending = PendingAction(
        id=f"hook-confirmation:{call.id}",
        run_id=context.run_id,
        tool_call_id=call.id,
        action_type="confirm_tool_invocation",
        title=title or "Confirm action",
        prompt=message or "Please confirm this action before it continues.",
        payload={
            "tool_name": call.name,
            "arguments": dict(call.arguments),
            "source": "lifecycle_hook",
        },
    )
    return ToolResult(
        call=call,
        status="requires_user_action",
        outcome="partial",
        summary=pending.prompt,
        pending_action=pending,
        metadata={"hook_confirmation": True, "executed": False},
    )


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        preflight: ToolPreflight | None = None,
        max_parallel_tools: int = 4,
        execution_store: ToolExecutionStore | None = None,
        policy_engine: PolicyEngine | None = None,
        budget_ledger: BudgetLedger | None = None,
        hook_runner: HookRunner | None = None,
        claim_lease_seconds: float = 300,
        max_idempotent_attempts: int = 2,
    ):
        self.registry = registry
        self.preflight = preflight
        self.max_parallel_tools = max(1, int(max_parallel_tools))
        self.execution_store = execution_store or InMemoryToolExecutionStore()
        self.policy_engine = policy_engine or DefaultPolicyEngine()
        self.budget_ledger = budget_ledger or InMemoryBudgetLedger()
        self.hook_runner = hook_runner
        self.claim_lease_seconds = max(0.001, float(claim_lease_seconds))
        self.max_idempotent_attempts = max(1, int(max_idempotent_attempts))
        self._handlers: dict[str, ToolHandler] = {}
        self._artifact_schemas: dict[str, dict[str, Any]] = {}
        self._artifact_contracts_configured = False

    def register(self, tool_name: str, handler: ToolHandler) -> None:
        self.registry.get(tool_name)
        self._handlers[tool_name] = handler

    def unregister(self, tool_name: str) -> None:
        self._handlers.pop(tool_name, None)

    def snapshot_handlers(self) -> dict[str, ToolHandler]:
        return dict(self._handlers)

    def restore_handlers(self, snapshot: dict[str, ToolHandler]) -> None:
        self._handlers = dict(snapshot)

    def configure_artifact_schemas(
        self,
        schemas: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self._artifact_contracts_configured = True
        self._artifact_schemas = {
            str(name): dict(schema)
            for name, schema in schemas.items()
            if str(name).strip()
        }

    @property
    def registered_tools(self) -> frozenset[str]:
        return frozenset(self._handlers)

    def execute(self, call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        """Compatibility entrypoint for synchronous hosts."""

        return asyncio.run(self.aexecute(call, context))

    async def aexecute(self, call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        active_call = call
        if self.hook_runner is not None:
            before = await self.hook_runner.arun(
                _tool_hook_invocation(
                    HookPoint.TOOL_BEFORE,
                    call,
                    context,
                    payload={"arguments": dict(call.arguments)},
                )
            )
            _emit_hook_audits(context, before.audits)
            if before.decision.tool_arguments is not None:
                active_call = call.model_copy(
                    update={"arguments": dict(before.decision.tool_arguments)}
                )
            if before.decision.action == HookAction.DENY:
                denied = _failed_result(
                    active_call,
                    before.decision.reason or "tool invocation denied by lifecycle hook",
                )
                denied.metadata["hook_denied"] = True
                return denied
            if before.decision.action == HookAction.WAIT_FOR_CONFIRMATION:
                return _hook_confirmation_result(
                    active_call,
                    context,
                    title=before.decision.confirmation_title,
                    message=before.decision.confirmation_message
                    or before.decision.reason,
                )

        result = await self._aexecute_core(active_call, context)
        if self.hook_runner is None:
            return result
        point = HookPoint.TOOL_FAILED if result.status == "failed" else HookPoint.TOOL_AFTER
        after = await self.hook_runner.arun(
            _tool_hook_invocation(
                point,
                active_call,
                context,
                payload={
                    "arguments": dict(active_call.arguments),
                    "status": result.status,
                    "outcome": result.outcome,
                    "summary": result.summary,
                    "error": result.error,
                    "artifact_ids": [artifact.id for artifact in result.artifacts],
                },
            )
        )
        _emit_hook_audits(context, after.audits)
        result.metadata.setdefault("hooks", {})
        result.metadata["hooks"].update(dict(after.decision.diagnostics))
        return result

    async def _aexecute_core(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        context.raise_if_fence_lost()
        context.run_control.raise_if_cancelled(context.run_id)
        ensure_time_remaining(context.deadline_monotonic)
        started_at = time.perf_counter()
        try:
            spec = self.registry.get(call.name)
        except KeyError:
            return _with_runtime_metrics(_failed_result(call, "unknown tool"), started_at, phase="lookup")

        try:
            normalized = call.model_copy(update={"arguments": normalize_arguments(call.arguments, spec)})
            normalized = _bind_authoritative_context_arguments(normalized, context, spec)
        except (TypeError, ValueError) as exc:
            return _with_runtime_metrics(
                _failed_result(call, str(exc)),
                started_at,
                phase="normalization",
            )
        clarification = clarification_for_missing_arguments(normalized, spec)
        if clarification is not None:
            return _with_runtime_metrics(
                clarification_tool_result(
                    normalized,
                    run_id=context.run_id,
                    request=clarification,
                    visible_label="Additional information required",
                ),
                started_at,
                phase="validation",
                executed=False,
            )
        schema_error = _argument_schema_error(normalized.arguments, spec)
        if schema_error:
            return _with_runtime_metrics(
                _invalid_arguments_result(normalized, context, spec, schema_error),
                started_at,
                phase="schema_validation",
                executed=False,
            )
        if self.preflight is not None:
            denial = self.preflight(normalized, context, spec)
            if denial:
                return _with_runtime_metrics(_failed_result(normalized, denial), started_at, phase="preflight")

        confirmation_grant = _confirmation_grant(context, normalized)
        policy = self.policy_engine.evaluate(
            PolicyRequest(
                action="tool.invoke",
                resource_type="tool",
                resource_id=normalized.name,
                run_id=context.run_id,
                user_id=context.user_id,
                tenant_id=str(context.metadata.get("tenant_id") or ""),
                risk_level="low" if spec.read_only else "write",
                context={
                    "skill_activation": context.metadata.get("skill_activation") or {},
                    "usage_policy": spec.usage_policy,
                    "runtime_policy": spec.runtime_policy,
                    "tool_call_id": normalized.id,
                    "confirmation_grant": confirmation_grant,
                    "governance_scope": context.metadata.get("governance_scope") or {},
                },
            )
        )
        if policy.requires_confirmation:
            return _with_runtime_metrics(
                _confirmation_required_result(normalized, context, spec, policy),
                started_at,
                phase="policy_confirmation",
                executed=False,
            )
        if not policy.allowed:
            denied = _failed_result(normalized, policy.reason)
            denied.metadata["governance"] = {"policy": policy.as_dict(), "budget": {}}
            return _with_runtime_metrics(denied, started_at, phase="policy")
        if (
            spec.runtime_policy.get("confirmation_required") is True
            and confirmation_grant.get("confirmed") is True
        ):
            normalized = normalized.model_copy(
                update={"arguments": {**normalized.arguments, "confirmed": True}}
            )

        handler = self._handlers.get(normalized.name)
        if handler is None:
            return _with_runtime_metrics(
                _failed_result(normalized, "tool handler is not registered"),
                started_at,
                phase="handler_lookup",
            )

        if not normalized.idempotency_key:
            result = await self._ainvoke(
                handler,
                normalized,
                context,
                spec=spec,
                policy=policy,
                started_at=started_at,
            )
            context.raise_if_fence_lost()
            return result

        lease_seconds = _positive_number(
            spec.runtime_policy.get("idempotency_lease_seconds"),
            self.claim_lease_seconds,
        )
        max_attempts = _positive_int(
            spec.runtime_policy.get("idempotency_max_attempts"),
            self.max_idempotent_attempts,
        )
        reexecution_safe = _tool_reexecution_safe(spec)
        try:
            claim = self.execution_store.claim(
                run_id=context.run_id,
                call=normalized,
                lease_seconds=lease_seconds,
                max_attempts=max_attempts,
                reexecution_safe=reexecution_safe,
            )
        except Exception as exc:
            return _with_runtime_metrics(
                _failed_result(normalized, f"tool execution claim failed: {exc}", retryable=True),
                started_at,
                phase="idempotent_claim",
                executed=False,
            )
        if claim.status == "replay":
            try:
                persisted = self.execution_store.replay(claim)
            except Exception as exc:
                failed = _failed_result(
                    normalized,
                    "Tool execution recovery failed. Retry the operation.",
                    retryable=True,
                )
                failed.metadata["replay_error"] = str(exc)
                return _with_runtime_metrics(
                    failed,
                    started_at,
                    phase="idempotent_replay",
                    executed=False,
                )
            return _with_runtime_metrics(
                _replayed_result(persisted, normalized),
                started_at,
                phase="idempotent_replay",
                executed=False,
            )
        if claim.status == "busy":
            busy = _failed_result(normalized, "tool execution is already in progress", retryable=True)
            busy.metadata = {
                "idempotency_busy": True,
                "retry_after_seconds": claim.retry_after_seconds,
                "claim_owner": claim.claim_owner,
            }
            return _with_runtime_metrics(
                busy,
                started_at,
                phase="idempotent_busy",
                executed=False,
            )
        if claim.status == "blocked":
            blocked = _failed_result(
                normalized,
                f"agent run is already {claim.terminal_status or 'terminal'}",
            )
            blocked.metadata = {
                "terminal_run_blocked": True,
                "terminal_status": claim.terminal_status,
            }
            return _with_runtime_metrics(
                blocked,
                started_at,
                phase="idempotent_blocked",
                executed=False,
            )
        if claim.status in {"corrupt", "uncertain", "exhausted"}:
            messages = {
                "corrupt": "The tool execution record is corrupt. The run ended safely; start again.",
                "uncertain": "The prior tool result is uncertain. The run ended safely to avoid duplication.",
                "exhausted": "Tool recovery attempts were exhausted and the run ended safely.",
            }
            failed = _failed_result(normalized, messages[claim.status])
            failed.metadata = {
                "durable_execution_status": claim.status,
                "durable_execution_detail": claim.detail,
                "tool_invocation_id": claim.record_id,
                "attempt_count": claim.attempt_count,
                "reexecution_safe": reexecution_safe,
            }
            return _with_runtime_metrics(
                failed,
                started_at,
                phase=f"idempotent_{claim.status}",
                executed=False,
            )
        result = await self._ainvoke(
            handler,
            normalized,
            context,
            spec=spec,
            policy=policy,
            started_at=started_at,
        )
        context.raise_if_fence_lost()
        settlement_error = None
        for settlement_attempt in range(3):
            try:
                self.execution_store.settle(claim, result)
                settlement_error = None
                break
            except Exception as exc:
                settlement_error = exc
                if settlement_attempt < 2:
                    await asyncio.sleep(0.05 * (settlement_attempt + 1))
        if settlement_error is not None:
            failed = _failed_result(
                normalized,
                f"tool execution settlement failed: {settlement_error}",
                retryable=reexecution_safe,
            )
            failed.metadata["settlement_failed"] = True
            failed.metadata["reexecution_safe"] = reexecution_safe
            failed.metadata["tool_invocation_id"] = claim.record_id
            return _with_runtime_metrics(failed, started_at, phase="idempotent_settle")
        return result

    def execute_many(self, calls: list[ToolCall], context: ToolExecutionContext) -> list[ToolResult]:
        if len(calls) < 2:
            return [self.execute(call, context) for call in calls]

        results: list[ToolResult | None] = [None] * len(calls)
        parallel: list[tuple[int, ToolCall]] = []
        serial: list[tuple[int, ToolCall]] = []
        for index, call in enumerate(calls):
            try:
                spec = self.registry.get(call.name)
            except KeyError:
                serial.append((index, call))
                continue
            can_isolate_session = context.session is None or context.session_factory is not None
            target = parallel if spec.read_only and spec.parallel_safe and can_isolate_session else serial
            target.append((index, call))

        if parallel:
            configured_limit = int(
                context.budget_limits.get(TOOL_CONCURRENCY)
                or self.max_parallel_tools
            )
            workers = max(1, min(self.max_parallel_tools, configured_limit, len(parallel)))
            concurrency_budget = self.budget_ledger.consume(
                BudgetRequest(
                    run_id=context.run_id,
                    metric=TOOL_CONCURRENCY,
                    amount=workers,
                    limit=configured_limit,
                    operation_id=(
                        f"tool-concurrency:{context.turn_id}:"
                        + ":".join(sorted(call.id for _, call in parallel))
                    ),
                    aggregation="max",
                    metadata={"parallel_call_count": len(parallel)},
                )
            )
            if not concurrency_budget.allowed:
                workers = 1
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent-tool") as pool:
                futures = {
                    pool.submit(self._execute_parallel, call, context): index
                    for index, call in parallel
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except (RunCanceledError, RunDeadlineExceededError):
                        for pending in futures:
                            pending.cancel()
                        raise
                    except Exception as exc:
                        failed = _failed_result(calls[index], str(exc), retryable=True)
                        failed.metadata["parallel_worker_failed"] = True
                        results[index] = failed
        suspension_seen = any(
            result is not None and result.status in {"requires_user_action", "waiting_async"}
            for result in results
        )
        for index, call in serial:
            if suspension_seen:
                rejected = _failed_result(
                    call,
                    "The previous action must finish before this tool can run.",
                )
                rejected.metadata["suspension_rejected"] = True
                rejected.metadata["executed"] = False
                results[index] = rejected
                continue
            result = self.execute(call, context)
            results[index] = result
            if result.status in {"requires_user_action", "waiting_async"}:
                suspension_seen = True
        return _keep_single_suspending_result(
            [result for result in results if result is not None]
        )

    async def aexecute_many(
        self,
        calls: list[ToolCall],
        context: ToolExecutionContext,
    ) -> list[ToolResult]:
        """Execute independent async-safe tools concurrently in the host loop."""

        if len(calls) < 2:
            return [await self.aexecute(call, context) for call in calls]

        results: list[ToolResult | None] = [None] * len(calls)
        parallel: list[tuple[int, ToolCall]] = []
        serial: list[tuple[int, ToolCall]] = []
        for index, call in enumerate(calls):
            try:
                spec = self.registry.get(call.name)
            except KeyError:
                serial.append((index, call))
                continue
            can_isolate_session = context.session is None or context.session_factory is not None
            target = parallel if spec.read_only and spec.parallel_safe and can_isolate_session else serial
            target.append((index, call))

        if parallel:
            configured_limit = int(
                context.budget_limits.get(TOOL_CONCURRENCY)
                or self.max_parallel_tools
            )
            workers = max(1, min(self.max_parallel_tools, configured_limit, len(parallel)))
            concurrency_budget = self.budget_ledger.consume(
                BudgetRequest(
                    run_id=context.run_id,
                    metric=TOOL_CONCURRENCY,
                    amount=workers,
                    limit=configured_limit,
                    operation_id=(
                        f"tool-concurrency:{context.turn_id}:"
                        + ":".join(sorted(call.id for _, call in parallel))
                    ),
                    aggregation="max",
                    metadata={"parallel_call_count": len(parallel)},
                )
            )
            if not concurrency_budget.allowed:
                workers = 1
            semaphore = asyncio.Semaphore(workers)

            async def execute_parallel(index: int, call: ToolCall) -> tuple[int, ToolResult]:
                async with semaphore:
                    try:
                        return index, await self._aexecute_parallel(call, context)
                    except (RunCanceledError, RunDeadlineExceededError):
                        raise
                    except Exception as exc:
                        failed = _failed_result(call, str(exc), retryable=True)
                        failed.metadata["parallel_worker_failed"] = True
                        return index, failed

            completed = await asyncio.gather(
                *(execute_parallel(index, call) for index, call in parallel)
            )
            for index, result in completed:
                results[index] = result

        suspension_seen = any(
            result is not None and result.status in {"requires_user_action", "waiting_async"}
            for result in results
        )
        for index, call in serial:
            if suspension_seen:
                rejected = _failed_result(
                    call,
                    "The previous action must finish before this tool can run.",
                )
                rejected.metadata["suspension_rejected"] = True
                rejected.metadata["executed"] = False
                results[index] = rejected
                continue
            result = await self.aexecute(call, context)
            results[index] = result
            if result.status in {"requires_user_action", "waiting_async"}:
                suspension_seen = True
        return _keep_single_suspending_result(
            [result for result in results if result is not None]
        )

    def _execute_parallel(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if context.session_factory is None:
            return self.execute(call, context.fork(session=None))
        isolated_session = context.session_factory()
        try:
            return self.execute(call, context.fork(session=isolated_session))
        finally:
            close = getattr(isolated_session, "close", None)
            if callable(close):
                close()

    async def _aexecute_parallel(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if context.session_factory is None:
            return await self.aexecute(call, context.fork(session=None))
        isolated_session = context.session_factory()
        try:
            return await self.aexecute(call, context.fork(session=isolated_session))
        finally:
            close = getattr(isolated_session, "aclose", None)
            if callable(close):
                closed = close()
                if inspect.isawaitable(closed):
                    await closed
            else:
                close = getattr(isolated_session, "close", None)
                if callable(close):
                    close()

    async def _ainvoke(
        self,
        handler: ToolHandler,
        call: ToolCall,
        context: ToolExecutionContext,
        *,
        spec: ToolSpec,
        policy: PolicyDecision,
        started_at: float | None = None,
    ) -> ToolResult:
        started_at = started_at or time.perf_counter()
        budget = self.budget_ledger.consume(
            BudgetRequest(
                run_id=context.run_id,
                metric=TOOL_CALLS,
                limit=context.budget_limits.get(TOOL_CALLS),
                operation_id=f"tool-call:{call.idempotency_key or call.id}",
                metadata={"tool_name": call.name},
            )
        )
        if not budget.allowed:
            result = _failed_result(call, budget.reason)
            result.metadata["governance"] = {
                "policy": policy.as_dict(),
                "budget": budget.as_dict(),
            }
            return _with_runtime_metrics(result, started_at, phase="budget", executed=False)
        try:
            raw = handler(call, context)
            if inspect.isawaitable(raw):
                raw = await raw
            context.raise_if_fence_lost()
            context.run_control.raise_if_cancelled(context.run_id, force=True)
            ensure_time_remaining(context.deadline_monotonic)
            if not isinstance(raw, ToolResult):
                raise TypeError("tool handlers must return ToolResult")
            result = raw
            result = _normalize_async_result(result, spec)
            artifact_error = self._artifact_contract_error(result)
            if artifact_error:
                result = _failed_result(call, artifact_error)
                result.metadata["artifact_contract_failed"] = True
        except (RunCanceledError, RunDeadlineExceededError):
            raise
        except Exception as exc:
            result = _failed_result(call, str(exc), retryable=True)
        result.metadata = {
            **(result.metadata if isinstance(result.metadata, dict) else {}),
            "governance": {
                "policy": policy.as_dict(),
                "budget": budget.as_dict(),
            },
        }
        return _with_runtime_metrics(result, started_at, phase="execution")

    def _artifact_contract_error(self, result: ToolResult) -> str:
        if not result.artifacts or not self._artifact_contracts_configured:
            return ""
        for artifact in result.artifacts:
            schema = self._artifact_schemas.get(artifact.artifact_type)
            if schema is None:
                return f"unregistered artifact type: {artifact.artifact_type}"
            errors = sorted(
                Draft202012Validator(schema).iter_errors(artifact.data),
                key=lambda item: list(item.absolute_path),
            )
            if errors:
                return (
                    f"artifact {artifact.artifact_type} does not match its contract: "
                    f"{errors[0].message}"
                )
        return ""


def _bind_authoritative_context_arguments(
    call: ToolCall,
    context: ToolExecutionContext,
    spec: ToolSpec,
) -> ToolCall:
    """Replace model-provided arguments with explicitly bound Host context."""

    bindings = spec.runtime_policy.get("context_argument_bindings")
    if not isinstance(bindings, dict) or not bindings:
        return call
    document = {
        **context.context_bundle,
        "run_id": context.run_id,
        "user_id": context.user_id,
        "thread_id": context.thread_id,
        "turn_id": context.turn_id,
    }
    arguments = dict(call.arguments)
    changed = False
    for argument_name, configured_paths in bindings.items():
        paths = configured_paths if isinstance(configured_paths, list) else [configured_paths]
        value = next(
            (
                resolved
                for path in paths
                if isinstance(path, str)
                and (resolved := _context_path_value(document, path)) not in (None, "")
            ),
            None,
        )
        name = str(argument_name).strip()
        if name and value is not None and arguments.get(name) != value:
            arguments[name] = deepcopy(value)
            changed = True
    return call.model_copy(update={"arguments": arguments}) if changed else call


def _context_path_value(document: Mapping[str, Any], path: str) -> Any:
    current: Any = document
    for segment in (part for part in path.split(".") if part):
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def _confirmation_grant(context: ToolExecutionContext, call: ToolCall) -> dict[str, Any]:
    grants = context.metadata.get("confirmation_grants")
    if not isinstance(grants, dict):
        return {}
    grant = grants.get(call.id)
    return grant if isinstance(grant, dict) else {}


def _confirmation_required_result(
    call: ToolCall,
    context: ToolExecutionContext,
    spec: ToolSpec,
    policy: PolicyDecision,
) -> ToolResult:
    label = str(spec.visible_label or call.name)
    pending = PendingAction(
        id=f"{call.id}:policy-confirmation",
        run_id=context.run_id,
        tool_call_id=call.id,
        action_type="policy_confirmation",
        title=f"Confirm {label}",
        prompt=policy.reason,
        handoff_view="policy_confirmation",
        payload={
            "policy_confirmation": True,
            "deferred_tool_call": call.model_dump(mode="json"),
            "policy_decision": policy.as_dict(),
            "tool_name": call.name,
            "tool_label": label,
            "arguments": dict(call.arguments),
            "confirm_tool_name": call.name,
            "confirm_tool_args": {**call.arguments, "confirmed": True},
        },
    )
    observation = Observation(
        id=f"{call.id}:policy-confirmation",
        run_id=context.run_id,
        tool_call_id=call.id,
        source=call.name,
        status="requires_user_action",
        summary=policy.reason,
        data={
            "policy_confirmation": True,
            "tool_name": call.name,
            "tool_label": label,
            "confirm_tool_name": call.name,
            "confirm_tool_args": {**call.arguments, "confirmed": True},
        },
        metadata={"policy": policy.as_dict()},
    )
    return ToolResult(
        call=call,
        status="requires_user_action",
        summary=policy.reason,
        data={
            "policy_confirmation": True,
            "tool_name": call.name,
            "tool_label": label,
            "confirm_tool_name": call.name,
            "confirm_tool_args": {**call.arguments, "confirmed": True},
        },
        observation=observation,
        pending_action=pending,
        metadata={"governance": {"policy": policy.as_dict(), "budget": {}}},
    )


def _copy_context_mapping(value: dict[str, Any]) -> dict[str, Any]:
    try:
        return deepcopy(value)
    except (TypeError, ValueError):
        return dict(value)


def _normalize_async_result(result: ToolResult, spec: ToolSpec) -> ToolResult:
    if not spec.async_tool or result.status != "succeeded":
        return result
    if bool(result.metadata.get("completed_inline")):
        return result
    observation = result.observation
    if observation is not None:
        observation = observation.model_copy(update={"status": "pending"})
    metadata = dict(result.metadata)
    metadata["typed_async"] = True
    return result.model_copy(
        update={
            "status": "waiting_async",
            "observation": observation,
            "metadata": metadata,
        }
    )


def _keep_single_suspending_result(results: list[ToolResult]) -> list[ToolResult]:
    pending_seen = False
    normalized: list[ToolResult] = []
    for result in results:
        if result.status not in {"requires_user_action", "waiting_async"}:
            normalized.append(result)
            continue
        if not pending_seen:
            pending_seen = True
            normalized.append(result)
            continue
        rejected_call = result.call or ToolCall(
            id=result.tool_call_id,
            name=result.name,
        )
        rejected = _failed_result(
            rejected_call,
            "multiple suspending tools must be executed sequentially",
        )
        rejected.metadata = {
            **result.metadata,
            "suspension_rejected": True,
            "original_status": result.status,
        }
        normalized.append(rejected)
    return normalized


def normalize_arguments(arguments: dict[str, Any], spec: ToolSpec) -> dict[str, Any]:
    normalized = dict(arguments or {})
    contract = spec.argument_contract if isinstance(spec.argument_contract, dict) else {}
    aliases = as_dict(contract.get("aliases"))
    for canonical, candidates in aliases.items():
        if _has_value(normalized, str(canonical)):
            continue
        for candidate in candidates if isinstance(candidates, list) else []:
            value = _nested_value(normalized, str(candidate))
            if _present(value):
                normalized[str(canonical)] = value
                if "." not in str(candidate) and str(candidate) != str(canonical):
                    normalized.pop(str(candidate), None)
                break
    coercions = as_dict(contract.get("coerce"))
    for field_name, target_type in coercions.items():
        if field_name not in normalized:
            continue
        normalized[field_name] = _coerce(normalized[field_name], str(target_type))
    return normalized


def _failed_result(call: ToolCall, error: str, *, retryable: bool = False) -> ToolResult:
    return ToolResult(
        call=call,
        tool_call_id=call.id,
        name=call.name,
        status="failed",
        summary=error,
        error=error,
        retryable=retryable,
    )


def _argument_schema_error(arguments: dict[str, Any], spec: ToolSpec) -> str:
    if spec.runtime_policy.get("argument_validation_authority") == "handler":
        return ""
    schema = spec.formal_parameters_schema()
    if not schema:
        return ""
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
    except SchemaError as exc:
        return f"tool parameters schema is invalid: {exc.message}"
    errors = sorted(
        validator.iter_errors(arguments),
        key=lambda item: ".".join(str(part) for part in item.absolute_path),
    )
    if not errors:
        return ""
    error = errors[0]
    path = ".".join(str(item) for item in error.absolute_path)
    return f"{path}: {error.message}" if path else error.message


def _invalid_arguments_result(
    call: ToolCall,
    context: ToolExecutionContext,
    spec: ToolSpec,
    error: str,
) -> ToolResult:
    if str(spec.runtime_policy.get("invalid_arguments_mode") or "") != "skip":
        result = _failed_result(call, f"tool arguments do not match schema: {error}")
        result.metadata = {
            "error_code": "TOOL_ARGUMENT_SCHEMA_INVALID",
            "schema_validation_error": error,
            "executed": False,
        }
        return result
    summary = "Tool arguments violated the contract; the call was skipped and control returned to the agent."
    data = {
        "status": "skipped",
        "reason_code": "invalid_tool_arguments",
        "fallback": "continue_with_parent_agent",
    }
    observation = Observation(
        id=f"{call.id}:invalid-arguments",
        run_id=context.run_id,
        tool_call_id=call.id,
        source=call.name,
        status="succeeded",
        outcome="skipped",
        summary=summary,
        data=data,
        metadata={
            "error_code": "TOOL_ARGUMENT_SCHEMA_INVALID",
            "schema_validation_error": error,
        },
    )
    return ToolResult(
        call=call,
        status="succeeded",
        outcome="skipped",
        summary=summary,
        data=data,
        observation=observation,
        metadata={
            "visible": False,
            "error_code": "TOOL_ARGUMENT_SCHEMA_INVALID",
            "schema_validation_error": error,
            "executed": False,
        },
    )


def _with_runtime_metrics(
    result: ToolResult,
    started_at: float,
    *,
    phase: str,
    executed: bool = True,
) -> ToolResult:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    metrics = as_dict(metadata.get("runtime_metrics"))
    result.metadata = {
        **metadata,
        "runtime_metrics": {
            **metrics,
            "latency_ms": max(0, int((time.perf_counter() - started_at) * 1000)),
            "phase": phase,
            "executed": executed,
        },
    }
    return result


def _replayed_result(cached: ToolResult, call: ToolCall) -> ToolResult:
    metadata = dict(cached.metadata)
    metadata["idempotent_replay"] = True
    observation = cached.observation
    if observation is not None:
        observation = observation.model_copy(
            update={
                "id": f"{call.id}:observation",
                "tool_call_id": call.id,
            }
        )
    pending_action = cached.pending_action
    if pending_action is not None:
        pending_action = pending_action.model_copy(
            update={
                "id": f"{call.id}:action",
                "tool_call_id": call.id,
            }
        )
    return cached.model_copy(
        deep=True,
        update={
            "call": call,
            "tool_call_id": call.id,
            "observation": observation,
            "pending_action": pending_action,
            "metadata": metadata,
        },
    )


def _nested_value(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _has_value(payload: dict[str, Any], path: str) -> bool:
    return _present(_nested_value(payload, path))


def _present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None and value != [] and value != {}


def _coerce(value: Any, target_type: str) -> Any:
    if target_type in {"string", "str"}:
        return str(value)
    if target_type in {"int", "integer"}:
        return int(value)
    if target_type in {"float", "number"}:
        return float(value)
    if target_type in {"bool", "boolean"}:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return bool(value)
    return value


def _positive_number(value: Any, default: float) -> float:
    try:
        return max(0.001, float(value))
    except (TypeError, ValueError):
        return default


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _tool_reexecution_safe(spec: ToolSpec) -> bool:
    explicit = spec.runtime_policy.get("idempotency_reexecution_safe")
    if isinstance(explicit, bool):
        return explicit
    side_effect = str(spec.runtime_policy.get("side_effect") or "").strip().lower()
    return bool(spec.read_only or side_effect == "none")
