from __future__ import annotations

import asyncio
from copy import deepcopy
import inspect
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Mapping, Protocol
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from deepkeel.budget import (
    TOOL_CONCURRENCY,
    TOOL_CALLS,
    BudgetLedger,
    BudgetRequest,
    InMemoryBudgetLedger,
)
from deepkeel.async_ports import (
    AsyncToolExecutionStore,
    AsyncToolExecutionStoreAdapter,
    run_sync_adapter,
)
from deepkeel.clarifications import clarification_for_missing_arguments, clarification_tool_result
from deepkeel.contracts import Observation, PendingAction, ToolCall, ToolResult
from deepkeel.deadlines import ensure_time_remaining
from deepkeel.failures import RunCanceledError, RunDeadlineExceededError
from deepkeel.hooks import (
    HookAction,
    HookAudit,
    HookInvocation,
    HookPoint,
    HookRunner,
)
from deepkeel.policy import (
    DefaultPolicyEngine,
    PolicyDecision,
    PolicyEngine,
    PolicyRequest,
)
from deepkeel.ports import RuntimeSession
from deepkeel.tool_execution import (
    InMemoryToolExecutionStore,
    ToolExecutionClaim,
    ToolExecutionContext,
    ToolExecutionStore,
)
from deepkeel.scope import scoped_adapter_operation
from deepkeel.tool_registry import ToolRegistry, ToolSpec
from deepkeel.type_narrowing import as_dict

from deepkeel.tool_execution_hooks import (
    _emit_hook_audits,
    _hook_confirmation_result,
    _tool_hook_invocation,
)
from deepkeel.tool_execution_support import (
    _argument_schema_error,
    _bind_authoritative_context_arguments,
    _confirmation_grant,
    _confirmation_required_result,
    _failed_result,
    _invalid_arguments_result,
    _keep_single_suspending_result,
    _normalize_async_result,
    _positive_int,
    _positive_number,
    _replayed_result,
    _tool_reexecution_safe,
    _with_runtime_metrics,
    normalize_arguments,
)
from deepkeel.tool_executor_contracts import ToolHandler, ToolPreflight


async def _araise_if_cancelled(context: ToolExecutionContext, *, force: bool = False) -> None:
    await run_sync_adapter(
        context.run_control.raise_if_cancelled,
        context.operational_run_id,
        force=force,
    )


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        preflight: ToolPreflight | None = None,
        max_parallel_tools: int = 4,
        execution_store: ToolExecutionStore | None = None,
        async_execution_store: AsyncToolExecutionStore | None = None,
        policy_engine: PolicyEngine | None = None,
        budget_ledger: BudgetLedger | None = None,
        hook_runner: HookRunner | None = None,
        claim_lease_seconds: float = 300,
        max_idempotent_attempts: int = 2,
    ):
        self.registry = registry
        self.preflight = preflight
        self.max_parallel_tools = max(1, int(max_parallel_tools))
        if execution_store is not None and async_execution_store is not None:
            raise ValueError("configure either execution_store or async_execution_store, not both")
        self.execution_store = (
            execution_store
            if execution_store is not None
            else None
            if async_execution_store is not None
            else InMemoryToolExecutionStore()
        )
        self.async_execution_store = async_execution_store
        if async_execution_store is not None:
            self._execution_store = async_execution_store
        else:
            assert self.execution_store is not None
            self._execution_store = AsyncToolExecutionStoreAdapter(self.execution_store)
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
            str(name): dict(schema) for name, schema in schemas.items() if str(name).strip()
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
                    message=before.decision.confirmation_message or before.decision.reason,
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
        await _araise_if_cancelled(context)
        ensure_time_remaining(context.deadline_monotonic)
        started_at = time.perf_counter()
        try:
            spec = self.registry.get(call.name)
        except KeyError:
            return _with_runtime_metrics(
                _failed_result(call, "unknown tool"), started_at, phase="lookup"
            )

        try:
            normalized = call.model_copy(
                update={"arguments": normalize_arguments(call.arguments, spec)}
            )
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
                return _with_runtime_metrics(
                    _failed_result(normalized, denial), started_at, phase="preflight"
                )

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
            claim_operation = scoped_adapter_operation(
                self._execution_store,
                "claim",
                context.scope,
            )
            claim_kwargs: dict[str, Any] = {
                "run_id": context.run_id,
                "call": normalized,
                "lease_seconds": lease_seconds,
                "max_attempts": max_attempts,
                "reexecution_safe": reexecution_safe,
            }
            if getattr(claim_operation, "__name__", "") == "claim_scoped":
                claim_kwargs["scope"] = context.scope
            claim = await claim_operation(**claim_kwargs)
        except Exception as exc:
            return _with_runtime_metrics(
                _failed_result(normalized, f"tool execution claim failed: {exc}", retryable=True),
                started_at,
                phase="idempotent_claim",
                executed=False,
            )
        if claim.status == "replay":
            try:
                persisted = await self._execution_store.replay(claim)
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
            busy = _failed_result(
                normalized, "tool execution is already in progress", retryable=True
            )
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
                await self._execution_store.settle(claim, result)
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

    def execute_many(
        self, calls: list[ToolCall], context: ToolExecutionContext
    ) -> list[ToolResult]:
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
            target = (
                parallel
                if spec.read_only and spec.parallel_safe and can_isolate_session
                else serial
            )
            target.append((index, call))

        if parallel:
            configured_limit = int(
                context.budget_limits.get(TOOL_CONCURRENCY) or self.max_parallel_tools
            )
            workers = max(1, min(self.max_parallel_tools, configured_limit, len(parallel)))
            concurrency_budget = self.budget_ledger.consume(
                BudgetRequest(
                    run_id=context.operational_run_id,
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
        return _keep_single_suspending_result([result for result in results if result is not None])

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
            target = (
                parallel
                if spec.read_only and spec.parallel_safe and can_isolate_session
                else serial
            )
            target.append((index, call))

        if parallel:
            configured_limit = int(
                context.budget_limits.get(TOOL_CONCURRENCY) or self.max_parallel_tools
            )
            workers = max(1, min(self.max_parallel_tools, configured_limit, len(parallel)))
            concurrency_budget = await run_sync_adapter(
                self.budget_ledger.consume,
                BudgetRequest(
                    run_id=context.operational_run_id,
                    metric=TOOL_CONCURRENCY,
                    amount=workers,
                    limit=configured_limit,
                    operation_id=(
                        f"tool-concurrency:{context.turn_id}:"
                        + ":".join(sorted(call.id for _, call in parallel))
                    ),
                    aggregation="max",
                    metadata={"parallel_call_count": len(parallel)},
                ),
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
        return _keep_single_suspending_result([result for result in results if result is not None])

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
                    await asyncio.to_thread(close)

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
        budget = await run_sync_adapter(
            self.budget_ledger.consume,
            BudgetRequest(
                run_id=context.operational_run_id,
                metric=TOOL_CALLS,
                limit=context.budget_limits.get(TOOL_CALLS),
                operation_id=f"tool-call:{call.idempotency_key or call.id}",
                metadata={"tool_name": call.name},
            ),
        )
        if not budget.allowed:
            result = _failed_result(call, budget.reason)
            result.metadata["governance"] = {
                "policy": policy.as_dict(),
                "budget": budget.as_dict(),
            }
            return _with_runtime_metrics(result, started_at, phase="budget", executed=False)
        try:
            native_execute = getattr(handler, "aexecute", None)
            if callable(native_execute):
                raw = await native_execute(call, context)
            else:
                # Tool handlers are sync by default. Running them in the event
                # loop would stall every active run while MCP, database, or
                # filesystem I/O is in flight.
                raw = await asyncio.to_thread(handler, call, context)
                if inspect.isawaitable(raw):
                    raw = await raw
            context.raise_if_fence_lost()
            await _araise_if_cancelled(context, force=True)
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
