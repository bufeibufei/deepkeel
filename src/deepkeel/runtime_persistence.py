from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from deepkeel.async_ports import run_sync_adapter
from deepkeel.events import envelope_runtime_event
from deepkeel.leases import ExecutionFence
from deepkeel.persistence import durable_state_from_result
from deepkeel.runtime_api import RuntimeResult, RuntimeStreamEvent
from deepkeel.scope import (
    RuntimeScope,
    require_legacy_compatible_scope,
    scoped_adapter_operation,
)
from deepkeel.state_store import RuntimeStateMutation
from deepkeel.type_narrowing import as_dict


EventSink = Callable[[dict[str, Any]], None]


def _runtime_state_mutation_id(
    run_id: str,
    status: str,
    durable_state: dict[str, Any],
) -> str:
    encoded = json.dumps(
        durable_state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    return f"{run_id}:{status}:{digest}"


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


async def acleanup_run(
    runtime: Any,
    result: RuntimeResult | None,
    *,
    run_id: str,
    session: Any = None,
    user_id: str = "",
    scope: RuntimeScope | None = None,
    graph_thread_ids: list[str] | None = None,
) -> dict[str, str]:
    """Clean portable and graph checkpoints using the configured async authority."""

    cleanup: dict[str, str] = {}
    errors: dict[str, str] = {}
    resolved_scope = scope or RuntimeScope(user_id=user_id)
    if runtime.async_checkpoint_store is not None:
        try:
            legacy_user_id = require_legacy_compatible_scope(
                resolved_scope,
                adapter_name=type(runtime.async_checkpoint_store).__name__,
            )
            await runtime.async_checkpoint_store.delete(
                run_id,
                session=session,
                user_id=legacy_user_id,
            )
            cleanup["durable"] = "deleted"
        except Exception as exc:
            cleanup["durable"] = "failed"
            errors["durable"] = str(exc)
    elif runtime.checkpoint_store is not None:
        try:
            legacy_user_id = require_legacy_compatible_scope(
                resolved_scope,
                adapter_name=type(runtime.checkpoint_store).__name__,
            )
            await run_sync_adapter(
                runtime.checkpoint_store.delete,
                run_id,
                session=session,
                user_id=legacy_user_id,
            )
            cleanup["durable"] = "deleted"
        except Exception as exc:
            cleanup["durable"] = "failed"
            errors["durable"] = str(exc)
    else:
        cleanup["durable"] = "not_configured"

    thread_ids = list(
        dict.fromkeys([resolved_scope.qualify_identity(run_id), *(graph_thread_ids or [])])
    )
    async_delete_thread = getattr(runtime.checkpointer, "adelete_thread", None)
    delete_thread = getattr(runtime.checkpointer, "delete_thread", None)
    if callable(async_delete_thread):
        for thread_id in thread_ids:
            try:
                await async_delete_thread(thread_id)
            except NotImplementedError:
                if not callable(delete_thread):
                    errors[f"langgraph:{thread_id}"] = (
                        "async checkpoint deletion is not implemented and no sync fallback exists"
                    )
                    continue
                try:
                    await run_sync_adapter(delete_thread, thread_id)
                except Exception as exc:
                    errors[f"langgraph:{thread_id}"] = str(exc)
            except Exception as exc:
                errors[f"langgraph:{thread_id}"] = str(exc)
        cleanup["langgraph"] = (
            "failed" if any(key.startswith("langgraph:") for key in errors) else "deleted"
        )
    elif callable(delete_thread):
        for thread_id in thread_ids:
            try:
                await run_sync_adapter(delete_thread, thread_id)
            except Exception as exc:
                errors[f"langgraph:{thread_id}"] = str(exc)
        cleanup["langgraph"] = (
            "failed" if any(key.startswith("langgraph:") for key in errors) else "deleted"
        )
    else:
        cleanup["langgraph"] = "unsupported"

    if result is not None:
        diagnostics = result.diagnostics
        recovery = as_dict(diagnostics.get("recovery"))
        recovery["checkpoint_cleanup"] = cleanup
        if errors:
            recovery["checkpoint_cleanup_errors"] = errors
        diagnostics["recovery"] = recovery
    operational_run_id = resolved_scope.qualify_identity(run_id)
    await run_sync_adapter(runtime.budget_ledger.clear, operational_run_id)
    await run_sync_adapter(runtime.run_control.release, operational_run_id)
    return cleanup


async def aload_authoritative_checkpoint(
    runtime: Any,
    run_id: str,
    *,
    session: Any,
    user_id: str,
    scope: RuntimeScope,
) -> tuple[dict[str, Any], str, list[str]]:
    """Load canonical state before falling back to compatibility stores."""

    errors: list[str] = []
    if runtime.async_runtime_state_store is not None:
        try:
            snapshot = await runtime.async_runtime_state_store.load_snapshot_scoped(
                run_id,
                session=session,
                scope=scope,
            )
            state = snapshot.checkpoint_state
            if isinstance(state, dict) and state:
                return dict(state), "runtime_state_store", errors
        except Exception as exc:
            errors.append(f"runtime_state_store:{type(exc).__name__}:{exc}")
    elif runtime.runtime_state_store is not None:
        state, authority, sync_errors = await run_sync_adapter(
            runtime._load_authoritative_checkpoint,
            run_id,
            session=session,
            user_id=user_id,
            scope=scope,
        )
        if state or authority != "session_projection":
            return state, authority, sync_errors
        errors.extend(sync_errors)
        if runtime.checkpoint_store is not None:
            return state, authority, errors

    if runtime.async_checkpoint_store is not None:
        try:
            legacy_user_id = require_legacy_compatible_scope(
                scope,
                adapter_name=type(runtime.async_checkpoint_store).__name__,
            )
            loaded_state = await runtime.async_checkpoint_store.load(
                run_id,
                session=session,
                user_id=legacy_user_id,
            )
            if isinstance(loaded_state, dict) and loaded_state:
                return dict(loaded_state), "durable_checkpoint_store", errors
        except Exception as exc:
            errors.append(f"durable_checkpoint_store:{type(exc).__name__}:{exc}")
    elif runtime.runtime_state_store is None and runtime.checkpoint_store is not None:
        state, authority, sync_errors = await run_sync_adapter(
            runtime._load_authoritative_checkpoint,
            run_id,
            session=session,
            user_id=user_id,
            scope=scope,
        )
        errors.extend(error for error in sync_errors if error not in errors)
        if state:
            return state, authority, errors
    return {}, "session_projection", errors


async def event_latest_sequence(
    runtime: Any,
    run_id: str,
    *,
    scope: RuntimeScope | None = None,
    fallback: int = 0,
) -> int:
    resolved_scope = scope or RuntimeScope()
    latest: Any
    if runtime.async_event_journal is not None:
        operation = scoped_adapter_operation(
            runtime.async_event_journal,
            "latest_sequence",
            resolved_scope,
        )
        if getattr(operation, "__name__", "") == "latest_sequence_scoped":
            latest = await operation(run_id, scope=resolved_scope)
        else:
            latest = await operation(run_id)
    elif runtime.event_journal is not None:
        operation = scoped_adapter_operation(
            runtime.event_journal,
            "latest_sequence",
            resolved_scope,
        )
        if getattr(operation, "__name__", "") == "latest_sequence_scoped":
            latest = await run_sync_adapter(operation, run_id, scope=resolved_scope)
        else:
            latest = await run_sync_adapter(operation, run_id)
    else:
        latest = fallback
    return max(0, int(latest if latest is not None else fallback))


def host_owns_terminal_settlement(runtime: Any) -> bool:
    state_store = (
        runtime.async_runtime_state_store
        if runtime.async_runtime_state_store is not None
        else runtime.runtime_state_store
    )
    return (
        state_store is not None
        and getattr(
            state_store,
            "terminal_settlement_owner",
            "runtime",
        )
        == "host"
    )


async def record_checkpoint_cleanup_event(
    runtime: Any,
    result: RuntimeResult,
    *,
    run_id: str,
    thread_id: str,
    turn_id: str,
    scope: RuntimeScope,
    event_sink: EventSink | None,
    fallback_sequence: int = 0,
) -> None:
    """Persist cleanup diagnostics after the immutable terminal snapshot."""

    del event_sink
    recovery = as_dict(result.diagnostics.get("recovery"))
    cleanup = as_dict(recovery.get("checkpoint_cleanup"))
    cleanup_errors = as_dict(recovery.get("checkpoint_cleanup_errors"))
    sequence = (
        await event_latest_sequence(
            runtime,
            run_id,
            scope=scope,
            fallback=fallback_sequence,
        )
        + 1
    )
    envelope = RuntimeStreamEvent.model_validate(
        envelope_runtime_event(
            {
                "event_type": "runtime.checkpoint.cleanup.recorded",
                "title": "Checkpoint cleanup recorded",
                "summary": str(cleanup.get("status") or cleanup.get("durable") or "recorded"),
                "visibility": "internal",
                "payload": {
                    "checkpoint_cleanup": cleanup,
                    "checkpoint_cleanup_errors": cleanup_errors,
                },
            },
            run_id=run_id,
            thread_id=thread_id,
            turn_id=turn_id,
            sequence=sequence,
            scope=scope,
        )
    )
    try:
        if runtime.async_event_journal is not None:
            operation = scoped_adapter_operation(
                runtime.async_event_journal,
                "append",
                scope,
            )
            if getattr(operation, "__name__", "") == "append_scoped":
                await operation(envelope, scope=scope)
            else:
                await operation(envelope)
        elif runtime.event_journal is not None:
            operation = scoped_adapter_operation(runtime.event_journal, "append", scope)
            if getattr(operation, "__name__", "") == "append_scoped":
                await run_sync_adapter(operation, envelope, scope=scope)
            else:
                await run_sync_adapter(operation, envelope)
    except Exception as exc:
        recovery["checkpoint_cleanup_event"] = "failed"
        recovery["checkpoint_cleanup_event_error"] = str(exc)
        result.diagnostics["recovery"] = recovery
        return

    # This post-terminal event is replay/debug data. Streaming it would make the
    # public terminal event cease to be the final item observed by consumers.


async def persist_runtime_snapshot(
    runtime: Any,
    result: RuntimeResult,
    *,
    run_id: str,
    thread_id: str,
    session: Any,
    user_id: str,
    context_bundle: dict[str, Any],
    execution_fence: ExecutionFence | None = None,
    scope: RuntimeScope | None = None,
) -> None:
    status = result.status.value
    if status not in {
        "waiting_user_action",
        "waiting_user_input",
        "task_running",
        "completed",
        "failed",
        "canceled",
    }:
        return

    diagnostics = result.diagnostics
    recovery = as_dict(diagnostics.get("recovery"))
    durable_state = durable_state_from_result(
        result,
        run_id=run_id,
        thread_id=thread_id,
    )
    state_store = (
        runtime.async_runtime_state_store
        if runtime.async_runtime_state_store is not None
        else runtime.runtime_state_store
    )
    if state_store is not None:
        if execution_fence is not None:
            execution_fence.raise_if_lost()
        resume_token = str(result.checkpoint.get("resume_token") or "")
        terminal = status in {"completed", "failed", "canceled"}
        if (
            terminal
            and getattr(
                state_store,
                "terminal_settlement_owner",
                "runtime",
            )
            == "host"
        ):
            recovery["atomic_checkpoint"] = "host_settlement_required"
            diagnostics["recovery"] = recovery
            return
        event_type = "run.settled" if terminal else "runtime.checkpoint.committed"
        mutation = RuntimeStateMutation(
            mutation_id=_runtime_state_mutation_id(run_id, status, durable_state),
            run_id=run_id,
            event_type=event_type,
            target_status=status,
            event_payload={
                "status": status,
                "stop_reason": result.stop_reason,
                "resume_token": resume_token,
                "checkpoint_schema_version": str(durable_state.get("schema_version") or ""),
            },
            event_visibility="public" if terminal else "internal",
            checkpoint_type="terminal" if terminal else "runtime",
            checkpoint_state=durable_state,
            resume_token=resume_token,
            error_code=(
                str(result.error["code"] if result.error is not None else "RUN_FAILED")
                if status == "failed"
                else None
            ),
            error_message=(
                str(result.error["message"] if result.error is not None else "")
                if status == "failed"
                else None
            ),
            delete_checkpoint_types=(
                ("runtime", "suspended", "resume", "settling") if terminal else ()
            ),
            expected_version=_optional_int(context_bundle.get("runtime_state_version")),
            expected_sequence=_optional_int(context_bundle.get("runtime_state_sequence")),
            fence_token=execution_fence.token if execution_fence is not None else "",
            fence_generation=(execution_fence.generation if execution_fence is not None else 0),
        )
        try:
            if runtime.async_runtime_state_store is not None:
                receipt = await runtime.async_runtime_state_store.commit_scoped(
                    mutation,
                    session=session,
                    scope=scope or RuntimeScope(user_id=user_id),
                )
            else:
                sync_state_store = runtime.runtime_state_store
                if sync_state_store is None:
                    raise RuntimeError("runtime state store is not configured")
                commit_scoped = getattr(sync_state_store, "commit_scoped", None)
                if scope is not None and callable(commit_scoped):
                    receipt = await run_sync_adapter(
                        commit_scoped,
                        mutation,
                        session=session,
                        scope=scope,
                    )
                else:
                    legacy_user_id = require_legacy_compatible_scope(
                        scope or RuntimeScope(user_id=user_id),
                        adapter_name=type(sync_state_store).__name__,
                    )
                    receipt = await run_sync_adapter(
                        sync_state_store.commit,
                        mutation,
                        session=session,
                        user_id=legacy_user_id,
                    )
        except Exception as exc:
            recovery["atomic_checkpoint"] = "failed"
            recovery["checkpoint_error"] = str(exc)
            diagnostics["recovery"] = recovery
            raise RuntimeError("atomic runtime checkpoint commit failed") from exc
        recovery["atomic_checkpoint"] = "settled" if terminal else "persisted"
        recovery["state_receipt"] = receipt.as_dict()
        diagnostics["recovery"] = recovery
        return

    checkpoint_store = (
        runtime.async_checkpoint_store
        if runtime.async_checkpoint_store is not None
        else runtime.checkpoint_store
    )
    if checkpoint_store is None:
        return
    try:
        legacy_user_id = require_legacy_compatible_scope(
            scope or RuntimeScope(user_id=user_id),
            adapter_name=type(checkpoint_store).__name__,
        )
        if runtime.async_checkpoint_store is not None:
            await runtime.async_checkpoint_store.save(
                run_id,
                durable_state,
                session=session,
                user_id=legacy_user_id,
            )
        else:
            sync_checkpoint_store = runtime.checkpoint_store
            if sync_checkpoint_store is None:
                raise RuntimeError("checkpoint store is not configured")
            await run_sync_adapter(
                sync_checkpoint_store.save,
                run_id,
                durable_state,
                session=session,
                user_id=legacy_user_id,
            )
        recovery["durable_checkpoint"] = "persisted"
    except Exception as exc:
        recovery["durable_checkpoint"] = "failed"
        recovery["checkpoint_error"] = str(exc)
    diagnostics["recovery"] = recovery
