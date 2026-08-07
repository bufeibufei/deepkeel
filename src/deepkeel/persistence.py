from __future__ import annotations

import json
from copy import deepcopy
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from deepkeel.contracts import AgentMessage, Artifact, Observation, RunContext, RunStatus, ToolCall
from deepkeel.migrations import StateMigrationError, StateMigrationRegistry

if TYPE_CHECKING:
    from deepkeel.runtime_api import RuntimeResult

from deepkeel.type_narrowing import as_dict


CHECKPOINT_SCHEMA_VERSION = "harness-checkpoint-v2"
DURABLE_CHECKPOINT_SCHEMA_VERSION = "harness-durable-checkpoint-v2"
RUNTIME_SCHEMA_VERSION = "harness-runtime-v2"


class CheckpointCompatibilityError(ValueError):
    code = "CHECKPOINT_INCOMPATIBLE"


def _require_supported_version(
    value: dict[str, Any],
    *,
    supported: set[str],
    contract_name: str,
) -> None:
    version = str(value.get("schema_version") or "").strip()
    if value and not version:
        raise CheckpointCompatibilityError(
            f"missing {contract_name} schema version"
        )
    if version and version not in supported:
        raise CheckpointCompatibilityError(
            f"unsupported {contract_name} schema version: {version}"
        )


class DurableCheckpointStore(Protocol):
    """Business recovery state, independent from engine graph checkpoints."""

    def load(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> dict[str, Any] | None: ...

    def save(
        self,
        run_id: str,
        state: dict[str, Any],
        *,
        session: Any = None,
        user_id: str = "",
    ) -> None: ...

    def delete(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> None: ...

    def exists(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> bool: ...

    def list_ids(
        self,
        *,
        session: Any = None,
        user_id: str = "",
        limit: int = 100,
    ) -> tuple[str, ...]: ...


class InMemoryDurableCheckpointStore:
    """Thread-safe reference adapter for the durable checkpoint Port."""

    def __init__(self) -> None:
        self._states: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = Lock()

    def load(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> dict[str, Any] | None:
        del session
        with self._lock:
            state = self._states.get((user_id, run_id))
            return deepcopy(state) if state is not None else None

    def save(
        self,
        run_id: str,
        state: dict[str, Any],
        *,
        session: Any = None,
        user_id: str = "",
    ) -> None:
        del session
        with self._lock:
            self._states[(user_id, run_id)] = deepcopy(state)

    def delete(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> None:
        del session
        with self._lock:
            self._states.pop((user_id, run_id), None)

    def exists(
        self,
        run_id: str,
        *,
        session: Any = None,
        user_id: str = "",
    ) -> bool:
        del session
        with self._lock:
            return (user_id, run_id) in self._states

    def list_ids(
        self,
        *,
        session: Any = None,
        user_id: str = "",
        limit: int = 100,
    ) -> tuple[str, ...]:
        del session
        with self._lock:
            run_ids = sorted(
                run_id
                for stored_user_id, run_id in self._states
                if stored_user_id == user_id
            )
        return tuple(run_ids[: max(0, int(limit))])


def checkpoint_from_runtime(
    previous_runtime: dict[str, Any] | None,
    *,
    migrations: StateMigrationRegistry | None = None,
) -> dict[str, Any]:
    runtime = previous_runtime if isinstance(previous_runtime, dict) else {}
    runtime = _migrate_if_needed(
        runtime,
        state_kind="runtime",
        target_version=RUNTIME_SCHEMA_VERSION,
        migrations=migrations,
    )
    _require_supported_version(
        runtime,
        supported={RUNTIME_SCHEMA_VERSION},
        contract_name="runtime",
    )
    checkpoint = as_dict(runtime.get("checkpoint"))
    checkpoint = _migrate_if_needed(
        checkpoint,
        state_kind="checkpoint",
        target_version=CHECKPOINT_SCHEMA_VERSION,
        migrations=migrations,
    )
    _require_supported_version(
        checkpoint,
        supported={CHECKPOINT_SCHEMA_VERSION},
        contract_name="checkpoint",
    )
    return checkpoint


def checkpoint_from_durable_state(
    value: dict[str, Any] | None,
    *,
    migrations: StateMigrationRegistry | None = None,
) -> dict[str, Any]:
    state = value if isinstance(value, dict) else {}
    version = str(state.get("schema_version") or "").strip()
    if version and version not in {CHECKPOINT_SCHEMA_VERSION, DURABLE_CHECKPOINT_SCHEMA_VERSION}:
        state = _migrate_if_needed(
            state,
            state_kind="durable_checkpoint",
            target_version=DURABLE_CHECKPOINT_SCHEMA_VERSION,
            migrations=migrations,
        )
    _require_supported_version(
        state,
        supported={CHECKPOINT_SCHEMA_VERSION, DURABLE_CHECKPOINT_SCHEMA_VERSION},
        contract_name="durable checkpoint",
    )
    if str(state.get("schema_version") or "") == CHECKPOINT_SCHEMA_VERSION:
        return state
    direct = as_dict(state.get("checkpoint"))
    if direct:
        direct = _migrate_if_needed(
            direct,
            state_kind="checkpoint",
            target_version=CHECKPOINT_SCHEMA_VERSION,
            migrations=migrations,
        )
        _require_supported_version(
            direct,
            supported={CHECKPOINT_SCHEMA_VERSION},
            contract_name="checkpoint",
        )
        return direct
    runtime = as_dict(state.get("runtime"))
    return checkpoint_from_runtime(runtime, migrations=migrations)


def durable_state_from_result(
    result: RuntimeResult,
    *,
    run_id: str,
    thread_id: str,
) -> dict[str, Any]:
    runtime = {
        "schema_version": result.schema_version,
        "core_contract_version": result.core_contract_version,
        "core_version": result.core_version,
        "loop_engine": result.loop_engine,
        "mode": result.mode,
        "status": result.status.value,
        "stop_reason": result.stop_reason,
        "step_count": result.step_count,
        "pending_action": result.pending_action.model_dump(mode="json")
        if result.pending_action is not None
        else None,
        "active_task": result.active_task,
        "checkpoint": result.checkpoint,
        "diagnostics": result.diagnostics,
    }
    return {
        "schema_version": DURABLE_CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "thread_id": thread_id,
        "question": result.question,
        "context_snapshot": result.context_snapshot,
        "skill_activation": result.skill_activation,
        "runtime": {
            key: runtime.get(key)
            for key in (
                "schema_version",
                "core_contract_version",
                "core_version",
                "loop_engine",
                "mode",
                "status",
                "stop_reason",
                "step_count",
                "resume_token",
                "pending_action",
                "active_task",
                "checkpoint",
                "diagnostics",
            )
            if runtime.get(key) not in (None, "", [], {})
        },
        "pending_action": result.pending_action.model_dump(mode="json")
        if result.pending_action is not None
        else None,
    }


def resume_payload_from_context(short_context: dict[str, Any] | None) -> dict[str, Any]:
    short = short_context if isinstance(short_context, dict) else {}
    observation = short.get("resume_observation")
    if isinstance(observation, dict):
        return observation
    observations = short.get("resume_observations")
    if isinstance(observations, list) and observations and isinstance(observations[-1], dict):
        return as_dict(observations[-1])
    return {"status": "succeeded", "summary": "The external action is complete."}


def restore_run_context(
    *,
    checkpoint: dict[str, Any],
    resume_payload: dict[str, Any],
    run_id: str,
    thread_id: str,
    turn_id: str,
    user_id: str,
    skill_activation: dict[str, Any] | None = None,
    model_policy: dict[str, Any] | None = None,
    migrations: StateMigrationRegistry | None = None,
) -> RunContext:
    checkpoint = _migrate_if_needed(
        checkpoint,
        state_kind="checkpoint",
        target_version=CHECKPOINT_SCHEMA_VERSION,
        migrations=migrations,
    )
    _require_supported_version(
        checkpoint,
        supported={CHECKPOINT_SCHEMA_VERSION},
        contract_name="checkpoint",
    )
    checkpoint_run_id = str(checkpoint.get("run_id") or "").strip()
    if checkpoint_run_id and checkpoint_run_id != run_id:
        raise CheckpointCompatibilityError(
            f"checkpoint run_id mismatch: expected {run_id}, found {checkpoint_run_id}"
        )
    messages = [AgentMessage.model_validate(item) for item in checkpoint.get("messages", []) if isinstance(item, dict)]
    observations = [Observation.model_validate(item) for item in checkpoint.get("observations", []) if isinstance(item, dict)]
    artifacts = [Artifact.model_validate(item) for item in checkpoint.get("artifacts", []) if isinstance(item, dict)]
    pending_action = as_dict(checkpoint.get("pending_action"))
    pending_async = as_dict(checkpoint.get("pending_async"))
    pending = pending_action or pending_async
    if _confirmed_policy_tool_call(pending_action, resume_payload):
        payload = as_dict(pending_action.get("payload"))
        deferred = as_dict(payload.get("deferred_tool_call"))
        call = ToolCall.model_validate(deferred)
        metadata = as_dict(checkpoint.get("metadata"))
        grants = as_dict(metadata.get("confirmation_grants"))
        metadata = {
            **metadata,
            "recovered_from_durable_checkpoint": True,
            "confirmation_grants": {
                **grants,
                call.id: {
                    "confirmed": True,
                    "run_id": run_id,
                    "tool_call_id": call.id,
                    "tool_name": call.name,
                    "policy_id": str(as_dict(payload.get("policy_decision")).get("policy_id") or ""),
                },
            },
        }
        return RunContext(
            run_id=run_id,
            thread_id=thread_id,
            turn_id=turn_id,
            user_id=user_id,
            status=RunStatus.EXECUTING_TOOLS,
            messages=messages,
            observations=observations,
            pending_tool_calls=[call],
            artifacts=artifacts,
            skill_activation=skill_activation or {},
            model_policy=model_policy or {},
            budget_state=as_dict(checkpoint.get("budget_state")),
            metadata=metadata,
            step_count=int(checkpoint.get("step_count") or 0),
        )
    tool_call_id = str(pending.get("tool_call_id") or resume_payload.get("tool_call_id") or "resume")
    source = str(
        pending.get("tool_name")
        or pending.get("action_type")
        or resume_payload.get("tool_name")
        or resume_payload.get("source")
        or "resume"
    )
    summary = str(resume_payload.get("summary") or "The external action is complete.")
    data = (
        as_dict(resume_payload.get("data"))
        if isinstance(resume_payload.get("data"), dict)
        else dict(resume_payload)
    )
    observation = Observation(
        id=f"resume-{uuid4()}",
        run_id=run_id,
        tool_call_id=tool_call_id,
        source=source,
        status="failed" if str(resume_payload.get("status") or "").lower() == "failed" else "succeeded",
        summary=summary,
        data=data,
        error=str(resume_payload.get("error") or ""),
        metadata={"resume_source": "durable_checkpoint"},
    )
    observations.append(observation)
    restored_skill = dict(skill_activation or {})
    if observation.status == "succeeded" and source:
        completed_tools = {
            str(name).strip()
            for name in restored_skill.get("completed_tools", [])
            if str(name).strip()
        }
        completed_tools.add(source)
        restored_skill["completed_tools"] = sorted(completed_tools)
        artifact_type = str(resume_payload.get("artifact_type") or "").strip()
        if artifact_type:
            artifact_id = str(
                resume_payload.get("artifact_id")
                or resume_payload.get("session_id")
                or resume_payload.get("case_id")
                or resume_payload.get("source_id")
                or f"resume-artifact-{uuid4()}"
            )
            if not any(item.id == artifact_id for item in artifacts):
                artifacts.append(
                    Artifact(
                        id=artifact_id,
                        run_id=run_id,
                        artifact_type=artifact_type,
                        title=str(resume_payload.get("title") or ""),
                        summary=summary,
                        source_id=str(resume_payload.get("source_id") or artifact_id),
                        data=dict(resume_payload),
                        metadata={"resume_observation": True},
                    )
                )
    observation_content = json.dumps(
        {"status": observation.status, "summary": observation.summary, "data": observation.data},
        ensure_ascii=False,
    )
    if messages:
        messages.append(
            AgentMessage(
                id=f"tool-resume-{uuid4()}",
                role="tool",
                tool_call_id=tool_call_id,
                name=source,
                content=observation_content,
            )
        )
    else:
        messages.append(
            AgentMessage(
                id=f"resume-context-{uuid4()}",
                role="system",
                content=f"External task resume information: {observation_content}",
            )
        )
    return RunContext(
        run_id=run_id,
        thread_id=thread_id,
        turn_id=turn_id,
        user_id=user_id,
        status=RunStatus.REASONING,
        messages=messages,
        observations=observations,
        artifacts=artifacts,
        skill_activation=restored_skill,
        model_policy=model_policy or {},
        budget_state=as_dict(checkpoint.get("budget_state")),
        metadata={
            **as_dict(checkpoint.get("metadata")),
            "recovered_from_durable_checkpoint": True,
        },
        step_count=int(checkpoint.get("step_count") or 0),
    )


def _migrate_if_needed(
    payload: dict[str, Any],
    *,
    state_kind: str,
    target_version: str,
    migrations: StateMigrationRegistry | None,
) -> dict[str, Any]:
    if not payload:
        return payload
    version = str(payload.get("schema_version") or "").strip()
    if not version or version == target_version or migrations is None:
        return payload
    try:
        return migrations.migrate(
            state_kind,
            payload,
            target_version=target_version,
        )
    except StateMigrationError as exc:
        raise CheckpointCompatibilityError(str(exc)) from exc


def _confirmed_policy_tool_call(
    pending_action: dict[str, Any],
    resume_payload: dict[str, Any],
) -> bool:
    payload = as_dict(pending_action.get("payload"))
    data = as_dict(resume_payload.get("data"))
    status = str(resume_payload.get("status") or "").lower()
    confirmed = data.get("confirmed") is True or resume_payload.get("confirmed") is True or status in {
        "confirmed",
        "approved",
    }
    return (
        confirmed
        and (
            pending_action.get("action_type") == "policy_confirmation"
            or payload.get("policy_confirmation") is True
        )
        and isinstance(payload.get("deferred_tool_call"), dict)
    )
