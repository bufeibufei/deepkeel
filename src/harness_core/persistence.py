from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from harness_core.contracts import AgentMessage, Artifact, Observation, RunContext, RunStatus, ToolCall

if TYPE_CHECKING:
    from harness_core.runtime_api import RuntimeResult


CHECKPOINT_SCHEMA_VERSION = "harness-checkpoint-v1"
DURABLE_CHECKPOINT_SCHEMA_VERSION = "harness-durable-checkpoint-v1"
RUNTIME_SCHEMA_VERSION = "harness-runtime-v1"


class CheckpointCompatibilityError(ValueError):
    code = "CHECKPOINT_INCOMPATIBLE"


def _require_supported_version(
    value: dict[str, Any],
    *,
    supported: set[str],
    contract_name: str,
) -> None:
    version = str(value.get("schema_version") or "").strip()
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


# Compatibility name retained for hosts using the v1 public SDK.
CheckpointStore = DurableCheckpointStore


def checkpoint_from_runtime(previous_runtime: dict[str, Any] | None) -> dict[str, Any]:
    runtime = previous_runtime if isinstance(previous_runtime, dict) else {}
    _require_supported_version(
        runtime,
        supported={RUNTIME_SCHEMA_VERSION},
        contract_name="runtime",
    )
    checkpoint = runtime.get("checkpoint") if isinstance(runtime.get("checkpoint"), dict) else {}
    _require_supported_version(
        checkpoint,
        supported={CHECKPOINT_SCHEMA_VERSION},
        contract_name="checkpoint",
    )
    return checkpoint


def checkpoint_from_durable_state(value: dict[str, Any] | None) -> dict[str, Any]:
    state = value if isinstance(value, dict) else {}
    _require_supported_version(
        state,
        supported={CHECKPOINT_SCHEMA_VERSION, DURABLE_CHECKPOINT_SCHEMA_VERSION},
        contract_name="durable checkpoint",
    )
    if str(state.get("schema_version") or "") == CHECKPOINT_SCHEMA_VERSION:
        return state
    direct = state.get("checkpoint") if isinstance(state.get("checkpoint"), dict) else {}
    if direct:
        _require_supported_version(
            direct,
            supported={CHECKPOINT_SCHEMA_VERSION},
            contract_name="checkpoint",
        )
        return direct
    runtime = state.get("agent_runtime") if isinstance(state.get("agent_runtime"), dict) else {}
    return checkpoint_from_runtime(runtime)


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
        "agent_runtime": {
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
        return observations[-1]
    return {"status": "succeeded", "summary": "外部操作已完成。"}


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
) -> RunContext:
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
    pending_action = (
        checkpoint.get("pending_action")
        if isinstance(checkpoint.get("pending_action"), dict)
        else {}
    )
    pending_async = (
        checkpoint.get("pending_async")
        if isinstance(checkpoint.get("pending_async"), dict)
        else {}
    )
    pending = pending_action or pending_async
    if _confirmed_policy_tool_call(pending_action, resume_payload):
        payload = pending_action.get("payload") if isinstance(pending_action.get("payload"), dict) else {}
        deferred = payload.get("deferred_tool_call") if isinstance(payload.get("deferred_tool_call"), dict) else {}
        call = ToolCall.model_validate(deferred)
        metadata = checkpoint.get("metadata") if isinstance(checkpoint.get("metadata"), dict) else {}
        grants = metadata.get("confirmation_grants") if isinstance(metadata.get("confirmation_grants"), dict) else {}
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
                    "policy_id": str((payload.get("policy_decision") or {}).get("policy_id") or ""),
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
            budget_state=checkpoint.get("budget_state")
            if isinstance(checkpoint.get("budget_state"), dict)
            else {},
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
    summary = str(resume_payload.get("summary") or "外部操作已完成。")
    data = resume_payload.get("data") if isinstance(resume_payload.get("data"), dict) else resume_payload
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
                content=f"外部任务恢复信息：{observation_content}",
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
        budget_state=checkpoint.get("budget_state")
        if isinstance(checkpoint.get("budget_state"), dict)
        else {},
        metadata={
            **(
                checkpoint.get("metadata")
                if isinstance(checkpoint.get("metadata"), dict)
                else {}
            ),
            "recovered_from_durable_checkpoint": True,
        },
        step_count=int(checkpoint.get("step_count") or 0),
    )


def _confirmed_policy_tool_call(
    pending_action: dict[str, Any],
    resume_payload: dict[str, Any],
) -> bool:
    payload = pending_action.get("payload") if isinstance(pending_action.get("payload"), dict) else {}
    data = resume_payload.get("data") if isinstance(resume_payload.get("data"), dict) else {}
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
