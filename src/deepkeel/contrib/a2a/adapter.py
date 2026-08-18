from __future__ import annotations

import asyncio
import hashlib
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Literal

from deepkeel.contrib.a2a.contracts import (
    A2AAgentCard,
    A2AArtifact,
    A2AClientPort,
    A2AMessage,
    A2APart,
    A2ASendResponse,
    A2ATask,
)
from deepkeel.failures import RunCanceledError
from deepkeel.skills import DelegationPolicy
from deepkeel.subagents.contracts import (
    SUBAGENT_EVENT_SCHEMA_VERSION,
    DelegationBatchResult,
    DelegationRequest,
    DelegationTask,
    SubAgentArtifactRef,
    SubAgentInputRequest,
    SubAgentResult,
    SubAgentSpec,
)
from deepkeel.subagents.registry import SubAgentRegistry
from deepkeel.subagents.store import SubAgentRunStore
from deepkeel.tools import ToolExecutionContext


_TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
}
_INTERRUPTED_STATES = {
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_AUTH_REQUIRED",
}


@dataclass(frozen=True, slots=True)
class A2ARemoteAgent:
    """One remote A2A endpoint projected as a bounded DeepKeel specialist."""

    agent_id: str
    card: A2AAgentCard
    client: A2AClientPort
    timeout_seconds: int = 90
    poll_interval_seconds: float = 0.5

    def to_subagent_spec(self) -> SubAgentSpec:
        skill_ids = [skill.id for skill in self.card.skills]
        return SubAgentSpec(
            id=self.agent_id,
            version=self.card.version,
            label=self.card.name,
            description=self.card.description,
            domain="a2a",
            model_role="reasoning",
            capabilities=skill_ids,
            input_contract={"type": "object"},
            output_contract={"type": "object"},
            timeout_seconds=max(5, min(int(self.timeout_seconds), 300)),
            metadata={
                "execution_kind": "remote_a2a",
                "protocol_version": "1.0",
                "skills": [
                    skill.model_dump(mode="json", by_alias=True)
                    for skill in self.card.skills
                ],
                "supported_interfaces": [
                    interface.model_dump(mode="json", by_alias=True)
                    for interface in self.card.supported_interfaces
                ],
            },
        )


class A2ADelegationExecutor:
    """Optional A2A executor that preserves DeepKeel parent-run ownership."""

    def __init__(
        self,
        agents: list[A2ARemoteAgent],
        *,
        run_store: SubAgentRunStore | None = None,
        max_parallel: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not agents:
            raise ValueError("at least one A2A remote agent is required")
        if len({agent.agent_id for agent in agents}) != len(agents):
            raise ValueError("A2A remote agent ids must be unique")
        self._agents = {agent.agent_id: agent for agent in agents}
        self.registry = SubAgentRegistry(
            [agent.to_subagent_spec() for agent in agents]
        )
        self.run_store = run_store
        self.max_parallel = max(1, min(int(max_parallel), 3))
        self._sleep = sleep
        self._monotonic = monotonic

    async def aexecute_many(
        self,
        request: DelegationRequest,
        *,
        context: ToolExecutionContext,
        providers: dict[str, Any],
        event_sink: Any = None,
    ) -> DelegationBatchResult:
        if context.session is not None and context.session_factory is None:
            raise RuntimeError(
                "async A2A delegation requires session_factory when a session is bound"
            )
        thread_context = context.fork(session=None) if context.session is not None else context
        return await asyncio.to_thread(
            self.execute_many,
            request,
            context=thread_context,
            providers=providers,
            event_sink=event_sink,
        )

    def execute_many(
        self,
        request: DelegationRequest,
        *,
        context: ToolExecutionContext,
        providers: dict[str, Any],
        event_sink: Any = None,
    ) -> DelegationBatchResult:
        del providers  # The remote A2A agent owns its model and internal tools.
        started_at = self._monotonic()
        self._validate_request(request, context)
        root_run_id = request.root_run_id or context.run_id
        parent_run_id = request.parent_run_id or context.run_id
        if self.run_store is not None and not self.run_store.parent_accepts_results(
            parent_run_id
        ):
            raise RuntimeError("parent agent run is already terminal")
        child_ids = {
            task.id: _child_run_id(parent_run_id, request.delegation_id, task)
            for task in request.tasks
        }
        bound_request = request.model_copy(
            update={
                "root_run_id": root_run_id,
                "parent_run_id": parent_run_id,
                "tasks": [
                    task.bind_lineage(
                        root_run_id=root_run_id,
                        parent_run_id=parent_run_id,
                        delegation_id=request.delegation_id,
                        depth=request.depth,
                        child_run_id=child_ids[task.id],
                    )
                    for task in request.tasks
                ],
            }
        )
        _emit_batch(
            event_sink,
            "subagent.batch.started",
            bound_request,
            status="running",
            duration_ms=0,
        )
        results: list[SubAgentResult] = []
        pending: list[DelegationTask] = []
        for task in bound_request.tasks:
            spec = self.registry.get(task.agent_id)
            child_run_id = child_ids[task.id]
            self._create_child(
                child_run_id,
                bound_request,
                task,
                spec,
                context,
            )
            restored = (
                self.run_store.load_child_result(child_run_id)
                if self.run_store is not None
                else None
            )
            if restored is not None:
                results.append(restored)
                _emit_task(
                    event_sink,
                    "subagent.replayed",
                    bound_request,
                    task,
                    spec,
                    child_run_id,
                    result=restored,
                )
            else:
                pending.append(task)
                _emit_task(
                    event_sink,
                    "subagent.started",
                    bound_request,
                    task,
                    spec,
                    child_run_id,
                )
        if pending:
            workers = min(bound_request.max_concurrency, self.max_parallel, len(pending))
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="deepkeel-a2a",
            ) as pool:
                futures = {
                    pool.submit(
                        self._execute_one,
                        task,
                        request=bound_request,
                        child_run_id=child_ids[task.id],
                        context=context,
                    ): task
                    for task in pending
                }
                results.extend(
                    self._collect(futures, bound_request, context, event_sink)
                )
        ordered = sorted(
            results,
            key=lambda result: next(
                index
                for index, task in enumerate(bound_request.tasks)
                if task.id == result.task_id
            ),
        )
        batch = DelegationBatchResult(
            delegation_id=bound_request.delegation_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            status=_batch_status(ordered),
            results=ordered,
            duration_ms=round((self._monotonic() - started_at) * 1000),
        )
        _emit_batch(
            event_sink,
            "subagent.batch.completed",
            bound_request,
            status=batch.status,
            duration_ms=batch.duration_ms,
        )
        return batch

    def close(self) -> None:
        seen: set[int] = set()
        for remote in self._agents.values():
            identity = id(remote.client)
            if identity in seen:
                continue
            seen.add(identity)
            remote.client.close()

    def _validate_request(
        self,
        request: DelegationRequest,
        context: ToolExecutionContext,
    ) -> None:
        if request.depth > 1:
            raise ValueError("A2A subagents cannot delegate recursively")
        if len(request.tasks) > self.max_parallel:
            raise ValueError(f"A2A task count exceeds limit: {self.max_parallel}")
        missing = sorted(
            {task.agent_id for task in request.tasks if task.agent_id not in self._agents}
        )
        if missing:
            raise ValueError("A2A remote agent is not registered: " + ", ".join(missing))
        snapshot = context.metadata.get("skill_activation")
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        if not isinstance(snapshot.get("delegation_policy"), dict):
            return
        policy = DelegationPolicy.from_snapshot(snapshot)
        if not policy.enabled:
            raise ValueError("delegation is disabled by the active Skill policy")
        if len(request.tasks) > policy.max_tasks:
            raise ValueError(
                f"delegation task count exceeds Skill limit: {policy.max_tasks}"
            )
        if request.max_concurrency > policy.max_concurrency:
            raise ValueError(
                "delegation concurrency exceeds Skill limit: "
                f"{policy.max_concurrency}"
            )
        denied = sorted(
            {task.agent_id for task in request.tasks if not policy.allows_agent(task.agent_id)}
        )
        if denied:
            raise ValueError(
                "delegated agents are not allowed by the active Skill: "
                + ", ".join(denied)
            )

    def _collect(
        self,
        futures: dict[Future[SubAgentResult], DelegationTask],
        request: DelegationRequest,
        context: ToolExecutionContext,
        event_sink: Any,
    ) -> list[SubAgentResult]:
        results: list[SubAgentResult] = []
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except RunCanceledError as exc:
                result = self._error_result(
                    request,
                    task,
                    status="canceled",
                    error=str(exc) or "A2A delegation was canceled",
                )
            except Exception as exc:
                result = self._error_result(
                    request,
                    task,
                    status="failed",
                    error=str(exc),
                )
            if (
                result.status == "completed"
                and task.cancellation.discard_late_result
                and self.run_store is not None
                and not self.run_store.parent_accepts_results(request.parent_run_id)
            ):
                result = self._error_result(
                    request,
                    task,
                    status="canceled",
                    error="parent agent run became terminal",
                )
            self._settle_child(result)
            results.append(result)
            event_type = {
                "completed": "subagent.completed",
                "canceled": "subagent.canceled",
                "needs_input": "subagent.needs_input",
            }.get(result.status, "subagent.failed")
            _emit_task(
                event_sink,
                event_type,
                request,
                task,
                self.registry.get(task.agent_id),
                result.child_run_id,
                result=result,
            )
        return results

    def _execute_one(
        self,
        task: DelegationTask,
        *,
        request: DelegationRequest,
        child_run_id: str,
        context: ToolExecutionContext,
    ) -> SubAgentResult:
        started_at = self._monotonic()
        remote = self._agents[task.agent_id]
        deadline = self._deadline(task, remote, context)
        remote_task: A2ATask | None = None
        checkpoint = self._load_checkpoint(child_run_id, task, remote)
        try:
            self._raise_if_canceled(context, request.parent_run_id, child_run_id, task)
            if checkpoint:
                remote_task = remote.client.get_task(
                    str(checkpoint["remote_task_id"]),
                    timeout_seconds=self._remaining(deadline),
                )
            else:
                response = remote.client.send_message(
                    _task_message(task, request, context),
                    accepted_output_modes=list(remote.card.default_output_modes),
                    timeout_seconds=self._remaining(deadline),
                )
                if response.message is not None:
                    return _message_result(
                        task,
                        child_run_id,
                        response,
                        duration_ms=round((self._monotonic() - started_at) * 1000),
                    )
                remote_task = response.task
            if remote_task is None:
                raise RuntimeError("A2A response did not contain a message or task")
            while remote_task.status.state not in _TERMINAL_STATES | _INTERRUPTED_STATES:
                self._checkpoint_remote(child_run_id, task, remote, remote_task)
                self._raise_if_canceled(
                    context,
                    request.parent_run_id,
                    child_run_id,
                    task,
                )
                remaining = self._remaining(deadline)
                self._sleep(min(remote.poll_interval_seconds, remaining))
                remote_task = remote.client.get_task(
                    remote_task.id,
                    timeout_seconds=self._remaining(deadline),
                )
            self._checkpoint_remote(child_run_id, task, remote, remote_task)
            return _task_result(
                task,
                child_run_id,
                remote_task,
                duration_ms=round((self._monotonic() - started_at) * 1000),
            )
        except (RunCanceledError, TimeoutError):
            if remote_task is not None:
                _best_effort_cancel(remote.client, remote_task.id)
            raise

    def _deadline(
        self,
        task: DelegationTask,
        remote: A2ARemoteAgent,
        context: ToolExecutionContext,
    ) -> float:
        durations = [float(remote.timeout_seconds)]
        if task.timeout_seconds is not None:
            durations.append(float(task.timeout_seconds))
        if task.budget.max_elapsed_seconds is not None:
            durations.append(float(task.budget.max_elapsed_seconds))
        deadline = self._monotonic() + max(0.001, min(durations))
        if context.deadline_monotonic is not None:
            deadline = min(deadline, context.deadline_monotonic)
        return deadline

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise TimeoutError("A2A delegation timed out")
        return remaining

    def _raise_if_canceled(
        self,
        context: ToolExecutionContext,
        parent_run_id: str,
        child_run_id: str,
        task: DelegationTask,
    ) -> None:
        if task.cancellation.propagate_parent:
            context.run_control.raise_if_cancelled(parent_run_id or context.run_id)
        context.run_control.raise_if_cancelled(child_run_id)
        if (
            self.run_store is not None
            and task.cancellation.propagate_parent
            and self.run_store.cancel_requested(child_run_id, parent_run_id)
        ):
            raise RunCanceledError()

    def _create_child(
        self,
        child_run_id: str,
        request: DelegationRequest,
        task: DelegationTask,
        spec: SubAgentSpec,
        context: ToolExecutionContext,
    ) -> None:
        if self.run_store is None:
            return
        self.run_store.create_child(
            child_run_id=child_run_id,
            root_run_id=request.root_run_id,
            parent_run_id=request.parent_run_id,
            delegation_id=request.delegation_id,
            task=task,
            spec=spec,
            user_id=context.user_id,
            thread_id=context.thread_id,
        )

    def _settle_child(self, result: SubAgentResult) -> None:
        if self.run_store is None:
            return
        if result.status != "needs_input":
            self.run_store.settle_child(result)
            return
        suspend = getattr(self.run_store, "suspend_child", None)
        if callable(suspend):
            suspend(result)
            return
        self.run_store.checkpoint_child(
            result.child_run_id,
            phase="needs_input",
            state={
                "schema_version": "deepkeel-a2a-child-v1",
                "task_id": result.task_id,
                "remote_task_id": str(result.metadata.get("remote_task_id") or ""),
                "remote_context_id": str(result.metadata.get("remote_context_id") or ""),
                "state": str(result.metadata.get("remote_state") or ""),
            },
        )

    def _load_checkpoint(
        self,
        child_run_id: str,
        task: DelegationTask,
        remote: A2ARemoteAgent,
    ) -> dict[str, Any]:
        if self.run_store is None:
            return {}
        value = self.run_store.load_child_checkpoint(child_run_id) or {}
        if (
            value.get("schema_version") != "deepkeel-a2a-child-v1"
            or str(value.get("task_id") or "") != task.id
            or str(value.get("agent_id") or task.agent_id) != remote.agent_id
            or not value.get("remote_task_id")
        ):
            return {}
        return dict(value)

    def _checkpoint_remote(
        self,
        child_run_id: str,
        task: DelegationTask,
        remote: A2ARemoteAgent,
        remote_task: A2ATask,
    ) -> None:
        if self.run_store is None:
            return
        self.run_store.checkpoint_child(
            child_run_id,
            phase="remote_a2a",
            state={
                "schema_version": "deepkeel-a2a-child-v1",
                "task_id": task.id,
                "agent_id": remote.agent_id,
                "remote_task_id": remote_task.id,
                "remote_context_id": remote_task.context_id,
                "state": remote_task.status.state,
            },
        )

    def _error_result(
        self,
        request: DelegationRequest,
        task: DelegationTask,
        *,
        status: Literal["failed", "canceled"],
        error: str,
    ) -> SubAgentResult:
        return SubAgentResult(
            task_id=task.id,
            agent_id=task.agent_id,
            child_run_id=_child_run_id(
                request.parent_run_id,
                request.delegation_id,
                task,
            ),
            status=status,
            idempotency_key=task.effective_idempotency_key,
            lineage=task.lineage,
            error=error,
            metadata={"transport": "a2a"},
        )


def _task_message(
    task: DelegationTask,
    request: DelegationRequest,
    context: ToolExecutionContext,
) -> A2AMessage:
    payload = {
        "objective": task.objective,
        "normalized_question": task.normalized_question,
        "input": task.input_data,
        "constraints": task.constraints,
        "context_refs": [item.model_dump(mode="json") for item in task.context_refs],
        "artifact_refs": [item.model_dump(mode="json") for item in task.artifact_refs],
        "expected_output": task.expected_output,
        "budget": task.budget.model_dump(mode="json"),
    }
    message_key = (
        f"{request.parent_run_id}|{request.delegation_id}|"
        f"{task.effective_idempotency_key}|{task.agent_id}"
    )
    message_id = hashlib.sha256(message_key.encode("utf-8")).hexdigest()[:32]
    resume = task.metadata.get("a2a_resume")
    resume = resume if isinstance(resume, dict) else {}
    return A2AMessage(
        messageId=message_id,
        role="ROLE_USER",
        parts=[A2APart(text=json.dumps(payload, ensure_ascii=False, default=str))],
        contextId=str(resume.get("context_id") or context.thread_id or ""),
        taskId=str(resume.get("task_id") or ""),
        metadata={
            "deepkeel": {
                "rootRunId": request.root_run_id,
                "parentRunId": request.parent_run_id,
                "delegationId": request.delegation_id,
                "taskId": task.id,
                "idempotencyKey": task.effective_idempotency_key,
            }
        },
    )


def _message_result(
    task: DelegationTask,
    child_run_id: str,
    response: A2ASendResponse,
    *,
    duration_ms: int,
) -> SubAgentResult:
    assert response.message is not None
    conclusion, structured = _message_content(response.message)
    return SubAgentResult(
        task_id=task.id,
        agent_id=task.agent_id,
        child_run_id=child_run_id,
        status="completed",
        conclusion=conclusion,
        output={"structured_data": structured} if structured else {},
        idempotency_key=task.effective_idempotency_key,
        lineage=task.lineage,
        duration_ms=duration_ms,
        raw_text=conclusion,
        metadata={
            "transport": "a2a",
            "response_kind": "message",
            "remote_context_id": response.message.context_id,
        },
    )


def _task_result(
    task: DelegationTask,
    child_run_id: str,
    remote_task: A2ATask,
    *,
    duration_ms: int,
) -> SubAgentResult:
    state = remote_task.status.state
    status_message, status_data = _message_content(remote_task.status.message)
    artifact_text, artifact_data = _artifact_content(remote_task.artifacts)
    conclusion = artifact_text or status_message
    metadata = {
        "transport": "a2a",
        "response_kind": "task",
        "remote_task_id": remote_task.id,
        "remote_context_id": remote_task.context_id,
        "remote_state": state,
    }
    common = {
        "task_id": task.id,
        "agent_id": task.agent_id,
        "child_run_id": child_run_id,
        "idempotency_key": task.effective_idempotency_key,
        "lineage": task.lineage,
        "duration_ms": duration_ms,
        "metadata": metadata,
    }
    if state in _INTERRUPTED_STATES:
        auth_required = state == "TASK_STATE_AUTH_REQUIRED"
        prompt = conclusion or (
            "Authorization is required to continue the remote agent task."
            if auth_required
            else "Additional input is required to continue the remote agent task."
        )
        return SubAgentResult.model_validate(
            {
                **common,
                "status": "needs_input",
                "outcome": "needs_input",
                "conclusion": prompt,
                "input_request": SubAgentInputRequest(
                    prompt=prompt,
                    requirements=[
                        "authorization" if auth_required else "additional input"
                    ],
                    resume_token=remote_task.id,
                    metadata={
                        "transport": "a2a",
                        "remote_task_id": remote_task.id,
                        "remote_context_id": remote_task.context_id,
                        "auth_required": auth_required,
                    },
                ),
            }
        )
    if state == "TASK_STATE_COMPLETED":
        structured = [*status_data, *artifact_data]
        return SubAgentResult.model_validate(
            {
                **common,
                "status": "completed",
                "conclusion": conclusion,
                "artifact_refs": _artifact_refs(remote_task.artifacts),
                "output": {"structured_data": structured} if structured else {},
                "raw_text": conclusion,
            }
        )
    if state == "TASK_STATE_CANCELED":
        return SubAgentResult.model_validate(
            {
                **common,
                "status": "canceled",
                "error": conclusion or "remote A2A task was canceled",
            }
        )
    return SubAgentResult.model_validate(
        {
            **common,
            "status": "failed",
            "error": conclusion or f"remote A2A task ended in {state}",
            "raw_text": conclusion,
        }
    )


def _message_content(message: A2AMessage | None) -> tuple[str, list[Any]]:
    if message is None:
        return "", []
    text = "\n".join(part.text.strip() for part in message.parts if part.text).strip()
    structured = [part.data for part in message.parts if part.data is not None]
    if not text and structured:
        text = json.dumps(structured[0], ensure_ascii=False, default=str)
    return text, structured


def _artifact_content(artifacts: list[A2AArtifact]) -> tuple[str, list[Any]]:
    text_parts: list[str] = []
    structured: list[Any] = []
    for artifact in artifacts:
        for part in artifact.parts:
            if part.text:
                text_parts.append(part.text.strip())
            if part.data is not None:
                structured.append(part.data)
    return "\n".join(value for value in text_parts if value).strip(), structured


def _artifact_refs(artifacts: list[A2AArtifact]) -> list[SubAgentArtifactRef]:
    refs: list[SubAgentArtifactRef] = []
    for artifact in artifacts:
        urls = [part.url for part in artifact.parts if part.url]
        media_types = [part.media_type for part in artifact.parts if part.media_type]
        refs.append(
            SubAgentArtifactRef(
                id=artifact.artifact_id,
                artifact_type=media_types[0] if media_types else "a2a_artifact",
                uri=urls[0] if urls else "",
                metadata={
                    "name": artifact.name,
                    "description": artifact.description,
                    "transport": "a2a",
                    **artifact.metadata,
                },
            )
        )
    return refs


def _best_effort_cancel(client: A2AClientPort, task_id: str) -> None:
    try:
        client.cancel_task(task_id, timeout_seconds=5)
    except Exception:
        return


def _child_run_id(
    parent_run_id: str,
    delegation_id: str,
    task: DelegationTask,
) -> str:
    identity = (
        f"{parent_run_id}|key:{task.idempotency_key}|{task.agent_id}"
        if task.idempotency_key
        else f"{parent_run_id}|{delegation_id}|{task.id}|{task.agent_id}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"{str(parent_run_id or 'root')[:72]}:sub:{digest}"


def _batch_status(
    results: list[SubAgentResult],
) -> Literal["completed", "partial", "failed", "canceled", "needs_input"]:
    if any(result.status == "needs_input" for result in results):
        return "needs_input"
    if results and all(result.status == "canceled" for result in results):
        return "canceled"
    if results and all(result.status == "completed" for result in results):
        return "completed"
    if any(result.status == "completed" for result in results):
        return "partial"
    return "failed"


def _emit_batch(
    sink: Any,
    event_type: str,
    request: DelegationRequest,
    *,
    status: str,
    duration_ms: int,
) -> None:
    if not callable(sink):
        return
    sink(
        {
            "event_type": event_type,
            "title": "Remote specialist delegation",
            "summary": f"{len(request.tasks)} A2A specialist task(s): {status}",
            "payload": {
                "schema_version": SUBAGENT_EVENT_SCHEMA_VERSION,
                "visible": True,
                "transport": "a2a",
                "delegation_id": request.delegation_id,
                "root_run_id": request.root_run_id,
                "parent_run_id": request.parent_run_id,
                "task_count": len(request.tasks),
                "status": status,
                "duration_ms": duration_ms,
            },
        }
    )


def _emit_task(
    sink: Any,
    event_type: str,
    request: DelegationRequest,
    task: DelegationTask,
    spec: SubAgentSpec,
    child_run_id: str,
    *,
    result: SubAgentResult | None = None,
) -> None:
    if not callable(sink):
        return
    sink(
        {
            "event_type": event_type,
            "title": spec.label,
            "summary": result.conclusion if result and result.conclusion else task.objective,
            "payload": {
                "schema_version": SUBAGENT_EVENT_SCHEMA_VERSION,
                "visible": True,
                "transport": "a2a",
                "delegation_id": request.delegation_id,
                "task_id": task.id,
                "agent_id": task.agent_id,
                "agent_label": spec.label,
                "child_run_id": child_run_id,
                "root_run_id": request.root_run_id,
                "parent_run_id": request.parent_run_id,
                "parent_task_id": task.lineage.parent_task_id,
                "idempotency_key": task.effective_idempotency_key,
                "spec_version": spec.version,
                "status": result.status if result else "running",
                "duration_ms": result.duration_ms if result else 0,
                "artifact_refs": (
                    [item.model_dump(mode="json") for item in result.artifact_refs]
                    if result
                    else []
                ),
                "needs_input": bool(result and result.status == "needs_input"),
                "input_request": (
                    result.input_request.model_dump(mode="json")
                    if result and result.input_request
                    else None
                ),
                "error": result.error if result else "",
            },
        }
    )
