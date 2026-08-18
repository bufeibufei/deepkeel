from __future__ import annotations

import json
import socket
from typing import Any

import httpx
import pytest

from deepkeel.a2a_sdk import (
    A2A_PROTOCOL_VERSION,
    A2AAgentCard,
    A2AAgentInterface,
    A2AAgentSkill,
    A2AArtifact,
    A2ADelegationExecutor,
    A2AMessage,
    A2APart,
    A2ARemoteAgent,
    A2ASendResponse,
    A2ATask,
    A2ATaskStatus,
    HttpJsonA2AClient,
)
from deepkeel.control import InMemoryRunControl
from deepkeel.orchestration_sdk import DelegationRequest, SubAgentResult, TaskBrief
from deepkeel.tools import ToolExecutionContext


def _card() -> A2AAgentCard:
    return A2AAgentCard(
        name="Remote reviewer",
        description="Reviews bounded evidence packages.",
        supportedInterfaces=[
            A2AAgentInterface(
                url="https://agent.example/a2a",
                protocolBinding="HTTP+JSON",
            )
        ],
        version="2.3.0",
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain", "application/json"],
        skills=[
            A2AAgentSkill(
                id="review.evidence",
                name="Evidence review",
                description="Checks a bounded evidence package.",
            )
        ],
    )


def _message(text: str, *, context_id: str = "remote-context") -> A2AMessage:
    return A2AMessage(
        messageId="message-1",
        role="ROLE_AGENT",
        parts=[A2APart(text=text)],
        contextId=context_id,
    )


def _task(
    state: str,
    *,
    task_id: str = "remote-task-1",
    message: str = "",
    artifacts: list[A2AArtifact] | None = None,
) -> A2ATask:
    return A2ATask(
        id=task_id,
        contextId="remote-context",
        status=A2ATaskStatus(
            state=state,
            message=_message(message) if message else None,
        ),
        artifacts=artifacts or [],
    )


class FakeA2AClient:
    def __init__(
        self,
        response: A2ASendResponse,
        *,
        task_updates: list[A2ATask] | None = None,
        on_get: Any = None,
    ) -> None:
        self.response = response
        self.task_updates = list(task_updates or [])
        self.on_get = on_get
        self.sent: list[A2AMessage] = []
        self.cancelled: list[str] = []
        self.closed = False

    def send_message(
        self,
        message: A2AMessage,
        *,
        accepted_output_modes: list[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> A2ASendResponse:
        del accepted_output_modes, timeout_seconds
        self.sent.append(message)
        return self.response

    def get_task(
        self,
        task_id: str,
        *,
        history_length: int = 5,
        timeout_seconds: float | None = None,
    ) -> A2ATask:
        del history_length, timeout_seconds
        if callable(self.on_get):
            self.on_get()
        assert task_id == "remote-task-1"
        return self.task_updates.pop(0)

    def cancel_task(
        self,
        task_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> A2ATask:
        del timeout_seconds
        self.cancelled.append(task_id)
        return _task("TASK_STATE_CANCELED", task_id=task_id)

    def close(self) -> None:
        self.closed = True


class RecordingStore:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.settled: list[SubAgentResult] = []
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.restored: dict[str, SubAgentResult] = {}

    def parent_accepts_results(self, _parent_run_id: str) -> bool:
        return True

    def create_child(self, **payload: Any) -> None:
        self.created.append(payload)

    def settle_child(self, result: SubAgentResult) -> None:
        self.settled.append(result)

    def load_child_result(self, child_run_id: str) -> SubAgentResult | None:
        return self.restored.get(child_run_id)

    def load_child_checkpoint(self, child_run_id: str) -> dict[str, Any] | None:
        return self.checkpoints.get(child_run_id)

    def checkpoint_child(
        self,
        child_run_id: str,
        *,
        phase: str,
        state: dict[str, Any],
    ) -> None:
        self.checkpoints[child_run_id] = {**state, "phase": phase}

    def cancel_requested(self, _child_run_id: str, _parent_run_id: str) -> bool:
        return False


def _request() -> DelegationRequest:
    return DelegationRequest(
        delegation_id="delegation-1",
        tasks=[
            TaskBrief(
                id="task-1",
                agent_id="remote.review",
                objective="Review the evidence.",
                input_data={"claim": "example"},
                idempotency_key="review-once",
            )
        ],
    )


def test_a2a_contracts_validate_parts_and_project_agent_card() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        A2APart(text="text", data={"also": "data"})

    client = FakeA2AClient(A2ASendResponse(message=_message("ok")))
    executor = A2ADelegationExecutor(
        [A2ARemoteAgent(agent_id="remote.review", card=_card(), client=client)]
    )
    spec = executor.registry.get("remote.review")

    assert A2A_PROTOCOL_VERSION == "1.0"
    assert spec.version == "2.3.0"
    assert spec.domain == "a2a"
    assert spec.capabilities == ["review.evidence"]
    assert spec.metadata["execution_kind"] == "remote_a2a"


def test_a2a_direct_message_is_projected_to_bounded_subagent_result() -> None:
    client = FakeA2AClient(A2ASendResponse(message=_message("Evidence is consistent.")))
    events: list[dict[str, Any]] = []
    executor = A2ADelegationExecutor(
        [A2ARemoteAgent(agent_id="remote.review", card=_card(), client=client)]
    )

    batch = executor.execute_many(
        _request(),
        context=ToolExecutionContext(
            run_id="run-1",
            user_id="user-1",
            thread_id="thread-1",
        ),
        providers={},
        event_sink=events.append,
    )

    assert batch.status == "completed"
    assert batch.results[0].conclusion == "Evidence is consistent."
    assert batch.results[0].metadata["transport"] == "a2a"
    assert client.sent[0].context_id == "thread-1"
    task_payload = json.loads(client.sent[0].parts[0].text or "{}")
    assert task_payload["objective"] == "Review the evidence."
    assert [event["event_type"] for event in events] == [
        "subagent.batch.started",
        "subagent.started",
        "subagent.completed",
        "subagent.batch.completed",
    ]


def test_a2a_task_polling_persists_checkpoint_and_projects_artifacts() -> None:
    artifact = A2AArtifact(
        artifactId="artifact-1",
        name="review.json",
        parts=[
            A2APart(data={"verdict": "supported"}),
            A2APart(url="https://agent.example/files/review.json", mediaType="application/json"),
        ],
    )
    client = FakeA2AClient(
        A2ASendResponse(task=_task("TASK_STATE_SUBMITTED")),
        task_updates=[
            _task("TASK_STATE_WORKING"),
            _task("TASK_STATE_COMPLETED", message="Review complete.", artifacts=[artifact]),
        ],
    )
    store = RecordingStore()
    executor = A2ADelegationExecutor(
        [
            A2ARemoteAgent(
                agent_id="remote.review",
                card=_card(),
                client=client,
                poll_interval_seconds=0.001,
            )
        ],
        run_store=store,
        sleep=lambda _seconds: None,
    )

    batch = executor.execute_many(
        _request(),
        context=ToolExecutionContext(run_id="run-1", user_id="user-1"),
        providers={},
    )
    result = batch.results[0]

    assert result.status == "completed"
    assert result.conclusion == "Review complete."
    assert result.output["structured_data"] == [{"verdict": "supported"}]
    assert result.artifact_refs[0].id == "artifact-1"
    assert result.artifact_refs[0].uri.endswith("review.json")
    assert next(iter(store.checkpoints.values()))["remote_task_id"] == "remote-task-1"
    assert store.settled == [result]
    assert store.created[0]["task"].lineage.child_run_id == result.child_run_id


def test_a2a_input_required_maps_to_typed_parent_suspension() -> None:
    client = FakeA2AClient(
        A2ASendResponse(
            task=_task(
                "TASK_STATE_INPUT_REQUIRED",
                message="Which date range should I review?",
            )
        )
    )
    executor = A2ADelegationExecutor(
        [A2ARemoteAgent(agent_id="remote.review", card=_card(), client=client)]
    )

    result = executor.execute_many(
        _request(),
        context=ToolExecutionContext(run_id="run-1", user_id="user-1"),
        providers={},
    ).results[0]

    assert result.status == "needs_input"
    assert result.input_request is not None
    assert result.input_request.resume_token == "remote-task-1"
    assert result.input_request.metadata["transport"] == "a2a"


def test_a2a_parent_cancellation_is_propagated_to_remote_task() -> None:
    control = InMemoryRunControl()

    def cancel_parent() -> None:
        control.cancel("run-1")

    client = FakeA2AClient(
        A2ASendResponse(task=_task("TASK_STATE_WORKING")),
        task_updates=[_task("TASK_STATE_WORKING")],
        on_get=cancel_parent,
    )
    executor = A2ADelegationExecutor(
        [
            A2ARemoteAgent(
                agent_id="remote.review",
                card=_card(),
                client=client,
                poll_interval_seconds=0.001,
            )
        ],
        sleep=lambda _seconds: None,
    )

    batch = executor.execute_many(
        _request(),
        context=ToolExecutionContext(
            run_id="run-1",
            user_id="user-1",
            run_control=control,
        ),
        providers={},
    )

    assert batch.status == "canceled"
    assert batch.results[0].status == "canceled"
    assert client.cancelled == ["remote-task-1"]


def test_http_json_a2a_client_uses_v1_endpoints_and_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/message:send"):
            return httpx.Response(
                200,
                json={
                    "message": _message("ok").model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    )
                },
            )
        state = (
            "TASK_STATE_CANCELED"
            if request.url.path.endswith(":cancel")
            else "TASK_STATE_COMPLETED"
        )
        return httpx.Response(
            200,
            json={
                "task": _task(state).model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )
            },
        )

    transport = httpx.MockTransport(handler)
    raw_client = httpx.Client(transport=transport)
    client = HttpJsonA2AClient(
        "https://agent.example/a2a",
        client=raw_client,
    )

    client.send_message(
        A2AMessage(
            messageId="request-1",
            role="ROLE_USER",
            parts=[A2APart(text="hello")],
        )
    )
    client.get_task("remote-task-1")
    client.cancel_task("remote-task-1")

    assert [request.url.path for request in requests] == [
        "/a2a/message:send",
        "/a2a/tasks/remote-task-1",
        "/a2a/tasks/remote-task-1:cancel",
    ]
    assert all(request.headers["a2a-version"] == "1.0" for request in requests)
    assert all(
        request.headers["content-type"] == "application/a2a+json"
        for request in requests
    )
