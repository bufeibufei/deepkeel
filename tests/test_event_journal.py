from __future__ import annotations

import pytest

from deepkeel.composition import HarnessRuntimeBuilder, RuntimePorts
from deepkeel.event_journal import (
    EventJournalConflict,
    InMemoryRuntimeEventJournal,
)
from deepkeel.events import (
    AgentEventPersistenceError,
    envelope_runtime_event,
    normalize_runtime_event,
)
from deepkeel.runtime_api import RuntimeEventEnvelope
from deepkeel.runtime_sdk import RuntimeRequest
from deepkeel.scope import RuntimeScope


class ScriptedProvider:
    model = "scripted-model"
    model_role = "reasoning"

    def complete_chat(self, _messages, **_kwargs):
        return {
            "message": {"role": "assistant", "content": "A durable answer."},
            "finish_reason": "stop",
            "model": self.model,
        }


def _event(
    *,
    sequence: int,
    event_type: str = "run.created",
    scope: RuntimeScope | None = None,
) -> RuntimeEventEnvelope:
    return RuntimeEventEnvelope.model_validate(
        envelope_runtime_event(
            {
                "event_type": event_type,
                "payload": {"value": sequence},
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            run_id="run-1",
            thread_id="thread-1",
            turn_id="turn-1",
            sequence=sequence,
            run_version=3,
            scope=scope,
        )
    )


def test_event_envelope_has_stable_identity_and_cursor():
    first = _event(sequence=1)
    replay = _event(sequence=1)

    assert first == replay
    assert first.schema_version == "harness-runtime-event-v1"
    assert first.cursor == "run-1:1"
    assert first.run_version == 3
    assert first.source_event_type == "run.created"


def test_persisted_event_wrapper_is_flattened_for_host_replay():
    normalized = normalize_runtime_event(
        {
            "id": "journal-1",
            "run_id": "run-1",
            "sequence": 4,
            "payload": {
                "event_type": "tool.started",
                "payload": {
                    "tool_name": "demo.lookup",
                    "tool_call": {"id": "call-1"},
                },
            },
        }
    )

    assert normalized["event_type"] == "tool.call.started"
    assert normalized["source_event_type"] == "tool.started"
    assert normalized["run_id"] == "run-1"
    assert normalized["sequence"] == 4
    assert normalized["payload"]["tool_name"] == "demo.lookup"
    assert "event_type" not in normalized["payload"]


def test_in_memory_journal_is_idempotent_and_supports_cursor_replay():
    journal = InMemoryRuntimeEventJournal()

    journal.append(_event(sequence=1))
    journal.append(_event(sequence=3, event_type="agent.reasoning"))
    duplicate = journal.append(_event(sequence=3, event_type="agent.reasoning"))

    assert duplicate.sequence == 3
    assert journal.latest_sequence("run-1") == 3
    assert [event.sequence for event in journal.read_after("run-1", after_sequence=1)] == [3]


def test_in_memory_journal_rejects_identity_and_sequence_conflicts():
    journal = InMemoryRuntimeEventJournal()
    original = _event(sequence=2)
    journal.append(original)

    conflicting_id = original.model_copy(update={"summary": "changed"})
    with pytest.raises(EventJournalConflict, match="event_id"):
        journal.append(conflicting_id)

    with pytest.raises(EventJournalConflict, match="must increase"):
        journal.append(_event(sequence=1, event_type="agent.reasoning"))


def test_event_journal_isolates_identical_run_ids_by_runtime_scope() -> None:
    journal = InMemoryRuntimeEventJournal()
    first_scope = RuntimeScope(tenant_id="tenant-a", user_id="user-1")
    second_scope = RuntimeScope(tenant_id="tenant-b", user_id="user-1")

    first = _event(sequence=1, scope=first_scope)
    second = _event(sequence=1, scope=second_scope)
    journal.append_scoped(first, scope=first_scope)
    journal.append_scoped(second, scope=second_scope)

    assert first.event_id != second.event_id
    assert journal.latest_sequence_scoped("run-1", scope=first_scope) == 1
    assert journal.latest_sequence_scoped("run-1", scope=second_scope) == 1
    assert journal.read_after_scoped("run-1", scope=first_scope) == (first,)
    assert journal.read_after_scoped("run-1", scope=second_scope) == (second,)
    with pytest.raises(EventJournalConflict, match="scope is required"):
        journal.read_after("run-1")


def test_runtime_replays_events_with_explicit_runtime_scope() -> None:
    journal = InMemoryRuntimeEventJournal()
    runtime = HarnessRuntimeBuilder().with_ports(RuntimePorts(event_journal=journal)).build()
    scope = RuntimeScope(tenant_id="tenant-a", user_id="user-a")

    runtime.run(
        RuntimeRequest(question="hello", run_id="shared", scope=scope),
        provider=ScriptedProvider(),
    )

    assert runtime.replay_events("shared", scope=scope)
    assert runtime.replay_events("shared")


def test_runtime_journals_events_before_publishing_and_replays_from_cursor():
    journal = InMemoryRuntimeEventJournal()
    runtime = HarnessRuntimeBuilder().with_ports(RuntimePorts(event_journal=journal)).build()
    published: list[dict[str, object]] = []

    result = runtime.run(
        RuntimeRequest(
            question="hello",
            run_id="run-runtime",
            thread_id="thread-runtime",
            turn_id="turn-runtime",
        ),
        provider=ScriptedProvider(),
        event_sink=published.append,
    )

    persisted = runtime.replay_events("run-runtime")
    assert result.final_answer.markdown == "A durable answer."
    assert persisted
    assert all(event.run_id == "run-runtime" for event in persisted)
    assert all(event.event_id for event in persisted)
    assert [event.sequence for event in persisted] == sorted(event.sequence for event in persisted)
    published_ids = {item["event_id"] for item in published}
    for event in persisted:
        if event.event_type == "runtime.checkpoint.cleanup.recorded":
            assert event.visibility == "internal"
            assert event.event_id not in published_ids
            continue
        assert event.event_id in published_ids

    cursor = persisted[0].sequence
    assert all(
        event.sequence > cursor
        for event in runtime.replay_events("run-runtime", after_sequence=cursor)
    )


def test_runtime_fails_closed_before_publishing_when_journal_append_fails() -> None:
    class FailingJournal:
        def append(self, _event):
            raise ConnectionError("journal unavailable")

        def latest_sequence(self, _run_id):
            return 0

        def read_after(self, _run_id, *, after_sequence=0, limit=100):
            del after_sequence, limit
            return ()

    published: list[dict[str, object]] = []
    runtime = (
        HarnessRuntimeBuilder().with_ports(RuntimePorts(event_journal=FailingJournal())).build()
    )

    with pytest.raises(AgentEventPersistenceError, match="journal append failed"):
        runtime.run(
            RuntimeRequest(question="hello", run_id="journal-failure"),
            provider=ScriptedProvider(),
            event_sink=published.append,
        )

    assert published == []


def test_runtime_normalizes_missing_journal_sequence_to_zero() -> None:
    class MissingSequenceJournal(InMemoryRuntimeEventJournal):
        def latest_sequence(self, _run_id):
            return None

    journal = MissingSequenceJournal()
    runtime = HarnessRuntimeBuilder().with_ports(RuntimePorts(event_journal=journal)).build()

    result = runtime.run(
        RuntimeRequest(
            question="hello",
            run_id="missing-journal-sequence",
        ),
        provider=ScriptedProvider(),
    )

    assert result.events
    assert result.events[0].sequence == 1
