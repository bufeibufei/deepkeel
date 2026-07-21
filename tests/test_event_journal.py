from __future__ import annotations

import pytest

from harness_core.composition import HarnessRuntimeBuilder, RuntimePorts
from harness_core.event_journal import (
    EventJournalConflict,
    InMemoryRuntimeEventJournal,
)
from harness_core.events import envelope_runtime_event, normalize_runtime_event
from harness_core.runtime_api import RuntimeEventEnvelope
from harness_core.runtime_sdk import RuntimeRequest


class ScriptedProvider:
    model = "scripted-model"
    model_role = "reasoning"

    def complete_chat(self, _messages, **_kwargs):
        return {
            "message": {"role": "assistant", "content": "A durable answer."},
            "finish_reason": "stop",
            "model": self.model,
        }


def _event(*, sequence: int, event_type: str = "run.created") -> RuntimeEventEnvelope:
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


def test_runtime_journals_events_before_publishing_and_replays_from_cursor():
    journal = InMemoryRuntimeEventJournal()
    runtime = HarnessRuntimeBuilder().with_ports(
        RuntimePorts(event_journal=journal)
    ).build()
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
    assert [event.sequence for event in persisted] == sorted(
        event.sequence for event in persisted
    )
    assert all(event.event_id in {item["event_id"] for item in published} for event in persisted)

    cursor = persisted[0].sequence
    assert all(
        event.sequence > cursor
        for event in runtime.replay_events("run-runtime", after_sequence=cursor)
    )
