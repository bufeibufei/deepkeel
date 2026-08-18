from __future__ import annotations

from deepkeel.context_contracts import ContextItem
from deepkeel.context_quality import ContextQualityGate, ContextQualityPolicy
from deepkeel.context_window import DeterministicContextWindowManager
from deepkeel.context_window_contracts import ContextSegment


def test_context_quality_audit_reports_but_does_not_quarantine() -> None:
    report = ContextQualityGate().evaluate(
        [
            ContextItem(
                key="identity",
                value={"name": "Ada"},
                tier="L1",
                source="",
                source_ref="",
            )
        ]
    )

    assert report.valid is False
    assert report.quarantined_keys == ()
    assert report.issues[0].code == "l1_source_missing"


def test_context_quality_enforcement_quarantines_invalid_l1_before_model_input() -> None:
    manager = DeterministicContextWindowManager(
        quality_gate=ContextQualityGate(ContextQualityPolicy(mode="enforce"))
    )

    result = manager.prepare(
        "question",
        {},
        {
            "runtime_context": {},
            "context_segments": [
                ContextSegment(
                    key="identity",
                    value={"name": "Ada"},
                    tier="L1",
                    source="",
                    source_ref="",
                )
            ],
        },
    )

    assert "identity" not in result.context_bundle["runtime_context"]
    assert result.context_bundle["quarantined_context"]["identity"] == {"name": "Ada"}
    assert result.diagnostics["quality"]["quarantined_keys"] == ["identity"]


def test_valid_source_linked_l1_and_l2_context_passes_enforcement() -> None:
    gate = ContextQualityGate(ContextQualityPolicy(mode="enforce"))

    report = gate.evaluate(
        [
            ContextItem(
                key="identity",
                value={"name": "Ada"},
                tier="L1",
                source="profile-store",
                source_ref="profile:1",
                subject_id="user-1",
            ),
            ContextItem(
                key="checkpoint",
                value={"goal": "finish"},
                tier="L2",
                authority="derived",
                source="event-journal",
                source_ref="events:1-10",
                source_fingerprint="sha256:test",
                subject_id="user-1",
            ),
        ],
        active_subject_id="user-1",
    )

    assert report.valid is True
    assert report.quarantined_keys == ()
