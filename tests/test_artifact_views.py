import pytest

from harness_core.runtime_sdk import Artifact, project_artifact_view


PRESENTATION = {
    "schema_version": "artifact-presentation-v1",
    "artifact_type": "calendar.candidates",
    "kind": "report",
    "label": "择日方案",
    "summary_paths": ["data.conclusion", "summary"],
    "fields": [
        {"label": "候选日期", "paths": ["data.candidates"], "format": "count"},
        {"label": "时间范围", "paths": ["data.range"], "format": "text"},
    ],
    "action": {
        "label": "查看方案",
        "running_label": "查看进度",
        "target_view": "date-selection-result",
        "target_id_paths": ["data.plan_id", "source_id"],
    },
}


def test_artifact_view_projects_portable_host_surface() -> None:
    artifact = Artifact(
        id="artifact-1",
        run_id="run-1",
        artifact_type="calendar.candidates",
        title="搬家择日",
        summary="已生成择日候选",
        source_id="plan-1",
        data={
            "status": "completed",
            "conclusion": "首选八月初八",
            "range": "2026-08",
            "candidates": [{"date": "2026-08-08"}, {"date": "2026-08-18"}],
        },
    )

    view = project_artifact_view(artifact, PRESENTATION)

    assert view.artifact_id == "artifact-1"
    assert view.summary == "首选八月初八"
    assert view.status == "completed"
    assert [(field.label, field.value) for field in view.fields] == [
        ("候选日期", "2"),
        ("时间范围", "2026-08"),
    ]
    assert view.action is not None
    assert view.action.target_id == "plan-1"
    assert view.action.target_view == "date-selection-result"


def test_artifact_view_rejects_presentation_for_another_type() -> None:
    artifact = Artifact(
        id="artifact-1",
        run_id="run-1",
        artifact_type="other",
    )

    with pytest.raises(ValueError, match="must match"):
        project_artifact_view(artifact, PRESENTATION)
