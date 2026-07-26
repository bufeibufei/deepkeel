from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from harness_core.contracts import Artifact
from harness_core.skill_packages import ArtifactPresentationSpec


class ArtifactViewField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    value: str
    format: str = "text"


class ArtifactViewAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    running_label: str = ""
    target_view: str
    target_id: str


class ArtifactView(BaseModel):
    """Host-ready projection generated from a portable artifact declaration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "artifact-view-v1"
    artifact_id: str
    artifact_type: str
    source_id: str = ""
    kind: str = "generic"
    label: str
    title: str
    summary: str = ""
    status: str = ""
    fields: list[ArtifactViewField] = Field(default_factory=list)
    action: ArtifactViewAction | None = None


def project_artifact_view(
    artifact: Artifact | dict[str, Any],
    presentation: ArtifactPresentationSpec | dict[str, Any],
) -> ArtifactView:
    typed_artifact = (
        artifact if isinstance(artifact, Artifact) else Artifact.model_validate(artifact)
    )
    spec = (
        presentation
        if isinstance(presentation, ArtifactPresentationSpec)
        else ArtifactPresentationSpec.model_validate(presentation)
    )
    if typed_artifact.artifact_type != spec.artifact_type:
        raise ValueError("artifact presentation type must match artifact type")

    payload = typed_artifact.model_dump(mode="json")
    fields = [
        ArtifactViewField(
            label=field.label,
            value=_format_value(
                _first_path_value(payload, field.paths),
                field.format,
            ),
            format=field.format,
        )
        for field in spec.fields
    ]
    fields = [field for field in fields if field.value]
    target_id = _format_value(
        _first_path_value(payload, spec.action.target_id_paths),
        "text",
    )
    summary = _format_value(
        _first_path_value(payload, spec.summary_paths),
        "text",
    ) or typed_artifact.summary
    status = _format_value(
        _first_path_value(
            payload,
            ["data.status", "data.report_status", "metadata.status"],
        ),
        "text",
    )
    return ArtifactView(
        artifact_id=typed_artifact.id,
        artifact_type=typed_artifact.artifact_type,
        source_id=typed_artifact.source_id,
        kind=spec.kind,
        label=spec.label,
        title=typed_artifact.title or spec.label,
        summary=summary,
        status=status,
        fields=fields,
        action=ArtifactViewAction(
            label=spec.action.label,
            running_label=spec.action.running_label,
            target_view=spec.action.target_view,
            target_id=target_id or typed_artifact.source_id or typed_artifact.id,
        ),
    )


def project_artifact_views(
    artifacts: list[Artifact],
    presentation: ArtifactPresentationSpec | dict[str, Any] | None,
) -> list[ArtifactView]:
    if presentation is None:
        return []
    spec = (
        presentation
        if isinstance(presentation, ArtifactPresentationSpec)
        else ArtifactPresentationSpec.model_validate(presentation)
    )
    return [
        project_artifact_view(artifact, spec)
        for artifact in artifacts
        if artifact.artifact_type == spec.artifact_type
    ]


def _first_path_value(payload: dict[str, Any], paths: list[str]) -> Any:
    for path in paths:
        current: Any = payload
        for segment in path.split("."):
            if not isinstance(current, dict) or segment not in current:
                current = None
                break
            current = current[segment]
        if current not in (None, "", [], {}):
            return current
    return None


def _format_value(value: Any, format_name: str) -> str:
    if value in (None, "", [], {}):
        return ""
    if format_name == "count":
        try:
            return str(len(value))
        except TypeError:
            return "1"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return "、".join(str(item) for item in value)
    return str(value)
