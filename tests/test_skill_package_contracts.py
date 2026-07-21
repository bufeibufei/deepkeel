from __future__ import annotations

import json

import pytest

from harness_core.extension_sdk import (
    ArtifactPresentationSpec,
    CompiledSkillSpec,
    SkillPackageManifest,
    load_skill_packages,
)


def _manifest_payload() -> dict:
    return {
        "package_id": "example.report-skill",
        "capability_pack": "example.reporting",
        "entry_tool": "report.build",
        "required_tools": ["report.build"],
        "artifact_types": ["report"],
        "skill_spec": {
            "id": "report-skill",
            "version": "1.0.0",
            "kind": "workflow",
            "label": "Report",
            "description": "Build a report artifact.",
            "icon_key": "report",
            "allowed_tools": ["report.build"],
            "required_tools": ["report.build"],
            "output_contract": {
                "requires_artifact": "report",
                "artifact_presentation": {
                    "schema_version": "artifact-presentation-v1",
                    "artifact_type": "report",
                    "kind": "report",
                    "label": "Report",
                    "summary_paths": ["payload.summary", "summary"],
                    "fields": [
                        {"label": "Sections", "paths": ["payload.sections"], "format": "count"}
                    ],
                    "action": {
                        "label": "View report",
                        "running_label": "View progress",
                        "target_view": "report",
                        "target_id_paths": ["run_id"],
                    },
                },
            },
            "completion_policy": {
                "required_transition": "report.build",
                "waiting_statuses": ["waiting_user_input"],
                "allow_model_clarification": True,
            },
            "delegation_policy": {
                "enabled": True,
                "allowed_agents": ["report.reviewer"],
                "max_tasks": 1,
                "max_concurrency": 1,
            },
        },
    }


def test_skill_package_normalizes_runtime_contract_and_digest() -> None:
    manifest = SkillPackageManifest.model_validate(_manifest_payload())
    compiled = manifest.compile()
    runtime = manifest.runtime_spec()

    assert isinstance(compiled, CompiledSkillSpec)
    assert compiled.id == "report-skill"
    assert compiled.package["digest"] == manifest.digest
    assert manifest.entry_tools == ["report.build"]
    assert manifest.resume_compatible_versions == ["1.0.0"]
    assert runtime["package"]["digest"] == manifest.digest
    assert runtime["package"]["entry_tools"] == ["report.build"]
    presentation = ArtifactPresentationSpec.model_validate(
        compiled.output_contract["artifact_presentation"]
    )
    assert presentation.fields[0].format == "count"


def test_skill_package_rejects_artifact_presentation_type_drift() -> None:
    payload = _manifest_payload()
    payload["skill_spec"]["output_contract"]["artifact_presentation"]["artifact_type"] = "other"

    with pytest.raises(ValueError, match="must match requires_artifact"):
        SkillPackageManifest.model_validate(payload)


def test_skill_package_rejects_unknown_runtime_fields() -> None:
    payload = _manifest_payload()
    payload["skill_spec"]["host_internal_flag"] = True

    with pytest.raises(ValueError, match="host_internal_flag"):
        SkillPackageManifest.model_validate(payload)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda payload: payload["skill_spec"].update({"required_tools": ["report.hidden"]}),
            "required_tools must also be allowed",
        ),
        (
            lambda payload: payload["skill_spec"]["completion_policy"].update(
                {"required_transition": "report.hidden"}
            ),
            "transitions must be allowed",
        ),
        (
            lambda payload: payload["skill_spec"]["completion_policy"].update(
                {"clarification_strategy": "tool_contract"}
            ),
            "tool_contract clarification",
        ),
        (
            lambda payload: payload["skill_spec"].update(
                {
                    "delegation_policy": {
                        "enabled": True,
                        "allowed_agents": [],
                    }
                }
            ),
            "must declare allowed_agents",
        ),
    ],
)
def test_skill_package_rejects_contract_drift(mutate, message: str) -> None:
    payload = _manifest_payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        SkillPackageManifest.model_validate(payload)


def test_skill_package_loader_orders_and_rejects_duplicates(tmp_path) -> None:
    later = _manifest_payload()
    later["order"] = 20
    earlier = _manifest_payload()
    earlier["package_id"] = "example.earlier"
    earlier["order"] = 10
    earlier["skill_spec"] = {
        **earlier["skill_spec"],
        "id": "earlier-skill",
    }
    (tmp_path / "later.json").write_text(json.dumps(later), encoding="utf-8")
    (tmp_path / "earlier.json").write_text(json.dumps(earlier), encoding="utf-8")

    manifests = load_skill_packages(tmp_path)
    assert [item.skill_id for item in manifests] == ["earlier-skill", "report-skill"]

    duplicate = _manifest_payload()
    duplicate["package_id"] = "example.duplicate"
    (tmp_path / "duplicate.json").write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate skill package manifest"):
        load_skill_packages(tmp_path)


def test_missing_skill_package_directory_is_empty(tmp_path) -> None:
    assert load_skill_packages(tmp_path / "missing") == []
