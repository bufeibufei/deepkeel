from __future__ import annotations

from pathlib import Path

import pytest

from verification.release_contract import (
    release_tag_for_version,
    verify_release_version,
    verify_release_workflow,
    verify_workflow_action_pins,
)


def test_release_contract_accepts_matching_semantic_version() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert verify_release_version(repo_root, "v4.1.0") == "4.1.0"


def test_release_contract_rejects_mismatched_tag() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    with pytest.raises(ValueError, match=r"release tag must be v4\.1\.0"):
        verify_release_version(repo_root, "v3.3.0")


def test_release_tag_normalizes_pep440_release_candidates() -> None:
    assert release_tag_for_version("4.0.0rc2") == "v4.0.0-rc.2"
    assert release_tag_for_version("4.0.0") == "v4.0.0"


def test_release_workflow_is_idempotent_for_existing_releases() -> None:
    verify_release_workflow(Path(__file__).resolve().parents[1])


def test_all_workflow_actions_are_pinned_to_immutable_commits() -> None:
    verify_workflow_action_pins(Path(__file__).resolve().parents[1])


def test_release_workflow_rejects_unconditional_creation(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "release.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        'run: gh release create "$RELEASE_TAG" dist/*\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="idempotently update"):
        verify_release_workflow(tmp_path)


def test_workflow_pin_contract_rejects_moving_tags(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("steps:\n  - uses: actions/checkout@v4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mutable ref"):
        verify_workflow_action_pins(tmp_path)
