from __future__ import annotations

from pathlib import Path

import pytest

from verification.release_contract import release_tag_for_version, verify_release_version


def test_release_contract_accepts_matching_semantic_version() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert verify_release_version(repo_root, "v4.0.0-rc.1") == "4.0.0rc1"


def test_release_contract_rejects_mismatched_tag() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    with pytest.raises(ValueError, match=r"release tag must be v4\.0\.0-rc\.1"):
        verify_release_version(repo_root, "v3.3.0")


def test_release_tag_normalizes_pep440_release_candidates() -> None:
    assert release_tag_for_version("4.0.0rc2") == "v4.0.0-rc.2"
    assert release_tag_for_version("4.0.0") == "v4.0.0"
