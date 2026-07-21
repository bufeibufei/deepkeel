from __future__ import annotations

from pathlib import Path

import pytest

from verification.release_contract import verify_release_version


def test_release_contract_accepts_matching_semantic_version() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert verify_release_version(repo_root, "v3.11.0") == "3.11.0"


def test_release_contract_rejects_mismatched_tag() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    with pytest.raises(ValueError, match="release tag must be v3.11.0"):
        verify_release_version(repo_root, "v3.3.0")
