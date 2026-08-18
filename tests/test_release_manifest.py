from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from verification.release_manifest import build_release_manifest


def test_release_manifest_is_sorted_and_excludes_itself(tmp_path: Path) -> None:
    wheel = tmp_path / "deepkeel-4.1.0-py3-none-any.whl"
    source = tmp_path / "deepkeel-4.1.0.tar.gz"
    source.write_bytes(b"source")
    wheel.write_bytes(b"wheel")

    manifest = build_release_manifest(tmp_path)

    assert manifest.read_text(encoding="ascii").splitlines() == [
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}",
        f"{hashlib.sha256(source.read_bytes()).hexdigest()}  {source.name}",
    ]
    assert build_release_manifest(tmp_path).read_bytes() == manifest.read_bytes()


def test_release_manifest_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has no artifacts"):
        build_release_manifest(tmp_path)
