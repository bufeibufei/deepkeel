from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


MANIFEST_NAME = "SHA256SUMS"


def build_release_manifest(dist: Path) -> Path:
    root = dist.resolve()
    if not root.is_dir():
        raise ValueError(f"distribution directory does not exist: {root}")

    artifacts = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name != MANIFEST_NAME
    )
    if not artifacts:
        raise ValueError(f"distribution directory has no artifacts: {root}")

    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in artifacts
    ]
    manifest = root / MANIFEST_NAME
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a deterministic SHA-256 manifest for release artifacts."
    )
    parser.add_argument("dist", type=Path)
    args = parser.parse_args()
    manifest = build_release_manifest(args.dist)
    print(f"wrote release manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
