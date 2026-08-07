from __future__ import annotations

import argparse
from pathlib import Path
import re
import tomllib

from deepkeel.version import DEEPKEEL_VERSION


def release_tag_for_version(version: str) -> str:
    match = re.fullmatch(r"(\d+\.\d+\.\d+)rc(\d+)", version)
    if match:
        return f"v{match.group(1)}-rc.{match.group(2)}"
    return f"v{version}"


def verify_release_version(repo_root: Path, tag: str) -> str:
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    package_version = str(project["project"]["version"])
    normalized_tag = str(tag or "").strip()

    if package_version != DEEPKEEL_VERSION:
        raise ValueError(
            "package metadata and runtime version differ: "
            f"{package_version} != {DEEPKEEL_VERSION}"
        )
    expected_tag = release_tag_for_version(package_version)
    if normalized_tag != expected_tag:
        raise ValueError(f"release tag must be {expected_tag}, got {normalized_tag or '<blank>'}")
    return package_version


def verify_release_workflow(repo_root: Path) -> None:
    workflow_path = repo_root / ".github" / "workflows" / "release.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    required_fragments = (
        'gh release view "$RELEASE_TAG"',
        'gh release upload "$RELEASE_TAG" dist/*',
        "--clobber",
        'gh release create "$RELEASE_TAG" dist/*',
        "needs: production-gate",
        "verification.postgres_multiworker",
        "verification/concurrency_benchmark.py",
        "skip-existing: true",
    )
    missing = [fragment for fragment in required_fragments if fragment not in workflow]
    if missing:
        raise ValueError(
            "release workflow must create or idempotently update a GitHub release; "
            f"missing: {', '.join(missing)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Core release version contract.")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    version = verify_release_version(repo_root, args.tag)
    verify_release_workflow(repo_root)
    print(f"release contract verified for {release_tag_for_version(version)}")


if __name__ == "__main__":
    main()
