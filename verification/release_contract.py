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
        "anchore/sbom-action@e22c389904149dbc22b58101806040fa8d37a610",
        "verification.release_manifest",
        "twine check dist/*",
        "skip-existing: true",
    )
    missing = [fragment for fragment in required_fragments if fragment not in workflow]
    if missing:
        raise ValueError(
            "release workflow must create or idempotently update a GitHub release; "
            f"missing: {', '.join(missing)}"
        )
    verify_workflow_action_pins(repo_root)


def verify_workflow_action_pins(repo_root: Path) -> None:
    """Reject mutable third-party action refs in every release-relevant workflow."""

    errors: list[str] = []
    for path in sorted((repo_root / ".github" / "workflows").glob("*.yml")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = re.search(r"\buses:\s*([^\s#]+)", line)
            if match is None:
                continue
            action = match.group(1)
            if action.startswith("./") or action.startswith("docker://"):
                continue
            reference = action.rpartition("@")[2]
            if not re.fullmatch(r"[0-9a-f]{40}", reference):
                errors.append(f"{path.name}:{line_number} uses mutable ref {action}")
    if errors:
        raise ValueError("GitHub Actions must be pinned to immutable SHAs:\n- " + "\n- ".join(errors))


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
