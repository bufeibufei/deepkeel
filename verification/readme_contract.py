from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

try:
    from verification.release_contract import release_tag_for_version
except ModuleNotFoundError:
    from release_contract import release_tag_for_version


SHARED_CONTRACT_MARKERS = (
    "python examples/quickstart/main.py",
    "HarnessRuntimeBuilder",
    "RuntimePorts",
    "CapabilityPackSpec",
    "build_production()",
    "arun()",
    "astream()",
    "docs/architecture.md",
    "docs/postgresql-reference.md",
    "docs/production-readiness.md",
    "docs/catalog-discovery.md",
    "docs/releasing.md",
    "deepkeel.runtime_sdk",
    "deepkeel.extension_sdk",
    "deepkeel.discovery_sdk",
    "deepkeel.adapter_sdk",
    "deepkeel.a2a_sdk",
    "docs/security-and-trust.md",
    "docs/interoperability.md",
    "docs/supply-chain.md",
)


def verify_readme_contract(repo_root: Path) -> None:
    root = repo_root.resolve()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = str(project["project"]["version"])
    release_tag = release_tag_for_version(version)
    install_command = (
        "pip install \"deepkeel @ "
        f"git+https://github.com/bufeibufei/deepkeel.git@{release_tag}\""
    )
    documents = {
        "README.md": (root / "README.md").read_text(encoding="utf-8"),
        "README.zh-CN.md": (root / "README.zh-CN.md").read_text(encoding="utf-8"),
    }
    errors: list[str] = []
    for name, content in documents.items():
        for marker in (install_command, *SHARED_CONTRACT_MARKERS):
            if marker not in content:
                errors.append(f"{name} is missing {marker!r}")
    if "[简体中文](README.zh-CN.md)" not in documents["README.md"]:
        errors.append("README.md is missing the Simplified Chinese language link")
    if "[English](README.md)" not in documents["README.zh-CN.md"]:
        errors.append("README.zh-CN.md is missing the English language link")
    if version not in documents["README.md"]:
        errors.append(f"README.md is missing package version {version}")
    if version not in documents["README.zh-CN.md"]:
        errors.append(f"README.zh-CN.md is missing package version {version}")
    if errors:
        raise ValueError("README contract failed:\n- " + "\n- ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify synchronized DeepKeel READMEs.")
    parser.add_argument("repo_root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    verify_readme_contract(args.repo_root)
    print("DeepKeel README contract verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
