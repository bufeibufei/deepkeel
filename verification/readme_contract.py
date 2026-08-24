from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

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

README_SECTION_MARKERS = {
    "README.md": (
        "## Why DeepKeel",
        "## 60-second quickstart",
        "## Architecture",
        "## Core capabilities",
        "## Proven in a real Host",
        "## Verification and release evidence",
        "## Documentation",
    ),
    "README.zh-CN.md": (
        "## 为什么需要 DeepKeel",
        "## 60 秒快速开始",
        "## 整体架构",
        "## 核心能力",
        "## 真实 Host 验证",
        "## 验证与发布证据",
        "## 文档导航",
    ),
}

SUPPORTING_DOCUMENTS = (
    "docs/case-study-kuitianjiandi.md",
    "docs/design-decisions.md",
    "docs/verification-matrix.md",
)

MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


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
        for heading in README_SECTION_MARKERS[name]:
            if heading not in content:
                errors.append(f"{name} is missing section {heading!r}")
        errors.extend(_invalid_local_links(root, root / name, content))
    if "[简体中文](README.zh-CN.md)" not in documents["README.md"]:
        errors.append("README.md is missing the Simplified Chinese language link")
    if "[English](README.md)" not in documents["README.zh-CN.md"]:
        errors.append("README.zh-CN.md is missing the English language link")
    if version not in documents["README.md"]:
        errors.append(f"README.md is missing package version {version}")
    if version not in documents["README.zh-CN.md"]:
        errors.append(f"README.zh-CN.md is missing package version {version}")
    for relative_path in SUPPORTING_DOCUMENTS:
        document = root / relative_path
        if not document.is_file():
            errors.append(f"missing supporting document: {relative_path}")
            continue
        errors.extend(
            _invalid_local_links(
                root,
                document,
                document.read_text(encoding="utf-8"),
            )
        )
    if errors:
        raise ValueError("README contract failed:\n- " + "\n- ".join(errors))


def _invalid_local_links(root: Path, document: Path, content: str) -> list[str]:
    errors: list[str] = []
    for raw_target in MARKDOWN_LINK_PATTERN.findall(content):
        target = raw_target.strip().strip("<>")
        if (
            not target
            or target.startswith("#")
            or "://" in target
            or target.startswith("mailto:")
        ):
            continue
        path_part = unquote(target.split("#", 1)[0])
        if not path_part:
            continue
        resolved = (document.parent / path_part).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            errors.append(f"{document.name} link escapes repository: {raw_target!r}")
            continue
        if not resolved.exists():
            errors.append(f"{document.name} has missing local link: {raw_target!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify synchronized DeepKeel READMEs.")
    parser.add_argument("repo_root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    verify_readme_contract(args.repo_root)
    print("DeepKeel README contract verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
