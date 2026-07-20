from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path


PACKAGE_IMPORT_ROOT = "harness_core"
FORBIDDEN_ROOTS = {
    "app",
    "catalog",
    "corpus",
    "frontend",
    "prompts",
    "rag",
    "scripts",
    "web",
}


def _normalized_member(name: str, *, sdist: bool) -> str:
    normalized = name.replace("\\", "/").lstrip("./")
    if sdist and "/" in normalized:
        normalized = normalized.split("/", 1)[1]
    return normalized


def _assert_distribution_contents(artifact: Path) -> None:
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            members = [_normalized_member(name, sdist=False) for name in archive.namelist()]
        assert f"{PACKAGE_IMPORT_ROOT}/__init__.py" in members
        assert f"{PACKAGE_IMPORT_ROOT}/py.typed" in members
        allowed_roots = {PACKAGE_IMPORT_ROOT}
        allowed_roots.update(
            name.split("/", 1)[0]
            for name in members
            if ".dist-info" in name.split("/", 1)[0]
        )
        unexpected = sorted(
            root
            for root in {name.split("/", 1)[0] for name in members if name}
            if root not in allowed_roots
        )
        assert not unexpected, f"wheel contains unexpected roots: {unexpected}"
        return

    with tarfile.open(artifact, mode="r:gz") as archive:
        members = [
            _normalized_member(member.name, sdist=True)
            for member in archive.getmembers()
            if member.isfile()
        ]
    assert f"src/{PACKAGE_IMPORT_ROOT}/__init__.py" in members
    assert f"src/{PACKAGE_IMPORT_ROOT}/py.typed" in members
    assert "tests/public_api_v3.sha256" in members
    forbidden = sorted(
        name for name in members if name.split("/", 1)[0] in FORBIDDEN_ROOTS
    )
    assert not forbidden, f"sdist contains product files: {forbidden[:10]}"


def _python_in(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _run_installed_conformance(artifact: Path, verifier: Path) -> None:
    with tempfile.TemporaryDirectory(prefix=f"harness-core-{artifact.suffix[1:]}-") as raw:
        root = Path(raw)
        environment = root / "venv"
        run_directory = root / "run"
        run_directory.mkdir()
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = _python_in(environment)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                str(artifact),
            ],
            check=True,
            cwd=run_directory,
        )
        copied_verifier = run_directory / verifier.name
        copied_verifier.write_bytes(verifier.read_bytes())
        subprocess.run(
            [str(python), copied_verifier.name],
            check=True,
            cwd=run_directory,
            env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Harness Core wheel and sdist in clean environments."
    )
    parser.add_argument("dist", type=Path)
    args = parser.parse_args()
    dist = args.dist.resolve()
    wheel = next(iter(sorted(dist.glob("*.whl"))), None)
    sdist = next(iter(sorted(dist.glob("*.tar.gz"))), None)
    if wheel is None or sdist is None:
        raise SystemExit("dist must contain one wheel and one sdist")
    verifier = Path(__file__).with_name("installed_conformance.py").resolve()
    for artifact in (wheel, sdist):
        _assert_distribution_contents(artifact)
        _run_installed_conformance(artifact, verifier)
        print(f"verified clean installation: {artifact.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
