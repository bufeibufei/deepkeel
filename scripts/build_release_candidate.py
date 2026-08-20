"""Build one immutable DeepKeel wheel and its provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dist/candidate")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--host-sha", default="")
    parser.add_argument("--release-id", default="")
    args = parser.parse_args()

    status = _git("status", "--porcelain")
    if status and not args.allow_dirty:
        raise SystemExit("refusing to build a release candidate from a dirty DeepKeel tree")
    source_sha = _git("rev-parse", "HEAD")
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("deepkeel-*.whl"):
        old.unlink()
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output)],
        cwd=ROOT,
        check=True,
    )
    wheels = list(output.glob("deepkeel-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one candidate wheel, found {len(wheels)}")
    wheel = wheels[0]
    manifest = {
        "schema_version": "deepkeel-release-candidate-v1",
        "project": "deepkeel",
        "version": _wheel_version(wheel),
        "source_sha": source_sha,
        "source_ref": _git("branch", "--show-current") or "detached",
        "source_commit_time": _git("show", "-s", "--format=%cI", "HEAD"),
        "host_sha": str(args.host_sha or "").strip(),
        "release_id": str(args.release_id or "").strip(),
        "wheel": wheel.name,
        "wheel_sha256": _sha256(wheel),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "builder": f"{Path(sys.executable).name} {platform.python_implementation()}",
        "built_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = wheel.with_suffix(wheel.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wheel": str(wheel), "manifest": str(manifest_path), **manifest}))
    return 0


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_version(path: Path) -> str:
    with ZipFile(path) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8", errors="replace")
    for line in metadata.splitlines():
        if line.startswith("Version: "):
            return line.removeprefix("Version: ").strip()
    raise RuntimeError("candidate wheel metadata has no Version")


if __name__ == "__main__":
    raise SystemExit(main())
