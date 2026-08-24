# Release process

[English](releasing.md) | [简体中文](releasing.zh-CN.md)

DeepKeel releases use an immutable Git tag, a verified wheel and
source distribution, and synchronized English and Simplified Chinese
documentation. `pyproject.toml`, `deepkeel.version`, README install commands,
the changelog, and the tag must identify the same version.

## Verification tiers

- Pull requests run one Python 3.12 fast gate for lint, typing, synchronized
  docs and deterministic non-PostgreSQL tests.
- Main runs the full Python 3.12-3.14 and Windows/Linux compatibility matrix,
  coverage, concurrency, clean distributions and PostgreSQL recovery contracts.
- A release tag repeats the failure/recovery, PostgreSQL and concurrency gates
  before building immutable artifacts. GitHub uploads are replace-safe for the
  same tag and PyPI publishing skips an already published immutable version.

## Release gate

1. Confirm the public SDK layers have no duplicate symbol owners and intentionally
   review any public API fingerprint change.
2. Run Ruff, mypy, the full test suite with at least 80% statement coverage, and
   the deterministic 300-request/32-worker Core benchmark.
3. Run the PostgreSQL contracts and multi-worker recovery baseline against a
   disposable database.
4. Build wheel and sdist, run metadata checks and clean-install conformance
   against both, then generate the SPDX JSON SBOM and `SHA256SUMS` manifest.
5. Verify both READMEs, the semantic tag contract, and the migration notes.
6. Push the release commit before creating the immutable tag and GitHub Release.
7. Update downstream Hosts to the exact released tag or commit, then run their
   own Capability Pack, persistence, frontend, and deployment regressions.

For a downstream Host candidate, build the wheel exactly once after committing
the Core tree:

```powershell
uv run python scripts/build_release_candidate.py
```

The command rejects a dirty tree and emits one wheel plus a sidecar manifest
containing the immutable source SHA/ref/commit time, package version, build
environment, optional Host SHA/release ID, build timestamp, and wheel SHA256.
For a coordinated Host release, pass `--host-sha <sha> --release-id <id>`.
A Host must install this wheel in an isolated environment, run
its gates there, deploy the same bytes without rebuilding, verify the SHA256
again before installation, and retain both Core and Host provenance in release
metadata.

Every third-party GitHub Action is pinned to a full commit SHA. PostgreSQL
reference adapters have an independent coverage gate because package-wide
coverage collected without the `postgres` marker does not exercise them.

On Windows, the package-owned gate is:

```powershell
$env:DEEPKEEL_TEST_POSTGRES_DSN = "postgresql://..."
./scripts/verify.ps1
uv run python -m verification.release_contract --tag v4.1.0
uv run python -m verification.readme_contract
```

Do not tag a dirty tree, move an existing release tag, or publish a release
whose downstream Host still imports internal `deepkeel.*` implementation
modules. Release artifacts are outputs; they are not committed to the source
tree.
