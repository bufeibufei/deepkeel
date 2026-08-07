# Release process

DeepKeel release candidates use an immutable Git tag, a verified wheel and
source distribution, and synchronized English and Simplified Chinese
documentation. `pyproject.toml`, `deepkeel.version`, README install commands,
the changelog, and the tag must identify the same version.

## Candidate gate

1. Confirm the six public SDKs have no duplicate symbol owners and intentionally
   review any public API fingerprint change.
2. Run Ruff, mypy, the full test suite with at least 80% statement coverage, and
   the deterministic 300-request/32-worker Core benchmark.
3. Run the PostgreSQL contracts and multi-worker recovery baseline against a
   disposable database.
4. Build wheel and sdist, then run clean-install conformance against both.
5. Verify both READMEs and the semantic tag contract.
6. Push the release commit before creating the immutable tag and GitHub Release.
7. Update downstream Hosts to the exact released tag or commit, then run their
   own Capability Pack, persistence, frontend, and deployment regressions.

On Windows, the package-owned gate is:

```powershell
$env:DEEPKEEL_TEST_POSTGRES_DSN = "postgresql://..."
./scripts/verify.ps1
uv run python -m verification.release_contract --tag v4.0.0-rc.2
uv run python -m verification.readme_contract
```

Do not tag a dirty tree, move an existing release tag, or publish a candidate
whose downstream Host still imports internal `deepkeel.*` implementation
modules. Release artifacts are outputs; they are not committed to the source
tree.
