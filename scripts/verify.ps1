$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$outputRoot = Join-Path $repoRoot "dist"

Push-Location $repoRoot
try {
    & uv sync --extra test
    if ($LASTEXITCODE -ne 0) {
        throw "Harness Core dependency sync failed."
    }

    & uv run ruff check src tests verification
    if ($LASTEXITCODE -ne 0) {
        throw "Harness Core lint contracts failed."
    }

    & uv run mypy src/harness_core
    if ($LASTEXITCODE -ne 0) {
        throw "Harness Core type contracts failed."
    }

    & uv run pytest -q --cov=harness_core --cov-report=term --cov-fail-under=80
    if ($LASTEXITCODE -ne 0) {
        throw "Harness Core package contract tests failed."
    }

    & uv run python verification/concurrency_benchmark.py --requests 300 --workers 32
    if ($LASTEXITCODE -ne 0) {
        throw "Harness Core concurrency baseline failed."
    }

    & uv build --out-dir $outputRoot --clear
    if ($LASTEXITCODE -ne 0) {
        throw "Harness Core distribution build failed."
    }

    & uv run python verification/verify_distributions.py $outputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Harness Core clean-install conformance failed."
    }
}
finally {
    Pop-Location
}
