$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$outputRoot = Join-Path $repoRoot "dist"

Push-Location $repoRoot
try {
    & uv sync --extra test
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel dependency sync failed."
    }

    & uv run ruff check src tests verification
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel lint contracts failed."
    }

    & uv run mypy src/deepkeel
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel type contracts failed."
    }

    & uv run pytest -q --cov=deepkeel --cov-report=term --cov-fail-under=80
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel package contract tests failed."
    }

    & uv run python verification/concurrency_benchmark.py --requests 300 --workers 32
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel concurrency baseline failed."
    }

    & uv build --out-dir $outputRoot --clear
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel distribution build failed."
    }

    & uv run python verification/verify_distributions.py $outputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel clean-install conformance failed."
    }
}
finally {
    Pop-Location
}
