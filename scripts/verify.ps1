$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$outputRoot = Join-Path $repoRoot "dist"

Push-Location $repoRoot
try {
    & uv sync --extra test --extra postgres
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel dependency sync failed."
    }

    & uv run deepkeel doctor
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel runtime diagnostics failed."
    }

    & uv run ruff check src tests verification
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel lint contracts failed."
    }

    & uv run mypy src/deepkeel
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel type contracts failed."
    }

    & uv run python verification/type_debt_budget.py
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel type debt budget failed."
    }

    & uv run python verification/readme_contract.py
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel synchronized documentation contract failed."
    }

    & uv run pytest -q --cov=deepkeel --cov-report=term --cov-report=json:coverage.json --cov-fail-under=80
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel package contract tests failed."
    }

    & uv run python verification/quality_budget.py --coverage coverage.json
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel critical coverage or complexity budget failed."
    }

    & uv run python verification/concurrency_benchmark.py --requests 300 --workers 32
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel concurrency baseline failed."
    }

    if (-not [string]::IsNullOrWhiteSpace($env:DEEPKEEL_TEST_POSTGRES_DSN)) {
        & uv run pytest -q -m postgres tests/test_postgres_reference.py
        if ($LASTEXITCODE -ne 0) {
            throw "DeepKeel PostgreSQL adapter contracts failed."
        }

        & uv run python -m verification.postgres_multiworker --requests 64 --workers 12
        if ($LASTEXITCODE -ne 0) {
            throw "DeepKeel PostgreSQL multi-worker baseline failed."
        }
    }

    & uv build --out-dir $outputRoot --clear
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel distribution build failed."
    }

    $distributionArtifacts = Get-ChildItem -Path $outputRoot -File |
        Where-Object { $_.Name -match '\.(whl|tar\.gz)$' } |
        Select-Object -ExpandProperty FullName
    & uv run twine check $distributionArtifacts
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel package metadata validation failed."
    }

    & uv run python verification/verify_distributions.py $outputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel clean-install conformance failed."
    }

    & uv run python -m verification.release_manifest $outputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "DeepKeel release checksum manifest failed."
    }
}
finally {
    Pop-Location
}
