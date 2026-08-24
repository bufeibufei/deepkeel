# Verification matrix

[English](verification-matrix.md) | [简体中文](verification-matrix.zh-CN.md)

DeepKeel separates pull-request feedback, full regression, and release evidence.
The purpose of this matrix is to connect each engineering claim to an executable
gate. It does not replace a downstream Host's business-quality evaluation.

## Gate layers

| Layer | Workflow or command | Purpose |
| --- | --- | --- |
| Pull-request fast gate | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | Fast deterministic feedback for lint, types, docs, and non-PostgreSQL tests |
| Main/full regression | [`.github/workflows/full-regression.yml`](../.github/workflows/full-regression.yml) | Coverage, structural budgets, distribution checks, PostgreSQL, and concurrency |
| Host compatibility | [`.github/workflows/host-compatibility.yml`](../.github/workflows/host-compatibility.yml) | Verify a downstream Host against an installed DeepKeel ref or wheel |
| Performance baseline | [`.github/workflows/performance.yml`](../.github/workflows/performance.yml) | Run stable Linux latency and throughput checks outside the fast correctness gate |
| Release gate | [`.github/workflows/release.yml`](../.github/workflows/release.yml) | Verify immutable source, build distributions, produce SBOM/checksums/provenance, and publish |
| Local release verification | [`scripts/verify.ps1`](../scripts/verify.ps1) | Reproduce the main package-owned checks before a release |

## Guarantee-to-test mapping

| Guarantee | Primary evidence | Failure detected |
| --- | --- | --- |
| Public SDK changes are explicit | `tests/test_public_sdk.py`, API snapshot and fingerprint checks | Symbol ownership drift, accidental export, incompatible contract change |
| Internal architecture remains acyclic and bounded | split-module tests, import-graph and quality-budget verifiers | New dependency cycle, facade regrowth, file or method complexity regression |
| Sync, async, and streaming share lifecycle semantics | `tests/test_async_runtime.py`, installed conformance scenarios | Divergent terminal state, dropped stream text, blocking async adapter |
| Canonical state remains authoritative | `tests/test_checkpoint_authority.py` | Stale fallback overrides canonical state, invalid recovery source |
| Crash and rollback paths settle consistently | `tests/test_fault_injection.py` | Event/state divergence, partial settlement, unsafe cleanup |
| Tool side effects are replay-safe | tool execution and PostgreSQL reference tests | Duplicate execution, claim race, invalid terminal replay |
| Stale workers cannot settle newer work | lease, execution-fence, and multi-worker tests | Lost ownership still writes a tool or terminal result |
| Runtime scope is isolated | scope, cache, state, event, Memory, and PostgreSQL tests | Cross-tenant or cross-namespace reuse |
| Capability visibility fails closed | entrypoint, package trust, graph reuse, and disclosure tests | Out-of-scope Skill/tool becomes visible or executable |
| Context compaction preserves authority | tiered-context and context-quality tests | Current turn truncation, unsupported summary, subject contamination |
| Model adapters honor declared behavior | model-provider certification and execution-safety tests | Unsupported structured output, broken stream/tool call, unsafe fallback |
| MCP cannot bypass the tool boundary | MCP transport, trust, schema, and egress tests | Unvalidated output, secret leak, unsafe endpoint, permission bypass |
| Artifacts and UI state are stable contracts | runtime API, artifact-view, and UI projection tests | Refresh loses result, terminal run keeps composer blocked, invalid action |
| Candidate distributions contain the required API and docs | `verification/verify_distributions.py`, installed conformance | Source tree passes but wheel/sdist is incomplete or behaves differently |
| Bilingual entry docs remain synchronized | `verification/readme_contract.py` | Missing version, install command, SDK, architecture, or local documentation link |

## Test styles

### Scenario evaluation

`EvalSuiteRunner` executes deterministic cases against a runtime result and trace.
Cases can assert final status, selected Skill, tool order, artifact schema, error
code, step budget, and event order. Capability Packages add domain answer-quality
evaluators without changing Core.

Use this layer for questions such as: “Did the model select the inventory Skill
and call the lookup tool before answering?” It identifies a decision or contract
regression, not just a Python exception.

### Adapter conformance

Public Port verifiers run the same behavioral contract against an in-memory
reference and the real Host adapter. They check scope, optimistic concurrency,
idempotency, cancellation, transaction, and capability declarations.

Use this layer for questions such as: “Does the custom PostgreSQL state store
reject a stale version exactly like the reference contract?”

### Fault injection

Fault tests deliberately fail storage, event delivery, checkpoint cleanup,
provider calls, cancellation, and Worker ownership at specific boundaries. They
assert that the run either recovers from an authoritative snapshot or fails
closed with one diagnosable terminal state.

Use this layer for questions such as: “If the process crashes after a state write
but before event projection, can replay converge without duplicating a public
terminal event?”

## Local commands

Fast package checks:

```powershell
uv sync --extra test
uv run ruff check src tests verification examples
uv run mypy src/deepkeel
uv run python verification/type_debt_budget.py
uv run python verification/readme_contract.py
uv run pytest -q -m "not postgres"
```

Coverage and structural budgets:

```powershell
uv run pytest -q -m "not postgres" `
  --cov=deepkeel `
  --cov-report=json:coverage.json `
  --cov-fail-under=80
uv run python verification/quality_budget.py --coverage coverage.json
```

PostgreSQL reference and multi-worker behavior:

```powershell
$env:DEEPKEEL_TEST_POSTGRES_DSN = "postgresql://..."
uv sync --extra test --extra postgres
uv run pytest -q -m postgres tests/test_postgres_reference.py
uv run python -m verification.postgres_multiworker --requests 64 --workers 12
```

Deterministic Core concurrency baseline:

```powershell
uv run python verification/concurrency_benchmark.py --requests 300 --workers 32
```

Complete local verification:

```powershell
.\scripts\verify.ps1
```

## What a downstream Host must still verify

Core gates cannot prove product correctness. A production Host should add:

- authentication and tenant authorization tests at every run operation endpoint;
- its actual database, queue, lease, secret, and model adapters to conformance;
- Capability Package scenarios for Skill/tool selection and answer quality;
- browser tests for SSE reconnect, refresh, pending actions, artifacts, and input
  locking;
- load tests at the expected traffic and long-task concurrency;
- privacy checks for prompts, tool payloads, traces, logs, and evaluation samples;
- a release manifest tying the Host commit, DeepKeel commit, wheel SHA-256,
  migrations, and deployed image together.

See [Production readiness](production-readiness.md),
[Host compatibility](host-compatibility.md), and
[Supply-chain controls](supply-chain.md) for the corresponding integration rules.
