# 验证矩阵

[English](verification-matrix.md) | [简体中文](verification-matrix.zh-CN.md)

DeepKeel 将 Pull Request 快速反馈、完整回归与发布证据分层。该矩阵把每项工程承诺
关联到可执行门禁，但不能替代下游 Host 的业务质量评测。

## 门禁层级

| 层级 | Workflow 或命令 | 目的 |
| --- | --- | --- |
| PR 快速门禁 | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | 快速检查 Lint、Type、文档和非 PostgreSQL 确定性测试 |
| Main 完整回归 | [`.github/workflows/full-regression.yml`](../.github/workflows/full-regression.yml) | Coverage、结构预算、Distribution、PostgreSQL 与并发 |
| Host 兼容 | [`.github/workflows/host-compatibility.yml`](../.github/workflows/host-compatibility.yml) | 针对已安装 DeepKeel Ref 或 Wheel 验证下游 Host |
| 性能基线 | [`.github/workflows/performance.yml`](../.github/workflows/performance.yml) | 在快速正确性门禁外运行稳定 Linux 延迟与吞吐检查 |
| 发布门禁 | [`.github/workflows/release.yml`](../.github/workflows/release.yml) | 校验不可变源码、构建 Distribution、生成 SBOM/Checksum/Provenance 并发布 |
| 本地发布验证 | [`scripts/verify.ps1`](../scripts/verify.ps1) | 发布前复现包自有主门禁 |

## 工程保证与测试映射

| 保证 | 主要证据 | 可发现的失败 |
| --- | --- | --- |
| 公开 SDK 变化显式 | `tests/test_public_sdk.py`、API Snapshot 与 Fingerprint | Symbol Owner 漂移、意外 Export、不兼容契约变化 |
| 内部架构无环且有界 | Split-module Test、Import Graph 与 Quality Budget | 新依赖环、Facade 回长、文件或方法复杂度回归 |
| Sync/Async/Streaming 生命周期一致 | `tests/test_async_runtime.py`、Installed Conformance | 终态分歧、Stream 丢字、Async Adapter 阻塞 |
| Canonical State 权威 | `tests/test_checkpoint_authority.py` | 旧 Fallback 覆盖规范状态、恢复源无效 |
| Crash/Rollback 一致结算 | `tests/test_fault_injection.py` | Event/State 分叉、部分结算、不安全清理 |
| Tool 副作用可安全 Replay | Tool Execution 与 PostgreSQL Reference Test | 重复执行、Claim Race、终态 Replay 无效 |
| 旧 Worker 不能提交新工作 | Lease、Execution Fence 与 Multi-worker Test | 丢失所有权后仍写 Tool 或终态 |
| Runtime Scope 隔离 | Scope、Cache、State、Event、Memory 与 PostgreSQL Test | 跨 Tenant 或 Namespace 复用 |
| Capability Visibility Fail-closed | EntryPoint、Package Trust、Graph Reuse、Disclosure Test | 越权 Skill/Tool 可见或可执行 |
| Context Compaction 保留权威 | Tiered Context 与 Context Quality Test | 当前 Turn 截断、无来源摘要、Subject 污染 |
| Model Adapter 符合声明 | Provider Certification 与 Execution Safety Test | 结构化输出不支持、Stream/Tool Call 损坏、不安全 Fallback |
| MCP 不能绕过 Tool Boundary | Transport、Trust、Schema 与 Egress Test | 输出未校验、Secret 泄露、Endpoint 不安全、Permission Bypass |
| Artifact/UI State 契约稳定 | Runtime API、Artifact View 与 UI Projection Test | 刷新丢结果、终态仍锁输入、Action 无效 |
| Candidate Distribution 完整 | `verification/verify_distributions.py`、Installed Conformance | 源码通过但 Wheel/Sdist 缺文件或行为不同 |
| 双语入口文档同步 | `verification/readme_contract.py` | 版本、安装命令、SDK、架构、中文对应文档或本地链接缺失 |

## 测试类型

### 场景评测

`EvalSuiteRunner` 针对 Runtime Result 与 Trace 执行确定性 Case，可以断言 Final Status、
Selected Skill、Tool Order、Artifact Schema、Error Code、Step Budget 与 Event Order。
Capability Package 在不修改 Core 的情况下添加领域 Answer-quality Evaluator。

该层回答“模型是否选择 Inventory Skill，并在回答前调用 Lookup Tool”之类问题，定位
决策或契约回归，而不只捕获 Python Exception。

### Adapter 契约

公开 Port Verifier 对 In-memory Reference 和真实 Host Adapter 执行同一行为契约，
检查 Scope、Optimistic Concurrency、Idempotency、Cancellation、Transaction 与
Capability Declaration。

该层回答“自定义 PostgreSQL State Store 是否与参考契约一样拒绝旧 Version”。

### 故障注入

故障测试在 Storage、Event Delivery、Checkpoint Cleanup、Provider Call、Cancellation
与 Worker Ownership 等边界主动失败，断言 Run 要么从权威 Snapshot 恢复，要么以
唯一、可诊断终态 Fail-closed。

该层回答“State Write 后、Event Projection 前进程崩溃，Replay 能否收敛且不重复公共
Terminal Event”。

## 本地命令

快速检查：

```powershell
uv sync --extra test
uv run ruff check src tests verification examples
uv run mypy src/deepkeel
uv run python verification/type_debt_budget.py
uv run python verification/readme_contract.py
uv run pytest -q -m "not postgres"
```

Coverage 与结构预算：

```powershell
uv run pytest -q -m "not postgres" `
  --cov=deepkeel `
  --cov-report=json:coverage.json `
  --cov-fail-under=80
uv run python verification/quality_budget.py --coverage coverage.json
```

PostgreSQL 与 Multi-worker：

```powershell
$env:DEEPKEEL_TEST_POSTGRES_DSN = "postgresql://..."
uv sync --extra test --extra postgres
uv run pytest -q -m postgres tests/test_postgres_reference.py
uv run python -m verification.postgres_multiworker --requests 64 --workers 12
```

确定性 Core 并发基线：

```powershell
uv run python verification/concurrency_benchmark.py --requests 300 --workers 32
```

完整本地验证：

```powershell
.\scripts\verify.ps1
```

## 下游 Host 仍须验证

Core Gate 无法证明产品正确性。生产 Host 应补充：每个 Run Operation Endpoint 的认证
与 Tenant Authorization；真实 DB、Queue、Lease、Secret、Model Adapter Conformance；
Skill/Tool Selection 与 Answer Quality 的 Capability Package 场景；SSE 重连、刷新、
PendingAction、Artifact 与 Input Locking 的浏览器测试；目标流量和长任务并发 Load
Test；Prompt、Tool Payload、Trace、Log、Eval Sample 的隐私检查；以及关联 Host
Commit、DeepKeel Commit、Wheel SHA-256、Migration 与 Image 的 Release Manifest。

对应规则见[生产就绪](production-readiness.zh-CN.md)、
[下游 Host 兼容性](host-compatibility.zh-CN.md)和
[供应链治理](supply-chain.zh-CN.md)。
