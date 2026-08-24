# 生产就绪

[English](production-readiness.md) | [简体中文](production-readiness.zh-CN.md)

DeepKeel 负责执行语义；生产 Host 负责持久化基础设施、身份、部署和运维 API。

## 必需的 Host Adapter

启动生产 Worker 前必须运行可执行 Builder Gate：

```python
builder = HarnessRuntimeBuilder(profile="production").with_ports(production_ports)
report = builder.production_readiness()
runtime = builder.build_production()
```

关键 Worker Port 缺失、歧义或已知为进程本地实现时，`build_production()` 会
Fail-closed；同步 Adapter 可能阻塞 `arun()` 时会给出 Warning。该门禁无法证明任意
自定义 Adapter 可持久化，因此仍必须执行下文 Conformance Suite。

生产装配还要求 `tool_view_mode="enforced"`。开发期可以暂时保留 Legacy Disclosure，
生产 Worker 不得静默向模型暴露完整 Allowed Catalog。

配置可选执行安全 Port 后，门禁还会拒绝显式 `NoopSandboxPort`，提示
`LocalWorkspacePort` 的进程本地限制，并识别 In-memory Guardrail Replay 或 Online
Eval Store。自定义 Sandbox 必须落实 `ToolSpec` 声明的限制；必需 Sandbox 返回
`enforced=False` 时，Handler 不会执行。

多 Worker 部署不得使用 `InMemory*`，应为以下边界安装持久化实现：

- `RuntimeStateStore`；
- 需要旧 Portable Checkpoint 兼容回退时的 `DurableCheckpointStore`；
- `RuntimeEventJournal`、`ModelInvocationStore`、`ToolExecutionStore`；
- `RunLeaseStore`、`BudgetLedger`、`TraceStore`；
- `ContextSummaryCache`、`CancellableRunControl`；
- 运行时可变更 Package 时的 `CapabilityPackageStore`。

Worker Gate 直接检查 Checkpointer、Canonical State、Event Journal、Lease、模型/工具
幂等、Budget、Model Health、Run Control 与 Telemetry Port。Trace、Summary Cache
和 Package Catalog 属于 Host Control Plane，需要单独验证。

必须针对生产相同数据库配置运行 `deepkeel.adapter_sdk` 中所有适用 Verifier。State
与 Checkpoint Adapter 必须隔离属于不同 User Scope 的相同 Run ID。

可选 `deepkeel[postgres]` 为 Canonical State、Runtime Event Journal、带 Fencing
Generation 的 Run Lease 与 Portable Checkpoint 提供事务和约束。契约与多 Worker
基线见 [PostgreSQL 参考适配](postgresql-reference.zh-CN.md)。

部署预检运行 `deepkeel doctor` 和 `deepkeel postgres status`，流量进入前通过独立
Migration Job 执行 `deepkeel postgres upgrade --yes`。可运行的
[`production_worker`](../examples/production_worker/README.zh-CN.md) 示例展示完整公开
SDK 装配和可选 OTel 投影。

`RuntimeScope` 是规范 Tenant、Namespace 与 User Boundary。非默认 Tenant/Namespace
使用未实现 Scoped Extension 的 Legacy Adapter 时 Core 会 Fail-closed。Trace Adapter
只有在持久化并过滤全部三维 Scope 后才能声明 `supports_runtime_scope = True`。

## Capability Package 生命周期

Package 安装状态必须持久化并以乐观并发更新。Enable、Disable、Upgrade、Rollback 与
Uninstall 必须通过 `CapabilityPackageManager` 校验后才能激活新 Worker Generation。

Run 捕获不可变 `RuntimeGeneration`。不得修改运行中 Generation 的 Tool Registry 或
Capability Catalog。新工作进入新 Generation；旧工作按捕获的 Generation 恢复；没有
可恢复 Run 引用旧 Generation 后才能回收。

Worker 使用 `HarnessRuntimeBuilder.with_runtime_generation()` 装配。Core 会校验已安装
Capability Pack 与 Generation 完全一致，并在模型或工具执行前拒绝不兼容的持久化
Generation。生产 Catalog Adapter 应通过
`verify_capability_package_store_contract`。

## 运维控制

`RunOperations` 是可移植 Control-plane Facade，可检查 Scoped Run、在 Adapter 实现
`QueryableRuntimeStateStore` 时枚举 Run、关联 Trace、分类恢复结果并请求协作取消。

Host 安装 `RunRecoveryExecutor` 处理幂等 `resume`、`retry`、`requeue` 与
`terminalize` 命令。`RunOperations` 会先通过 Scoped State Projection 授权，再提交
命令；Queue Delivery、Scheduling 与 Remediation Policy 属于 Host。

只有 User-scoped State Lookup 成功后才能查询 Trace。Host 仍须在构造
`RunOperations` 前执行 Tenant Authorization，不能向不可信调用者暴露 Adapter Session。

## 恢复拓扑

使用全局唯一 `run_id`、一条 Durable State Mutation Stream，并保证同一时刻只有一个
Lease Owner。Worker 丢失 Execution Fence 后必须立即停止提交。允许消息重投；模型与
工具 Store 必须原子 Claim 并重放已结算结果。

恢复顺序：

1. 读取 User-scoped `RuntimeStateStore` 规范投影；
2. 获取或续租 Run Lease 与 Execution Fence；
3. 从 Portable Runtime Checkpoint 恢复；
4. 重放已结算模型/工具调用，不再次执行；
5. 提交唯一终态结算并释放控制资源。

Core 先提交终态，再删除兼容 Checkpoint。清理结果作为 Replay-only Internal Event
追加，不投递 Live Event Sink，从而保持公共终态事件边界。

`RunOperations.inspect()` 用于向运维人员展示恢复分类和 Trace 证据；批量调度和修复
策略仍属于 Host。`list_recovery_candidates()` 根据 Host 选择的截止时间查找旧的
非终态投影；生产 State Adapter 必须在同一原子事务中更新 Status、Checkpoint 与
`updated_at`。

## 异步 Adapter

网络 Adapter 应提供原生异步实现。`AsyncRuntimeStateStore`、
`AsyncDurableCheckpointStore`、`AsyncRuntimeEventJournal`、`AsyncRunLeaseStore`、
`AsyncTraceStore` 与 `AsyncToolExecutionStore` 定义可移植契约；Memory Recall 和有界
Delegation 也有异步 Policy、Store、Executor 与 Dispatcher 边界。

提供的 `Async*Adapter` Bridge 使用 `asyncio.to_thread`，仅适合线程安全的同步
Adapter。ORM Session 除非 Driver 明确允许跨线程，否则不得 Offload。同步 SubAgent
Bridge 遇到绑定 Session 会拒绝执行，除非 Host 提供 `session_factory`。

规范 `arun()` 直接消费原生异步 State、Event、Checkpoint、Lease、Memory、Tool
Idempotency 与 Delegation Port。同一个 Port 只能配置同步或异步一种形式。
`astream()` 使用有界 Queue 和同 Loop Backlog，并合并连续 Answer Delta 而不丢文本。

Adapter 可以通过 Adapter SDK 发布 `AdapterCapabilities`。生产 Readiness 根据声明
验证 Durability、Process Sharing、RuntimeScope、Transactionality 与 Cancellation
Safety。未声明的第三方 Adapter 仍兼容，但在宣称生产支持前应补充声明。

每个模型 Binding 都应运行 `certify_model_provider()`。静态认证检查 Streaming、
Native Tool、JSON Schema 声明；可选 Live Probe 检查实际文本、流、强制工具和结构化
输出。相同模型名在不同 Endpoint 或 Credential 下可能表现不同，因此认证以 Binding
为单位。

Remote Streamable HTTP MCP 默认拒绝 Private、Loopback 与 Link-local 目标；每次请求
前重新解析 DNS，把连接固定到已验证地址并保留原 Host/SNI，拒绝 Redirect 与继承的
Proxy Credential，并限制请求/响应字节。Host 必须配置 Hostname Allowlist；私网访问
只能显式用于受控内部 MCP。Capability URI Trust 按解析后的 Scheme、Host、Port 和
Path Segment 判断，不能使用裸字符串前缀。

Canonical State Adapter Error 必须 Fail-closed。只有 Canonical State 不存在或显式
Migration Marker 授权时才允许 Legacy Checkpoint 回退；传输、超时与反序列化错误
必须进入恢复诊断。没有健康、能力兼容模型时返回 `NO_ELIGIBLE_MODEL`；Memory Policy
失败时默认拒绝 Search，除非 Host 明确安装 Degraded-mode Policy。

Remote A2A 专家遵循相同 Egress 与 Credential Policy。Agent Card、Message、Task
State 和 Artifact 都是不可信输入；父 Run 保留最终答案所有权并传播取消。详见
[MCP 与 A2A 互操作](interoperability.zh-CN.md)。

## 评测

`EvalSuiteRunner` 对 Runtime Status、Tool Selection、Artifact Contract、Error Code、
Step Budget 与 Trace Event Order 做确定性回归，不负责业务答案质量。

业务 Capability Pack 应携带自己的 `EvalCase` 数据集。DeepEval 等语义或模型 Grader
可以在确定性门禁之上消费同一 `RuntimeResult` 与 Trace Record。

## 故障注入

发布前至少覆盖：模型/工具 Claim 时持久化不可用；外部副作用后、结算前 Worker 丢失；
旧 Lease Owner 提交；Event/Model/Tool 重复投递；并行工具部分失败；持久化 Replay
数据损坏；模型流或工具执行中取消；恢复后二次中断。包测试包含参考故障注入，但每个
生产 Adapter 都需要针对真实 Backend 的等效集成测试。

## 并发基线

发布前运行确定性 Core 基线，验证共享、预编译 Runtime 能安全服务独立 Turn：

```powershell
uv run python verification/concurrency_benchmark.py `
  --requests 300 `
  --workers 32 `
  --min-success-rate 1 `
  --max-p95-ms 2000
```

该基线只测本地确定性 Provider 下的 Core 开销，不能替代包含生产模型 Gateway、DB
Pool、Event Journal 和真实 Tool Adapter 的 Host Load Test。

PostgreSQL 可用时再运行数据库门禁：

```powershell
$env:DEEPKEEL_TEST_POSTGRES_DSN = "postgresql://..."
uv sync --extra test --extra postgres
uv run pytest -q -m postgres tests/test_postgres_reference.py
uv run python -m verification.postgres_multiworker --requests 64 --workers 12
```
