# 版本历史

[English](CHANGELOG.md) | [简体中文](CHANGELOG.zh-CN.md)

本文件提供当前版本线的中文发布说明。英文 `CHANGELOG.md` 是逐 Commit 历史的权威
记录；中文版本保留当前未发布改动、GA/RC 关键变化和旧版本演进索引。

## 未发布

- 为项目 README 关联的每份文档与示例补充简体中文版本和双向语言入口；缺失中文文件或
  中文 README 未链接对应版本时，Release Contract 会直接失败；
- 围绕可运行离线 Quickstart、架构/生命周期图、显式设计决策和可执行验证矩阵，重构
  双语项目入口；README Release Contract 会拒绝缺失章节、本地链接或中文对应文档；
- 在 LangGraph 外增加内部 Execution-engine Seam、紧凑 `RuntimeResultSummary` 和
  Event-to-telemetry 类型化投影，同时保留 v4 `RuntimeResult` 与 Event Contract；
- 增加 Capability Pack 脚手架/验证 CLI、进程内与隔离 Package Trust Policy、可运行
  HTTP/SSE Reference Host 和可复用 Downstream Host Compatibility Workflow；
- 现代 MCP Discovery 超时进入 Legacy Negotiation 前重新启动 stdio Server，避免已
  Reset Transport 破坏 Fallback；
- 轻量 Package 未声明 Manifest 时保留 Host 显式可信 Governance Scope；受治理
  Package 仍以 Manifest Permission 为权威。

## 4.1.0 - 2026-08-18

- 增加版本化、面向用户的 Agent EntryPoint 与逐会话不可变 Capability View；专家
  Agent 复用规范已编译 Graph，并收窄 Package、Skill、Tool、SubAgent、Context、
  Memory、Permission、Prompt 与 Model Policy；
- 在规范 ReAct Graph 内增加可选、有界 Plan & Execute：DAG 校验、安全并行读、串行
  副作用、有界 Revision、类型化进度事件与 Portable Interruption Recovery；
- 增加最小嵌入的 `AgentHarness` Golden Path，并保留生产装配所需
  `HarnessRuntimeBuilder`；
- 增加 Input、Model Output、Tool Boundary Guardrail，具备有序 Decision、Replay
  Safety、Audit、Provenance 与 Required Policy Fail-closed；
- 增加可替换 Sandbox/Workspace Port、执行上限、确定性 Cleanup、Policy Enforcement
  与 Production Readiness Check；
- 增加渐进式 Skill/Tool Discovery、可替换 Recall/Rerank，以及 L1/L2/L3 Context
  Provenance、Subject、Authority、Dedup 与 Token Budget 质量检查；
- 增加隐私受限 Online Eval 与 OTel GenAI 属性，默认不导出 Prompt、Tool Payload 或
  Model Output；
- MCP 升级到 `2026-07-28` 协议时代，支持 Stateless Discovery、Task Lifecycle、
  Safe Parameter Header、`outputSchema` 与有界 Legacy Fallback；
- 增加实验性 A2A 1.0 Adapter，把 Remote Agent Card/Task 映射到已有 SubAgent、
  Checkpoint、Cancellation、Artifact 与 Parent-owned Final Answer 生命周期；
- 扩展 Production/Release Gate，覆盖安全 Adapter、Clean Package、SBOM、Build
  Provenance、双语文档与 4.1 Migration Checklist；
- 持久化 Capability Pack Contract 保持 `harness-core-v3`，4.1 无需迁移已有 State。

## 4.1.0rc1 - 2026-08-10

- 将每个未中断 Run Segment 投影为一条层次化 OTel Trace，并在恢复 Segment 间建立
  Link；
- 增加受确定性 Source Range、Fingerprint、Subject 与 Fact Reference 校验的语义
  Context Checkpoint Builder；
- Adaptive Model Route 纳入 Health、Context Capacity、Modality、Native Tool、Latency、
  Cost 与 Remaining Budget，同时保留用户选择的 Single-model 语义；
- 拆分 Model Node、Result Projection、SubAgent 与 Claimed Turn Lifecycle，并建立
  Complexity Ratchet；
- 明确 Canonical Product State、Portable Recovery Checkpoint 与 LangGraph Continuation
  Checkpoint 的非重叠权威；
- 按 `RuntimeScope` 隔离 Event、Lease、Idempotency、Checkpoint 与运维 Identity；
- Async Host Loop 不再被线程安全同步 Adapter 阻塞，并保留有界 Streaming 与协作取消；
- 增加 Adapter Capability、Provider Certification、OTel Metric、MCP Egress/Size
  Control、Semantic API Snapshot 与关键 Coverage/Complexity Gate。

## 4.0.0 - 2026-08-08

- 将 Turn Preparation、Durable Snapshot Commit 与 Tool Idempotency 拆成可独立测试阶段；
- 冻结 Stable Runtime、Extension 与 Memory SDK 的完整语义描述；
- PostgreSQL Reference Bundle 增加 Capability Catalog、Context Summary、Memory 与
  SubAgent Store；
- 增加 Development/Testing/Production Profile，生产装配要求强制渐进式 Tool
  Disclosure；
- 将 `HarnessRuntimeBuilder` 提升为 Stable Runtime SDK 规范 Owner；
- 以 `deepkeel.contrib.postgres` 提供统一 PostgreSQL Worker Bundle；
- 增加 Memory、Tool Idempotency 与 SubAgent Delegation 的 Native Async Contract；
- 使用 Forward-only、带 Checksum 的 Schema Registry 代替临时建表；
- 增加隐私安全 OTel Projection、机器可读 Diagnostic、Migration CLI 与生产 Worker 示例。

## 4.0.0 RC 与更早版本

4.0 RC 完成 DeepKeel 品牌与 Python Namespace 重命名、Apache-2.0 授权、公开 v4 SDK、
产品中立 Quickstart、Runtime Port 分组、内部单体拆分、结构预算、PostgreSQL Reference、
Production Profile、双语 README 与自动 Release Contract。

3.x 是 DeepKeel 正式命名前的工程演进线，集中建立 Runtime State、Tool/Model 幂等、
Lease/Fencing、Checkpoint、Memory、Capability Package、SubAgent、Context 分层、
可观测性、评测与故障恢复基础。2.x 建立稳定 SDK 与 Host/Package/Core 边界；1.x 完成
最初的类型化 Agent Runtime Loop。逐版本变更请查看[英文完整历史](CHANGELOG.md)。
