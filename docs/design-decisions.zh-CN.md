# 架构决策

[English](design-decisions.md) | [简体中文](design-decisions.zh-CN.md)

本文记录 DeepKeel 有意保留的边界，供维护者与集成方使用；它不主张所有 Agent
应用都必须采用同等程度的 Runtime 治理。

## 1. DeepKeel 是 Runtime，不是产品框架

DeepKeel 负责需要跨产品保持一致的语义：统一模型/工具生命周期；类型化 Run State、
中断、取消、恢复和结算；Policy、Budget、幂等、Lease 和 Fencing 边界；版本化事件、
Artifact、Observation 和失败契约；可替换 Port 与 Capability Package 装配。

Host 负责 HTTP、SSE、认证、租户、队列、凭据、数据库连接、部署、产品策略和前端；
Capability Package 负责领域行为。该边界避免 Runtime 吸收某个产品的 ORM、路由、
Prompt 与 UI 假设。

## 2. LangGraph 是内部引擎，不是公开抽象

LangGraph 适合支持 Checkpoint 的图执行、Interrupt 与 Resume Cursor。DeepKeel 将其
放在内部 `TurnExecutionEngine` 契约后，不向公开 SDK 暴露 `StateGraph`、`Command`
或 Saver 类型。

这样 Host 与 Package 依赖 DeepKeel 稳定契约而非图细节；产品状态与恢复策略不会
委托给 Graph Saver；执行引擎也可以独立测试或替换。代价是增加 Adapter 代码，但
产品生命周期本身远大于图遍历，DeepKeel 接受这一成本。

## 3. 一张已编译图，多个不可变 Capability View

DeepKeel 不为每个用户或专家编译新图。Worker 复用规范 Graph，并通过 Turn State
携带请求数据。`AgentEntrypointSpec` 解析为不可变 `CapabilityView`，收窄 Package、
Skill、Tool、SubAgent、Context、Memory、Permission、Prompt 与 Model Policy。

Capability View 是会话不变量。切换专家会创建新根会话；子任务只能收窄父 View。
这既限制图编译成本，也让能力作用域可审计。Package 必须通过 State 与注册契约表达
变化，不能在运行时修改共享 Graph。

## 4. 能力发现必须权限优先

大目录不能完整复制到每个 Prompt。DeepKeel 按以下顺序处理：

1. Runtime Generation 与当前 Agent EntryPoint 取交集；
2. 应用 Tenant、Role、Policy 与显式 Skill 约束；
3. 从剩余目录召回候选；
4. 重排并只披露有界 Descriptor；
5. 激活后才加载完整 Skill 或 Tool 详情。

Discovery Adapter 可以排序或 Abstain，但不能恢复已被权限过滤的能力。默认词法实现
保证可移植，大目录可通过 `deepkeel.discovery_sdk` 接入语义检索与重排。Core 提供
候选边界、Trace 证据和确定性契约，但正确选择仍需要产品评测。

## 5. Skill、Workflow、SubAgent 与 Handoff 各有职责

| 机制 | 适用场景 | 执行所有权 |
| --- | --- | --- |
| Prompt Skill | 同一 Agent 需要专业指令或示例 | 父 Run 保持 Loop |
| Interactive Workflow | 有界步骤之间仍需模型决策 | 父 Run 保持 Loop 与 State |
| Delegated Workflow | 确定性或异步领域工作有独立生命周期 | Domain Worker 执行，父 Run 以 Artifact 恢复 |
| SubAgent | 专家需要独立 Context 和模型/工具 Loop | 子 Run 受父 Scope、Budget、Depth 与取消约束 |
| Handoff | 必须等待用户、外部系统或后台任务 | 父 Run 持久化 `PendingAction`，再用 `Observation` 恢复 |

Delegated Workflow 默认将结构化 Artifact 交回父模型，再生成最终回答；只有产品明确
把 Artifact 本身定义为终态响应时才例外。

## 6. 三种持久化机制权威互不重叠

| 机制 | 权威职责 |
| --- | --- |
| `RuntimeStateStore` | 产品可见状态、有序事件、结算、Fence Generation 与当前 Portable Checkpoint 投影 |
| `DurableCheckpointStore` | Portable Recovery Envelope 与旧 Run 的显式兼容回退 |
| LangGraph Checkpointer | Super-step 或 Interrupt Cursor 的内部续跑信息 |

LangGraph Checkpoint 不能决定 Run 是否终止或 UI 是否可解锁。恢复先读规范 State；
只有能证明权威性和新鲜度时才允许兼容回退，存储、超时或反序列化错误默认 Fail-closed。

## 7. 有序 Runtime Event 是投影源

版本化 `RuntimeEventEnvelope` 是生命周期事实。SSE、Durable Journal、OTel Span、
紧凑 Trace 与前端 `ui_state` 都从同一事件流投影。Answer Delta 可以合并；终态结算、
PendingAction、Artifact 和恢复事实必须持久化。重放不能产生重复公共副作用。

## 8. 副作用需要幂等与执行所有权

Tool Call 使用由 Run、Turn 与 Call 派生的稳定执行身份。Execution Store 原子 Claim，
可以重放已结算结果，或拒绝竞争 Owner。结算前 Core 校验当前 Execution Fence，失去
Lease 的旧 Worker 不能在新 Worker 接管后提交迟到结果。写工具还应把同一 Idempotency
Key 与 Fencing Token 传递给真正执行副作用的下游系统。

## 9. Context Layer 表达不同权威性

- L1：受保护的当前 Turn Context 和完整 Tool Exchange；
- L2：带 Source Range、Fingerprint、Subject 与 Checkpoint Lineage 的紧凑工作上下文；
- L3：由 Host Memory Policy 和 Storage Port 选择的 Durable Recall。

Compaction 不能把 Assistant 文本当成已完成工作，也不能静默截断当前请求。受保护
上下文无法适配模型时，应明确失败，而不是构造误导 Prompt。

## 10. Production 是可执行 Profile

`build()` 用于测试与本地嵌入；生产使用 `build_production()`，当必须的 Durable 或
Governed Port 缺失、歧义，或使用已知 `InMemory*`/`Noop*` Adapter 时 Fail-closed。
Profile 无法证明自定义 Adapter 正确，Host 仍须执行 Conformance、PostgreSQL 或
等效多 Worker 测试、故障注入与下游兼容门禁。

## 非目标与取舍

- DeepKeel 不直接优化模型质量，而是让路由、上下文、能力选择和失败可测、可替换；
- Core 不定义通用业务 Memory Taxonomy，Host 负责语义与保留策略；
- Core 不提供认证、租户数据库、队列或 UI；
- 内部模块与实验 SDK 可以演进，稳定 SDK 受公开兼容策略约束；
- 更多治理意味着更多 Adapter 与测试工作，应按风险选择 Production Profile，而不是
  把全部复杂度强加给简单的本地应用。
