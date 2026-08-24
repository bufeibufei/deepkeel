# 上下文管理

[English](context-management.md) | [简体中文](context-management.zh-CN.md)

DeepKeel 同时按层级和相互独立的 Scope、Visibility、Retention、Representation、
Authority、Subject 维度划分上下文：

- L1：固定控制信息与权威 Subject Context；
- L2：当前 Thread 与 Run 的工作集；
- L3：检索得到的 Memory 与 Evidence，最先允许省略。

规划过程感知 Token，且与模型上下文上限相关。Tool Call 与 Tool Result 始终作为
原子交换保留。L2 过大时，DeepKeel 保留最近的原始后缀并创建带来源的 Checkpoint；
原始事件仍是权威事实。Subject 不一致的权威上下文会被隔离，而不是静默混合。

调用模型前，`ContextQualityGate` 可以审计 Provenance、Authority、Subject 对齐、
重复内容、空项与声明的 Token 估算。`observe` 模式只记录问题，不改变输入；
`enforce` 模式会拒绝关键 L1/L2 缺陷，并允许可替换策略决定哪些 L3 质量问题可被
容忍。质量报告属于诊断证据，不是新的事实来源。

## 语义 Checkpoint

`DeterministicContextWindowManager` 可以接收可选的
`ContextCheckpointBuilder`。Builder 获得确定性草稿，以及被省略和保留消息的防御性
副本，可以补充目标、决策、进度、失败和下一步。Core 会根据不可变 Source Field
校验返回的 Checkpoint，并要求每个关键事实引用输入消息或之前 Checkpoint 中的事实。
输出无效或 Builder 失败时，系统回退到确定性草稿，并将原因写入 Context Diagnostic。
因此语义压缩可以替换，但模型摘要不会成为权威事实。

同一个 Working-context Compactor 会同时服务 Runtime Context Window 和最终模型
Provider 的输入预算。Host 只需配置一次语义边界，不会在路由与调用阶段得到两套
不同摘要。

Host 通过 `RuntimePorts` 提供领域上下文。Capability Package 可以通过注册的
Contributor 丰富通用 Envelope，但不能绕过 Visibility、Budget 或 Provenance Policy。

## 选择性 Memory Recall

长期 Memory 是可选 L3 上下文，不应在每个请求中强制检索。Host 可以注入
`RuntimePorts(memory_recall_coordinator=...)` 并实现产品侧
`MemoryRecallPolicy`。Policy 接收精简、可序列化的 `MemoryRecallRequest`，其中包含
请求身份、Subject Scope、近期工作历史、PendingAction 与 Skill 元数据，并返回：

- `skip`：当前工作上下文已经足够；
- `prefetch`：在 Context Window 规划前检索 Memory；
- `agent_decide`：不预取，由 ReAct Loop 在后续 Observation 表明需要历史时调用受
  治理的 Memory Search Tool。

`DefaultMemoryRecallCoordinator` 通过通用 `MemoryPort` 执行 Prefetch，应用有界
TTL Cache，将投影后的 Record 注入 L3，并在 Memory 基础设施不可用时 Fail-open。
Policy 也可以为已退出或不安全的 Subject Scope 禁用 Runtime Memory Search Tool。
新 Turn 的决策与结果会生成内部 `memory.recall.*` 事件并写入诊断；Resume 路径仍以
`run.resumed` 为首个事件，只在诊断中保留 Recall 跳过决策。Coordinator 不会把
Prompt 或敏感原始记录写入公开 Telemetry。

异步 Host 可以实现 `AsyncMemoryRecallPolicy` 与 `AsyncMemoryPort`。Coordinator 优先
调用原生 `adecide()` 和 `asearch()`；同步实现通过明确的线程 Offload 保持兼容，
避免阻塞 Host Event Loop。

Coordinator 支持 `legacy`、`shadow` 和 `enforced` 三种上线模式。`legacy` 与
`shadow` 保留 Eager Prefetch，供 Host 比较 Trace；`enforced` 才严格执行 Policy
决策。Core 不负责分类产品意图或抽取 Memory Domain，这些规则属于 Host；存储与
检索则继续通过 `MemoryPort` 可替换。
