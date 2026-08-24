# Agent EntryPoint

[English](agent-entrypoints.md) | [简体中文](agent-entrypoints.zh-CN.md)

DeepKeel 可以从同一个已安装 Runtime Generation 暴露多个可直接访问的根 Agent。
通用助手与领域专家共享生命周期、Graph、持久化和可观测契约，但不必共享全部能力。

## 契约

Capability Package 通过 Extension SDK 注册 `AgentEntrypointSpec`。该规格定义面向
产品的身份和收窄策略：

- Package 与依赖作用域；
- Skill、Tool 与 SubAgent Allowlist；
- 专家 System Prompt 与模型策略；
- Context、Memory 与 Handoff 策略元数据；
- 用于保证持久化会话可复现的版本号。

Turn 准备阶段会将规格与不可变 `RuntimeGeneration` 解析为 `CapabilityView`。该视图
包含最终有效的 Package、Skill、Tool、SubAgent、上下文贡献器、Artifact、Memory
Namespace、Permission Scope 与稳定 `scope_hash`。

## 执行

Host 将 `agent_entrypoint_id` 与 `agent_entrypoint_version` 随会话持久化，并在每次
`RuntimeRequest` 中传入。EntryPoint 是会话不变量；切换专家应创建另一个根会话，
不能静默改变已有历史的语义。

所有 EntryPoint 都运行在规范 ReAct Graph 上。DeepKeel 只编译一次 Graph，并通过
每轮 State 携带 `CapabilityView`。工具披露、上下文贡献器、生命周期 Hook 和显式
Skill 激活均在运行时过滤，不会为每个 EntryPoint 创建 Graph 或 Worker Pool。

## 安全

有效作用域默认 Fail-closed。EntryPoint 引用了不存在或越权的 Capability 时，装配
会失败。Host 与 Package 完成上下文增强后，Runtime Control 字段会被重新应用，
避免贡献器删除或替换权威作用域。子执行可以调用 `narrow_capability_view`，但不能
添加父级没有的 Tool、Skill 或 SubAgent。

空 EntryPoint ID 会选择向后兼容的 unrestricted default view，便于现有 Host 渐进
迁移到持久化、版本化的专家 EntryPoint。
