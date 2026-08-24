# 整体架构

[English](architecture.md) | [简体中文](architecture.zh-CN.md)

DeepKeel 是一个与具体产品无关的 Agent Runtime 内核。Host 负责传输层、身份、
持久化数据库、模型凭据、产品策略和用户体验；Capability Package 通过公开的
Extension SDK 提供版本化 Skill、工具、Artifact、Handoff、上下文贡献器、MCP
Server 与 SubAgent。可选的 MCP 和 A2A Adapter 位于互操作边界，并统一投影为
受治理的 `ToolProvider` 或 `SubAgent` 契约。

```text
Host 应用
  -> RuntimeRequest + RuntimePorts
  -> DeepKeel HarnessRuntime
       -> 上下文规划与模型路由
       -> 受治理的模型/工具循环，以及可选的有界执行计划
       -> 中断、恢复与结算
       -> 类型化事件、Artifact 与诊断信息
  -> RuntimeResult
Capability Packages -> 版本化贡献 -> RuntimeGeneration
```

一个 Host 可以从同一个不可变 `RuntimeGeneration` 暴露多个面向用户的根 Agent。
Capability Package 声明 `AgentEntrypointSpec`，每个会话将其解析为不可变的
`CapabilityView`：

```text
已安装的 RuntimeGeneration
  与 AgentEntrypointSpec 的包/依赖作用域取交集
  与当前 Skill 策略取交集
  与渐进式工具披露结果取交集
  与 Runtime Policy、Budget 决策取交集
  -> 当前会话最终可用的 CapabilityView
```

EntryPoint 只是为同一张已编译图选择一个受限视图，不会创建第二套 Runtime。
内部 SubAgent 是有界子任务，只能进一步收窄父任务的有效作用域。

LangGraph 是内置执行 Adapter，负责图状态与支持 Checkpoint 的控制流。Turn
Coordinator 依赖内部 `TurnExecutionEngine` 契约，`LangGraphExecutionEngine`
负责 invoke、resume 与 recovery 映射。DeepKeel 的公开契约不暴露 LangGraph
类型，因此 Host 与 Capability Package 不会耦合到图构建细节。

规范的 `RuntimeEventEnvelope` 是持久化、流式输出和可观测性的共同事件事实。
Telemetry 与紧凑 Trace 行均由该 Envelope 投影，而不是各自生成。完整
`RuntimeResult` 保留恢复和诊断信息；`RuntimeResult.to_summary()` 提供更精简的
产品读取模型。

公开集成层包括：

- `runtime_sdk`：请求、结果、状态、作用域、生命周期与执行；
- `extension_sdk`：Capability Package、Skill、工具、Artifact 与 Handoff；
- `adapter_sdk`：Host 基础设施 Port、契约验证与装配；
- `discovery_sdk`：与 Provider 无关的 Skill/Tool 混合检索和重排；
- `memory_sdk`：产品无关的 Memory Record 与 Memory Port；
- `mcp_sdk`：受治理的 MCP Server 与传输集成；
- `orchestration_sdk`：有界执行计划、SubAgent 与多方推演契约；
- `a2a_sdk`：实验性的 A2A 1.0 远程专家互操作。

`RuntimePorts` 保留扁平兼容契约，也可以由四组内聚 Port 组合：

- `RuntimePersistencePorts`：状态、Checkpoint、Journal 与 Replay；
- `RuntimeGovernancePorts`：Policy、Budget、Health 与 Control；
- `RuntimeObservabilityPorts`：Telemetry、Trace 与评测证据；
- `RuntimeExecutionPorts`：模型、工具、上下文、Reference 与 Secret 边界。

实现层刻意隐藏在公开 Facade 后：Turn 协调委托给执行和失败处理模块，模型与工具
Gateway 委托 Provider 调用与结算，Graph Node 委托模型步骤，上下文与 SubAgent
执行使用有界支持模块。自动化 AST 导入图测试会拒绝内部依赖环，结构预算则防止
Facade 再次膨胀为单体模块。
