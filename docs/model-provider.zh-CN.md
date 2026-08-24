# 模型 Provider

[English](model-provider.md) | [简体中文](model-provider.zh-CN.md)

模型访问是一个 Port。Provider 描述模型身份、角色、Context Limit、结构化输出能力
与 Usage。路由可以选择 fast、reasoning、structured、embedding 或 rerank 角色，
而无需把厂商 Client 暴露给 Runtime Loop。

`single` 策略始终使用 Host 或用户选择的角色；`adaptive` 策略按每个模型步骤独立
决策。路由前，Core 会构造脱敏候选画像，其中包括声明能力、Context Window 是否
足够、共享健康状态、预计输入规模、当前预算用量，以及 Host 声明的延迟/成本层级。
不健康或能力不兼容的角色会被排除，再由阶段语义从可用 fast/reasoning 角色中选择。
完整决策证据写入 `model.route.selected`，自动路由不会成为不可见覆盖。

每次尝试都通过 Durable Invocation Envelope 记录。重试与 Fallback 受 Policy 和
Budget 约束，Replay Settlement 防止结果不确定时静默重复执行。

Secret 与 Provider Catalog 属于 Host。DeepKeel 诊断只暴露脱敏后的身份、能力、
延迟和 Usage 元数据，不暴露 Prompt 或模型 Payload。
