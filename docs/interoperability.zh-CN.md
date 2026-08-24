# MCP 与 A2A 互操作

[English](interoperability.md) | [简体中文](interoperability.zh-CN.md)

DeepKeel 将互操作协议放在 Runtime Kernel 外。MCP Tool 转换为受治理 `ToolSpec`，
A2A Remote Agent 转换为有界 `SubAgentSpec`。两种协议都不能绕过父 Run 的 Policy、
Budget、Event、Checkpoint、Cancellation 或 Artifact 契约。

## MCP

`deepkeel.mcp_sdk` 支持本地 stdio 与 Streamable HTTP。新连接优先使用现代
`2026-07-28` 协议时代，并只为已声明旧版本提供有界兼容回退。

现代路径提供：

- 无状态 `server/discover` 协商与逐请求协议元数据；
- 可缓存、可分页的工具发现；
- MCP Task 创建、轮询、更新、取消与输入续接；
- 安全的 `Mcp-Name` 与 `Mcp-Param-*` Header；
- 对静态可达 Primitive Property 的 `x-mcp-header` 校验；
- Server `outputSchema` 校验，之后结果才能进入 Runtime Observation；
- 类型化、脱敏的远程错误与有界请求/响应 Payload。

发现时会隔离无效 Tool，避免污染整个 Catalog。非 ASCII 或对空白敏感的 Header 值
使用 MCP Base64 Sentinel；缺失值或 Null 不会被镜像。

## A2A

`deepkeel.a2a_sdk` 是实验性的 A2A 1.0 Adapter。`A2ARemoteAgent` 将 Agent Card
映射到现有 SubAgent Registry。`A2ADelegationExecutor` 发送 Message，接受直接
Message 或 Task，执行有界轮询，投影 Remote Artifact，并把输入/认证要求映射成
类型化 PendingAction。

Remote Task Identity 会写入 Checkpoint，因此 Worker 重启后继续轮询，而不会重复
提交任务。父级取消会传播到 Remote Task。父 Agent 保留综合与最终答案所有权；远程
Agent 是产生 Observation 的专家，不是用户会话的独立权威。

## 如何选择边界

远程系统暴露适合一次 Tool Call 的操作或资源时使用 MCP；远程系统拥有独立生命周期
和 Artifact 的多步骤专家任务时使用 A2A；代码与信任边界均在本地时使用进程内
Capability Package。三条路径最终都汇入 DeepKeel 受治理的 Runtime Contract。
