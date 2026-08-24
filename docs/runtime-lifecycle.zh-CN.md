# Runtime 生命周期

[English](runtime-lifecycle.md) | [简体中文](runtime-lifecycle.zh-CN.md)

每个请求都有稳定的 Run Identity，并通过同一个规范 Runtime Loop 执行。Run 可以
完成、失败、取消、等待用户输入、等待用户操作，或因异步工作而挂起。

Runtime 在模型工作前校验上下文和 Capability Generation，记录类型化的模型与工具
Observation，只结算一个终态，并发出可重放的生命周期事件。恢复时读取持久化的
Run Snapshot、PendingAction 与 Observation，而不是从 Assistant 文本推测进度。

启用执行计划后，模型可以通过内部 `runtime.create_plan` 控制工具创建有界 DAG。
计划同时存在于 Graph State 和 Portable Runtime Checkpoint 中。已就绪、只读且可
并行的步骤可以有界并发执行；有副作用、会挂起或不安全的步骤串行执行。用户操作
或异步中断恢复时，会继续同一个计划步骤；即使换了 Worker，也从 Portable
Checkpoint 恢复，不重放已完成工作。

Host 负责提供状态、Checkpoint、幂等、模型调用结算和事件投递的持久化实现。
参考 `InMemory*` Adapter 只用于测试与单进程嵌入。
