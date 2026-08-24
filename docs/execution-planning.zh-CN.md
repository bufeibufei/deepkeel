# 执行计划

[English](execution-planning.md) | [简体中文](execution-planning.zh-CN.md)

DeepKeel 在规范 ReAct Graph 内提供可选 Plan & Execute 层，适用于多能力协作、依赖
排序、有界并行证据收集、持久化用户 Handoff 或显式结果综合。它不是第二套 Runtime，
不应用于问候、直接回答或单工具查询。

## 装配与 Skill 策略

Host 通过 `RuntimePorts(planning_enabled=True)` 启用内部规划控制工具，当前 Skill
再声明 `planning_policy`：

```json
{
  "mode": "preferred",
  "max_steps": 8,
  "max_revisions": 2,
  "max_parallel_steps": 4,
  "max_attempts_per_step": 2
}
```

`disabled` 隐藏控制工具，`allowed` 允许模型自行选择，`preferred` 提示模型只为真正
多步骤工作规划，`required` 强制首次模型迁移先经过规划工具。全局装配开关仍是最高
权威；Skill 不能启用 Host 未安装的控制工具。

## Plan 契约

`ExecutionPlan` 是版本化、可持久化 DAG。每个 `PlanStep` 都有稳定 Identity、目标、
Executor Kind、Capability Reference、Arguments、Dependencies、Success Criteria、
有界 Attempts、Execution Status 与 Result Projection。可执行步骤引用普通已注册 Tool；
Workflow 与 SubAgent 仍通过既有 Tool Boundary，不得绕开治理。

采用计划前，Core 校验计划与 Revision 上限、Step ID 唯一性和无环依赖、Tool 存在性
及 Skill Allowlist、Runtime Control Tool 隔离、已完成 Step 在 Revision 中不可变，
以及权威 `ToolSpec` 的 Read-only/Parallel-safety Metadata。

## 调度

Core 只调度依赖已满足的步骤。独立、只读、并行安全的 Tool 可以组成有界 Batch；
有副作用、会挂起、异步 Workflow 与 Synthesis 串行执行。每个 Tool Call 都携带稳定的
Plan、Revision、Step、Attempt、Idempotency 和 Resource Identity，然后继续通过既有
ToolExecutor、Policy、Budget、Hook、Checkpoint 与 Event 路径。

Retryable Failure 可以在 Attempt 上限内重试；Terminal Failure 把控制交回模型，进行
有界计划 Revision 或诚实的 Partial Answer。已完成 Step 不能被重规划删除或改写。

## 中断与恢复

Active Plan 同时存在于 LangGraph State 与 Portable Runtime Checkpoint。用户操作或
异步 Tool Result 会把对应 Plan Step 标记为 Waiting；恢复时精确解析该 Step，调度
新就绪依赖，不重放已完成工作。Live LangGraph Resume 与跨 Worker Durable Recovery
遵守相同语义。

## 事件与展示

计划事件使用 `plan.*` Namespace，包括 Validation、Start、Step Start/Complete/Wait/
Retry/Failure、Revision、Synthesis、Complete 与 Partial Complete。Payload 暴露目标、
进度、Step Identity、Capability Reference、Attempt 和 Status，但不暴露隐藏思维链。
Host 应展示紧凑进度，并把完整 Event History 保存在 Trace 或 Debug Surface。
