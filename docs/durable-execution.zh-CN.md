# 持久化执行

[English](durable-execution.md) | [简体中文](durable-execution.zh-CN.md)

DeepKeel 将执行契约与持久化实现分离。Runtime State、Checkpoint、幂等、调用、
任务与事件 Port 可以由 PostgreSQL 或其他持久化存储实现，而无需修改 Agent Loop。

生产 Adapter 必须提供乐观并发、稳定所有权，并在并发工作边界提供 Lease 或 Fence
语义和幂等结算。应针对真实实现运行公开的 Adapter Conformance Suite。

挂起是显式行为：工具返回类型化 `PendingAction`，Host 持久化该动作，随后以类型化
Observation 恢复同一个 Run。取消采用协作式语义，并与完成、失败共用统一结算路径。

## 持久化权威边界

DeepKeel 为每种状态机制分配互不重叠的职责：

| 机制 | 权威职责 | 不得决定 |
| --- | --- | --- |
| `RuntimeStateStore` | 产品可见的规范 Run 状态、有序事件序列、结算、Fence Generation 与最新 Portable Checkpoint 投影 | LangGraph Node 调度细节 |
| `DurableCheckpointStore` | Portable Recovery Envelope，以及规范状态无法加载旧 Run 时的兼容回退 | Host 是否解锁输入框，或是否把 Run 显示为终态 |
| LangGraph Checkpointer | 当前 Super-step 的图内续跑信息和 Interrupt/Resume Cursor | 产品可见状态、最终结算或跨版本恢复策略 |

恢复首先读取 `RuntimeStateStore` 并记录规范读取失败；只有满足兼容条件时，才读取
Durable Store。Graph 执行随后才访问自己的 Checkpointer，且绝不能覆盖 Portable
Checkpoint 的权威性。
