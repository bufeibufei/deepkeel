# 可观测性

[English](observability.md) | [简体中文](observability.zh-CN.md)

DeepKeel 为模型路由、Tool Policy、Budget、Context Manifest、Capability Generation、
Recovery、SubAgent Lineage 与终态结算发出类型化 Runtime Event 和结构化 Diagnostic。

Telemetry 通过 `RuntimePorts` 注入，默认 Fail-open。默认记录不包含 Prompt、Tool
Argument 和模型结果。Host 应按 Run、Thread、Tenant 与 Invocation ID 关联记录，只在
明确数据策略下保留原始 Payload。

评测与生产消费同一 Result 和 Trace Contract。测试失败应能归因到 Runtime、Adapter、
Capability Package、模型或 Host Projection，而不是从用户可见文本猜测。

## OpenTelemetry 导出

需要标准 Trace 时安装 `deepkeel[otel]`。DeepKeel 只提供 Telemetry Adapter；Host
负责 OpenTelemetry SDK、Resource、Sampler、Processor 与 Exporter 配置。

```python
from deepkeel.contrib.otel import OpenTelemetryTelemetry

otel = OpenTelemetryTelemetry()
builder.configure_ports(telemetry=otel)
```

同一个未中断 Run Segment 的 Durable Record 会投影为一条 Trace：
`deepkeel.run.segment` 是 Root Span；Model、Tool、MCP 与 SubAgent 操作是 Child Span；
Lifecycle 与 Routing Decision 是 Root-span Event。Segment 结算后立即关闭；后续 Resume
创建新 Trace，并 Link 到之前 Segment。Replay Record 按 Event Identity 去重。临时
Stream Delta 聚合为 Root Span 的 Delta Count，不为每个 Token 创建 Span。

默认排除 User ID，Prompt、Tool Argument、Model Output 与任意嵌套值永远不会自动
提升为 Span Attribute。只有在明确 Sampling 与 Privacy Policy 下才设置
`include_ephemeral=True` 或 `include_user_id=True`。

关闭 Host SDK Provider 前调用 `otel.shutdown()`，让未完成 Segment 以
`deepkeel.segment.closed_reason=shutdown` 导出。

Adapter 同时记录 `deepkeel.runtime.events`、`deepkeel.runtime.failures`、
`deepkeel.operation.duration` 与 `deepkeel.runs.active`。Attribute Projection 会拒绝
Credential、Prompt、Question、Argument、Content、Result 与 Token 字段，并丢弃过长
字符串。Metric 是运维信号，不能替代权威 Event Journal 或 Runtime State Store。

Model Span 同时使用 OTel GenAI 语义属性描述 Provider、Requested/Response Model、
Operation、Token Usage、Finish Reason 与 Error Type，并保留稳定 `deepkeel.*` 属性，
避免 Dashboard 绑定某一版实验语义约定。Prompt 与 Completion Body 默认不导出。

PostgreSQL 作为权威 Diagnostic Trace、OTel 作为外部投影时使用
`CompositeTelemetry`。Exporter 失败不能改变 Run 语义，应单独监控 Exporter Health。

## 在线评测

Run 结算后，`OnlineEvalPort` 接收确定性、隐私受限的 Sample。`OnlineEvalPolicy`
控制 Status Filter、Sampling Rate，以及 Answer Content 是省略、Hash 还是包含。参考
`OnlineEvalPipeline` 运行于进程内，只适合开发；生产 Host 应持久化入队并在请求路径
外评测。评测失败只产生 Telemetry Signal，绝不改写权威 `RuntimeResult`。
