# 下游 Host 兼容性

[English](host-compatibility.md) | [简体中文](host-compatibility.zh-CN.md)

DeepKeel 的独立 Core 测试通过并不代表一次发布已经完成。Host 应维护一组只使用
公开 SDK 的小型契约测试，覆盖真实模型 Provider、Capability Package、持久化
Adapter、SSE 投影、取消与恢复行为。

可复用的 `Downstream Host Compatibility` Workflow 会检出候选 DeepKeel Revision，
将其安装到下游 Host 环境，并执行 Host 自有测试命令。可以在发布前手动运行，也可
由 Host Workflow 调用。私有仓库需要能够访问两个仓库的 Checkout Credential。

如果下游测试导入私有 `deepkeel.*` 模块、依赖 LangGraph 类型、在
`to_summary()` 足够时读取完整诊断，或未经 Trust Decision 就安装进程内第三方
Package，测试应当失败。这样可以把 Host 边界变成可执行契约，而不只是一张架构图。
