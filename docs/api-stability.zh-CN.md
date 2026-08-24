# API 稳定性

[English](api-stability.md) | [简体中文](api-stability.zh-CN.md)

`RuntimeResultSummary` 与 `RuntimeResult.to_summary()` 是 v4.1 的增量投影。完整
`RuntimeResult` 仍是恢复与诊断的规范对象，因此现有 Host 无需迁移。新的产品读取
路径应优先使用 Summary，只在需要时通过专用运维接口获取 Trace 或 Checkpoint。

Stable 与 Candidate Release 均可从不可变 Git Tag 和 GitHub Release Artifact 安装。
只有仓库配置 Trusted Publisher 后才启用 PyPI 发布。

DeepKeel 遵循 Semantic Versioning。v4 公开 SDK 由 `tests/public_api_v4.sha256`
冻结；Public Symbol 或 Signature 变化必须经过显式 API Review，同时更新 Changelog、
Snapshot 与 Migration Note。`tests/public_api_semantics_v4.sha256` 还会冻结 Stable
Runtime、Extension 与 Memory 层的 Callable Signature、Model/Dataclass Field、Enum
Value 与选定行为入口，避免保留名称但破坏语义。

Package Root 与版本化 `*_sdk` 是公开 API；除非明确记录，其他模块都是实现细节。
持久化 Schema Version 与 Capability Pack Contract 独立于 Package Version 演进。

每个 Public Symbol 只有一个规范 SDK Owner。机器可读 Manifest 位于
`deepkeel.public_api.PUBLIC_API_MANIFEST`，Release Test 会拒绝新的跨层重复 Export。
稳定性分为：

- `stable`：受 v4 兼容策略保护的 Runtime、Extension 与 Memory Contract；
- `advanced`：面向基础设施作者的 Adapter 与 MCP Contract，只通过显式兼容审查变化；
- `experimental`：有界 Orchestration 与 A2A Adapter，可在 Minor Release 中完善。

`PUBLIC_API_BY_STABILITY` 提供对应机器可读 Facade。普通应用应停留在 Stable 层；
基础设施代码可选 Advanced；实验 Orchestration 应固定版本并藏在 Capability Package 后。

Release Candidate 用于 Host Compatibility Test。Stable v4 保留已记录 SDK 和持久化
契约行为；Deprecated API 在后续 Major Release 删除前必须提供迁移路径。

v4 Distribution Rename 保留持久化 `harness-core-v3` Contract。兼容最终 v3 Package
Line 的 Capability Generation 可以通过显式 Bridge 加载；新 Package 应声明 v4 范围。

v4 RC 周期中，`HarnessRuntimeBuilder` 从 `deepkeel.adapter_sdk` 移至
`deepkeel.runtime_sdk`，因为应用作者在普通构造中就需要它。旧模块保留过渡 Attribute，
新代码必须从 Runtime SDK 导入规范 Symbol。

4.1 还增量增加 Memory Recall、Tool Idempotency 与实验性有界 Delegation 的异步版本。
同步契约保持源码兼容；Async Host 应优先实现 Native Protocol，并只为线程安全 Legacy
Adapter 使用 Bridge。

DeepKeel 4.1 增加 Stable `AgentHarness` Golden Path，以及增量 Guardrail、Sandbox、
Context Quality、Skill/Tool Discovery 与 Online Eval Contract；可选 `a2a_sdk` 明确为
Experimental。持久化契约仍是 `harness-core-v3`，详见
[迁移到 4.1](migrating-to-4.1.zh-CN.md)。
