# Migrating to DeepKeel 4.1

[English](migrating-to-4.1.md) | [简体中文](migrating-to-4.1.zh-CN.md)

DeepKeel 4.1 is an additive GA release. The package and public SDK version moves
to `4.1.0`, while the persisted Capability Pack and runtime contract remains
`harness-core-v3`. Existing durable run state does not require a schema rewrite.

## English checklist

1. Pin the immutable `v4.1.0` tag and rebuild wheel/sdist consumers.
2. Keep application imports in the versioned `deepkeel.*_sdk` modules.
3. Prefer `AgentHarness` for minimal embedding and keep
   `HarnessRuntimeBuilder` for explicit production composition.
4. Review the additive Guardrail, Sandbox/Workspace, context-quality, discovery,
   and online-evaluation ports. They are opt-in unless a ToolSpec or production
   profile explicitly requires them.
5. If using MCP, certify the modern server path and verify task, header, and
   `outputSchema` behavior. Legacy fallback remains bounded and observable.
6. Treat `a2a_sdk` as experimental and pin the DeepKeel minor version when using
   it. Parent runs retain final-answer ownership.
7. Run the public API snapshot, adapter conformance, fault-injection, PostgreSQL,
   distribution, and downstream Capability Pack tests before promotion.

No migration is required for `RuntimeRequest`, `RuntimeResult`, RuntimeScope,
Artifact, Observation, PendingAction, or `harness-core-v3` package manifests.
The new SDK symbols are additive; removing or renaming a v4 stable symbol still
requires a later major release.

## 中文迁移清单

DeepKeel 4.1 是增量兼容的稳定版本。包版本与公开 SDK 版本升级为 `4.1.0`，持久化
Capability Pack 和 Runtime 协议仍然是 `harness-core-v3`，已有运行状态不需要重写。

1. 将依赖固定到不可变标签 `v4.1.0`，并重新构建 wheel/sdist 消费环境。
2. 应用代码继续只从 `deepkeel.*_sdk` 公开模块导入。
3. 最小接入优先使用 `AgentHarness`，生产装配继续使用
   `HarnessRuntimeBuilder` 和显式 `RuntimePorts`。
4. Guardrail、Sandbox/Workspace、上下文质量、渐进式发现和在线评测均为增量能力；
   只有 ToolSpec 或生产策略明确要求时才会 fail-closed。
5. 使用 MCP 时验证现代协议的任务、Header 和 `outputSchema` 行为；旧协议回退仍然
   有界且可观测。
6. `a2a_sdk` 当前是实验层，使用时应固定 DeepKeel 次版本，最终答案仍由父运行负责。
7. 发布前执行公开 API、适配器契约、故障注入、PostgreSQL、制品安装和下游能力包
   回归。
