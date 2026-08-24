# 迁移到 DeepKeel 4.1

[English](migrating-to-4.1.md) | [简体中文](migrating-to-4.1.zh-CN.md)

DeepKeel 4.1 是增量兼容的 GA 版本。Package 与公开 SDK Version 升级为 `4.1.0`，
持久化 Capability Pack 和 Runtime Contract 仍为 `harness-core-v3`，已有 Durable Run
State 不需要 Schema Rewrite。

## 迁移清单

1. 将依赖固定到不可变 Tag `v4.1.0`，并重新构建 Wheel/Sdist 消费环境；
2. 应用代码继续只从版本化 `deepkeel.*_sdk` 公开模块导入；
3. 最小接入优先使用 `AgentHarness`，显式生产装配继续使用
   `HarnessRuntimeBuilder`；
4. 审查新增 Guardrail、Sandbox/Workspace、Context Quality、Discovery 与 Online Eval
   Port；除非 ToolSpec 或 Production Profile 明确要求，否则它们是 Opt-in；
5. 使用 MCP 时认证现代 Server Path，并验证 Task、Header 与 `outputSchema`；Legacy
   Fallback 仍有界且可观测；
6. `a2a_sdk` 属于 Experimental，使用时固定 DeepKeel Minor Version；Final Answer
   所有权仍属于父 Run；
7. 发布前运行 Public API Snapshot、Adapter Conformance、Fault Injection、PostgreSQL、
   Distribution 与下游 Capability Pack Test。

`RuntimeRequest`、`RuntimeResult`、`RuntimeScope`、`Artifact`、`Observation`、
`PendingAction` 与 `harness-core-v3` Package Manifest 均无需迁移。新 SDK Symbol 是
增量能力；删除或重命名 v4 Stable Symbol 仍需要后续 Major Release。
