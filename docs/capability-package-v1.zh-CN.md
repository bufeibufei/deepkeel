# Capability Package V1

[English](capability-package-v1.md) | [简体中文](capability-package-v1.zh-CN.md)

## 开发流程

使用 `deepkeel pack init` 创建 Manifest-first 骨架，使用 `pack inspect` 查看规范化
契约，使用 `pack validate --factory` 通过公开 SDK 装配可执行 Package 并发现声明漂移。
发布认证会进一步通过 `certify_capability_package()` 执行场景评测。

Package Permission 在安装后治理 Tool 与 Resource，但不会 Sandbox 已导入的 Python。
导入第三方代码前，Host 必须应用[Capability Package 信任模型](capability-trust.zh-CN.md)。

Capability Package 是向 DeepKeel 增加业务行为的唯一受支持边界。Package 负责领域
Tool、Skill、Artifact、Handoff、Prompt、UI Projection Metadata 与 Eval Case；不得
导入 Host API、ORM、Web Framework 或 Core 私有模块。

## 必需文件

```text
my_capability/
├── manifest.json
├── package.py
├── eval_cases.py
└── tests/
    └── test_certification.py
```

`manifest.json` 是唯一事实来源。`package.py` 通过
`capability_pack_spec_from_manifest` 派生 `CapabilityPackSpec`，不支持在 Python
中重复声明 Tool 或 Skill。

多能力依赖任务可以在 Skill 中声明 `planning_policy`，支持 `disabled`、`allowed`、
`preferred`、`required`，并为 `max_steps`、`max_revisions`、
`max_parallel_steps`、`max_attempts_per_step` 设定上限。Planning 不会扩大 Skill
Allowlist；每个计划步骤仍须引用当前 Skill 已可见的 Tool。

Manifest 必须声明：

- Package Identity、语义版本、Core Contract 与 EntryPoint；
- Tool、Skill、Artifact Type、Handoff 与可选 MCP Provider；
- Governance Permission、逐 Tool Permission 映射与可移植 Budget Ceiling；
- State Schema 与 Resume-compatible Package Version；
- Package 所有的 Memory Namespace 与 UI Surface。

## 安装边界

Package 只能接收 `CapabilityInstallContext`。它注册公开 Extension Object，并返回与
Manifest 完全一致的 `CapabilityContribution`；不能获取 Host DB Session、修改其他
Package Registry 或创建第二个 Agent Loop。

所有写工具必须使用 Idempotency Key 并声明 Side-effect Policy。Tool Permission 必须
同时存在于 Manifest 和相应 `ToolSpec.runtime_policy.required_scopes` 中。

## 认证门禁

每次发布运行 `certify_capability_package`，以下任一条件不满足即失败：

1. Manifest、Package Spec、安装贡献、Schema、Permission 与 Handler 一致；
2. Install、Discovery、Disable、Enable、Upgrade、Rollback 与 Runtime Generation
   Resume Compatibility 均通过；
3. 提供覆盖 `tool_selection`、`argument_generation`、`task_completion`、`recovery`
   和 `answer_quality` 的可执行评测；
4. 所有评测都通过公开 `HarnessRuntime` API 执行。

使用依赖 Tool 的 Package 必须同时向认证 API 传入 `dependency_manifests` 与
`dependency_packs`。仅 Manifest 用于校验生命周期元数据，真实 Dependency Pack
提供装配时所需 Tool、Handler 与其他贡献。

Host 可以增加语义质量 Grader，但确定性契约检查始终是必需门禁，不能由 LLM 分数替代。

## 版本管理

每次行为或契约变化都要增加 Package Version；持久化 Package State 变化时增加
`state_schema_version`。新版本只有在旧版本出现在 `resume_compatible_versions`，
且 Schema 相同或声明了 Migration 时，才能恢复旧 Run。

Worker 执行不可变 `RuntimeGeneration`。安装新版本只影响新 Run；挂起 Run 只能在
兼容 Generation 上恢复。
