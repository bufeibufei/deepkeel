# 参与贡献

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md)

DeepKeel 是独立、产品无关的 Runtime。Core 改动不得导入 Host 应用模块、数据库模型、
Web Handler 或业务 Capability 实现。

提交改动前，在仓库根目录运行：

```powershell
.\scripts\verify.ps1
```

macOS 或 Linux 请使用 `uv` 运行 `.github/workflows/ci.yml` 中的等效命令。Pull
Request 必须以 `main` 为目标，官方 Tag 也只能从 `main` 创建。

公开 SDK 变化必须是有意行为，并同时更新 API Version、兼容说明、Public API Snapshot
与 Changelog。Runtime 行为变化必须包含包内测试；产品特有预期应放在消费仓库的 Host
Contract Test 中，而不是 Core Package。

新集成应通过 Port、`ToolProvider`、Capability Package 或版本化 SDK 模块进入。避免
向 Kernel 增加产品词汇或兼容别名。
