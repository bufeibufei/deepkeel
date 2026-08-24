# 发布流程

[English](releasing.md) | [简体中文](releasing.zh-CN.md)

DeepKeel 发布使用不可变 Git Tag、已验证 Wheel/Sdist 与同步的中英文文档。
`pyproject.toml`、`deepkeel.version`、README 安装命令、Changelog 与 Tag 必须指向同一版本。

## 验证层级

- Pull Request 在 Python 3.12 运行快速门禁：Lint、Typing、双语文档与确定性非
  PostgreSQL 测试；
- Main 运行 Python 3.12 至 3.14、Windows/Linux 完整矩阵，以及 Coverage、Concurrency、
  Clean Distribution 与 PostgreSQL Recovery Contract；
- Release Tag 在构建不可变 Artifact 前再次运行故障/恢复、PostgreSQL 和并发门禁。
  同一 Tag 的 GitHub Upload 可安全重试；PyPI 遇到已发布不可变版本会跳过。

## 发布门禁

1. 确认各公开 SDK 层没有重复 Symbol Owner，并审查 Public API Fingerprint 变化；
2. 运行 Ruff、mypy、完整测试、至少 80% Statement Coverage，以及 300 Request/32
   Worker 的确定性 Core Benchmark；
3. 在一次性数据库运行 PostgreSQL Contract 与 Multi-worker Recovery Baseline；
4. 构建 Wheel/Sdist，检查 Metadata 与 Clean-install Conformance，再生成 SPDX JSON
   SBOM 和 `SHA256SUMS`；
5. 校验两份 README、Semantic Tag Contract 与 Migration Note；
6. 创建不可变 Tag 与 GitHub Release 前先推送 Release Commit；
7. 下游 Host 固定精确 Tag 或 Commit，再执行 Capability Pack、Persistence、Frontend
   与 Deployment Regression。

下游 Host Candidate 必须在提交 Core Tree 后只构建一次 Wheel：

```powershell
uv run python scripts/build_release_candidate.py
```

命令会拒绝 Dirty Tree，并输出一个 Wheel 及 Sidecar Manifest，记录不可变 Source SHA/
Ref/Commit Time、Package Version、Build Environment、可选 Host SHA/Release ID、Build
Time 与 Wheel SHA256。协调发布可传 `--host-sha <sha> --release-id <id>`。Host 应在
隔离环境安装该 Wheel 并运行门禁，部署时不得重建，安装前再次校验 SHA256，并保留
Core 与 Host Provenance。

第三方 GitHub Action 必须固定完整 Commit SHA。PostgreSQL Adapter 有独立 Coverage
Gate，因为不带 `postgres` Marker 的包级 Coverage 不会执行这些路径。

Windows 发布门禁：

```powershell
$env:DEEPKEEL_TEST_POSTGRES_DSN = "postgresql://..."
./scripts/verify.ps1
uv run python -m verification.release_contract --tag v4.1.0
uv run python -m verification.readme_contract
```

禁止为 Dirty Tree 打 Tag、移动已有 Release Tag，或发布仍由下游 Host 导入私有
`deepkeel.*` 实现模块的版本。Release Artifact 是输出，不提交到源码树。
