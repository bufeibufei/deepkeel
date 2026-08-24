# 供应链治理

[English](supply-chain.md) | [简体中文](supply-chain.zh-CN.md)

DeepKeel 从不可变语义版本 Tag 构建 Release。发布 Workflow 会校验 Tag 与
`pyproject.toml`、`deepkeel.version` 一致，运行生产回归矩阵，只构建一次 Wheel 与
Sdist，在干净环境验证二者，并发布完全相同的 Artifact。

## 发布证据

每个 GA Release 应提供：

- Metadata 一致的 Wheel 与 Source Distribution；
- SPDX JSON 软件物料清单；
- 覆盖全部上传 Artifact 的确定性 `SHA256SUMS`；
- 所有 Release Artifact 的 GitHub Build-provenance Attestation；
- 冻结的公开 API 与 Semantic-contract Fingerprint；
- 同步的英文与简体中文 README 及其关联文档；
- 对应 Tag 的 Changelog 与 Migration Note；
- 通过确定性、故障注入、并发、PostgreSQL 与 Clean-install Conformance Gate。

PyPI 使用 OIDC Trusted Publishing，不保存长期 Upload Token。已有不可变 Tag 的 GitHub
Release Job 必须幂等，只能替换上传 Asset，不能移动 Source Tag。

## 消费方验证

1. 只从带 Tag 的 GitHub Release 或已配置 PyPI Project 下载 Artifact；
2. 根据 Repository 与 Workflow 验证 GitHub Artifact Attestation；
3. 安装前逐项核对 `SHA256SUMS`；
4. 检查 SBOM 并应用组织漏洞策略；
5. 生产 Lockfile 固定精确版本或不可变 Commit；
6. 针对已安装 Wheel 运行 Host Adapter 与 Capability Pack 测试。

SBOM 描述组成，不代表不存在漏洞；Provenance 说明构建来源，不能替代 Code Review、
Dependency Policy 或 Runtime Isolation。
