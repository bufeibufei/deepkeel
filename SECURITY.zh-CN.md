# 安全策略

[English](SECURITY.md) | [简体中文](SECURITY.zh-CN.md)

## 漏洞报告

请勿在公开 Issue 中发布 Credential、Prompt、Model Payload、Tenant Data 或可利用
细节。请将疑似漏洞私下发送至 `bufeibufei@users.noreply.github.com`，附上受影响版本、
影响范围与不含真实用户数据的最小复现。

当前支持版本以最新 GitHub Release 为准。Release Candidate 只在主动兼容验证期间接收
修复。

## Runtime 安全边界

DeepKeel 将模型输出、Tool Argument、MCP Response、Checkpoint 与 Capability Pack
Metadata 视为不可信输入。安全修复必须保留 Policy Check、Schema Validation、
Redaction、Tenant Boundary 与 Budget Enforcement。Telemetry 默认不得包含 Prompt、
Tool Payload 或模型结果。

Guardrail 是纵深防御，不能替代 Host Authorization。Required Guardrail 超时或 Handler
失败时必须 Fail-closed；Optional Guardrail 只有在 Policy 明确允许时才能 Fail-open。
ToolSpec Sandbox Requirement 在 Handler 前执行，Non-enforcing Sandbox 不能满足
Required Policy。

Remote MCP 与 A2A Endpoint 属于不可信 Egress。生产 Host 应使用 Hostname Allowlist、
Scoped Secret、有界 Timeout 与 Response Size，且不得为任意用户控制 Endpoint 开启
Private-network Access。Release Artifact 包含 Provenance Attestation 与 SPDX JSON
SBOM，提升到生产前应同时验证二者。
