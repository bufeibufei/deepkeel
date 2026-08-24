# Capability Package 信任模型

[English](capability-trust.md) | [简体中文](capability-trust.zh-CN.md)

Capability Manifest 声明的是 Runtime Permission，不是 Python Sandbox。导入 Python
Capability Pack 会在 Host 进程中执行代码，因此 DeepKeel 区分两种部署模式：

- `trusted_in_process`：Host 在导入前校验 Allowlist 中的 SHA-256 Digest；本地开发
  可以显式选择未验证 Package；
- `isolated`：实现运行在 MCP 或 A2A 后方，Host 对 Endpoint Prefix 做 Allowlist，
  常规 Egress、认证和 Policy 控制仍然生效。

`CapabilityPackageSource`、`CapabilityTrustPolicy` 与
`evaluate_capability_trust()` 构成可移植决策契约。Host 必须在导入不可信
EntryPoint 前执行该检查。建立来源信任后，Package Permission、Tool Policy 和
Runtime Guardrail 仍然是强制边界；来源可信不等于拥有业务权限。

CLI 支持本地开发流程：

```bash
deepkeel pack init ./my_pack --package-id company.my-pack
deepkeel pack inspect ./my_pack/manifest.json
deepkeel pack digest ./my_pack/manifest.json ./my_pack/package.py
deepkeel pack validate ./my_pack/manifest.json \
  --factory my_pack.package:MyPack
```
