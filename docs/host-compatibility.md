# Downstream Host compatibility

[English](host-compatibility.md) | [简体中文](host-compatibility.zh-CN.md)

DeepKeel releases are not complete when Core tests pass in isolation. A Host
should keep a small public-SDK-only contract suite covering its real provider,
Capability Packages, persistence adapters, SSE projection, cancellation and
resume behavior.

The reusable `Downstream Host Compatibility` workflow checks out the candidate
DeepKeel revision, installs it over the downstream Host environment, and runs a
Host-owned test command. Configure it manually before a release or call it from
the Host workflow. Private repositories need a checkout credential with access
to both repositories.

The downstream suite should fail if it imports private `deepkeel.*` modules,
depends on LangGraph types, reads full diagnostics where `to_summary()` is
sufficient, or installs an in-process third-party package without a trust
decision. This keeps the Host boundary executable instead of relying on an
architecture diagram alone.
