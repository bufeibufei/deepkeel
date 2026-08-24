# Model providers

[English](model-provider.md) | [简体中文](model-provider.zh-CN.md)

Model access is a Port. Providers describe model identity, role, context limits,
structured-output support and usage. Routing can select fast, reasoning,
structured, embedding or rerank roles without exposing vendor-specific clients
to the runtime loop.

`single` policy always uses the role selected by the Host or user. `adaptive`
policy evaluates every model step independently. Before routing, Core builds a
sanitized candidate profile containing declared capabilities, context-window
fit, shared health state, estimated input size, current budget usage and
Host-declared latency/cost tiers. Unhealthy or capability-incompatible roles
are excluded; phase semantics then choose between eligible fast and reasoning
roles. The complete decision evidence is emitted in `model.route.selected`, so
automatic routing never becomes an invisible override.

Every attempt is recorded through a durable invocation envelope. Retry and
fallback are bounded by policy and budget, and replay settlement prevents an
uncertain provider response from silently producing duplicate work.

Secrets and provider catalogs belong to the Host. DeepKeel diagnostics expose
sanitized identity, capability, latency and usage metadata rather than prompts
or model payloads.
