# Model providers

Model access is a Port. Providers describe model identity, role, context limits,
structured-output support and usage. Routing can select fast, reasoning,
structured, embedding or rerank roles without exposing vendor-specific clients
to the runtime loop.

Every attempt is recorded through a durable invocation envelope. Retry and
fallback are bounded by policy and budget, and replay settlement prevents an
uncertain provider response from silently producing duplicate work.

Secrets and provider catalogs belong to the Host. DeepKeel diagnostics expose
sanitized identity, capability, latency and usage metadata rather than prompts
or model payloads.
