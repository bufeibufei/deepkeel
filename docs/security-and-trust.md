# Security and trust

DeepKeel treats user input, model output, tool arguments, tool results, MCP
responses, A2A payloads, checkpoints, and Capability Package metadata as
untrusted data. Security policy is enforced by typed runtime boundaries rather
than prompt text alone.

## Guardrail pipeline

`GuardrailRunner` executes ordered policies at input, model-output, and tool
boundaries. Each decision can allow, transform, redact, require confirmation,
or deny an operation. Required Guardrails fail closed when they time out or
raise; optional Guardrails follow their declared failure policy. Decisions are
keyed by guardrail, stage, and operation so a durable Host store can replay them
without repeating an external moderation call.

Guardrails complement, but never replace, Host authentication, tenant
authorization, ToolSpec permission checks, budget policy, and schema validation.
Audit sinks should persist decision metadata and digests, not raw sensitive
payloads.

## Provenance and external content

`DataProvenance` labels trusted configuration, user-provided content, model
output, retrieved evidence, and external tool data. External content remains
data and is never promoted into system instructions. Context quality checks can
quarantine subject or authority mismatches before model invocation.

## Sandbox and workspace ports

A ToolSpec can declare a required sandbox, execution limits, network policy,
and workspace requirements. Core acquires those resources before invoking the
handler, injects only typed lease metadata, and releases them on every terminal
path. If a required adapter is absent or reports `enforced=False`, the handler
does not run.

`NoopSandboxPort` and `LocalWorkspacePort` are development adapters. Production
Hosts should provide an OS, container, VM, or remote-execution adapter that
actually enforces wall-time, CPU, memory, process, output, filesystem, and
network boundaries. Workspace roots must be tenant-scoped and cleanup must
reject paths outside the allocated root.

## Remote interoperability

MCP and A2A endpoints use egress controls, scoped secrets, bounded payloads,
timeouts, protocol validation, and sanitized diagnostics. Private-network
access is denied by default for Streamable HTTP MCP. Apply equivalent allowlist
and DNS-rebinding controls to custom A2A clients.

## Host checklist

1. Authorize the `RuntimeScope` before constructing a request.
2. Keep secrets behind `SecretProvider`; never place them in context or events.
3. Use enforced Tool/Skill disclosure and deny unknown capabilities.
4. Configure durable replay, state, lease, and audit stores for multi-worker use.
5. Require confirmation for external writes and irreversible operations.
6. Export only privacy-bounded telemetry and online-evaluation samples.
7. Verify the release provenance and SBOM before deployment.
