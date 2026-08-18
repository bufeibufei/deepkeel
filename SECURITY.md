# Security Policy

## Reporting

Do not publish credentials, prompts, model payloads, tenant data or exploitable
details in a public issue. Report suspected vulnerabilities privately to
`bufeibufei@users.noreply.github.com` with the affected version, impact and a
minimal reproduction that contains no real user data.

Supported versions are listed in the latest GitHub release. Release candidates
receive fixes only while they are under active compatibility validation.

## Runtime Boundary

DeepKeel treats model output, tool arguments, MCP responses,
checkpoints and Capability Pack metadata as untrusted input. Security fixes must
preserve policy checks, schema validation, redaction, tenant boundaries and
budget enforcement. Telemetry must not include prompts, tool payloads or model
results by default.

Guardrails are defense-in-depth controls, not a replacement for Host
authorization. Required Guardrails fail closed on timeout or handler failure;
optional Guardrails may fail open only when their policy explicitly permits it.
ToolSpec sandbox requirements are enforced before handler execution, and a
non-enforcing sandbox cannot satisfy a required policy.

Remote MCP and A2A endpoints are untrusted egress. Production Hosts should use
hostname allowlists, scoped secrets, bounded timeouts and response sizes, and
must not enable private-network access for arbitrary user-controlled endpoints.
Release artifacts include provenance attestations and an SPDX JSON SBOM; verify
both before promoting an artifact into a production environment.
