# Security Policy

## Reporting

Do not publish credentials, prompts, model payloads, tenant data or exploitable
details in a public issue. Report suspected vulnerabilities privately to
`bufeibufei@users.noreply.github.com` with the affected version, impact and a
minimal reproduction that contains no real user data.

## Runtime Boundary

Harness Agent Core treats model output, tool arguments, MCP responses,
checkpoints and Capability Pack metadata as untrusted input. Security fixes must
preserve policy checks, schema validation, redaction, tenant boundaries and
budget enforcement. Telemetry must not include prompts, tool payloads or model
results by default.
