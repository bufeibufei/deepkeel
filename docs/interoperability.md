# MCP and A2A interoperability

[English](interoperability.md) | [简体中文](interoperability.zh-CN.md)

DeepKeel keeps interoperability protocols outside the runtime kernel. MCP tools
become governed `ToolSpec` entries, while A2A remote Agents become bounded
`SubAgentSpec` entries. Neither protocol can bypass the parent run's policy,
budget, event, checkpoint, cancellation, or Artifact contracts.

## MCP

`deepkeel.mcp_sdk` supports local stdio and Streamable HTTP. New connections
prefer the modern `2026-07-28` protocol era and use a bounded compatibility
fallback for declared legacy versions.

The modern path provides:

- stateless `server/discover` negotiation and per-request protocol metadata;
- cacheable, paginated tool discovery;
- MCP task creation, polling, update, cancellation, and input continuation;
- safe `Mcp-Name` and `Mcp-Param-*` headers;
- `x-mcp-header` validation on statically reachable primitive properties;
- server `outputSchema` validation before data becomes a Runtime Observation;
- typed, sanitized remote errors and bounded request/response payloads.

Invalid tools are isolated during discovery instead of poisoning the full
catalog. Non-ASCII or whitespace-sensitive header values use the MCP Base64
sentinel form. Missing or null values are not mirrored.

## A2A

`deepkeel.a2a_sdk` is an experimental A2A 1.0 adapter. `A2ARemoteAgent` maps an
Agent Card to the existing SubAgent registry. `A2ADelegationExecutor` sends a
Message, accepts either a direct Message or a Task, polls bounded work, projects
remote Artifacts, and maps input/auth requirements into typed pending actions.

Remote task identity is checkpointed so a worker restart resumes polling rather
than submitting duplicate work. Parent cancellation propagates to the remote
task. The parent Agent owns synthesis and the final answer; a remote Agent is an
observation-producing specialist, not an independent authority over the user
conversation.

## Choosing the boundary

Use MCP when the remote system exposes operations or resources that fit a tool
call. Use A2A when the remote system owns a multi-step specialist task with its
own lifecycle and Artifacts. Use an in-process Capability Package when the code
and trust boundary are local. All three paths still converge on DeepKeel's
governed runtime contracts.
