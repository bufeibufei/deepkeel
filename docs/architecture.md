# Architecture

DeepKeel is a product-neutral runtime kernel. A Host owns transport, identity,
durable databases, model credentials, product policy and user experience.
Capability Packages contribute versioned Skills, tools, artifacts, handoffs,
context contributors, MCP servers and SubAgents through the public Extension
SDK.

```text
Host application
  -> RuntimeRequest + RuntimePorts
  -> DeepKeel HarnessRuntime
       -> context planning and model routing
       -> governed model/tool loop
       -> interruption, resume and settlement
       -> typed events, artifacts and diagnostics
  -> RuntimeResult
Capability Packages -> versioned contributions -> RuntimeGeneration
```

LangGraph is an internal execution adapter for graph state and checkpoint-aware
control flow. DeepKeel's public contracts do not expose LangGraph types, so a
Host or Capability Package is not coupled to graph construction details.

The stable integration layers are `runtime_sdk`, `extension_sdk`,
`adapter_sdk`, `memory_sdk`, `mcp_sdk`, and `orchestration_sdk`.
Internal modules may change without compatibility guarantees.
