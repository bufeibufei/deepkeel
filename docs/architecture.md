# Architecture

DeepKeel is a product-neutral runtime kernel. A Host owns transport, identity,
durable databases, model credentials, product policy and user experience.
Capability Packages contribute versioned Skills, tools, artifacts, handoffs,
context contributors, MCP servers and SubAgents through the public Extension
SDK. Optional MCP and A2A adapters remain at the interoperability edge and are
projected into the same governed ToolProvider or SubAgent contracts.

```text
Host application
  -> RuntimeRequest + RuntimePorts
  -> DeepKeel HarnessRuntime
       -> context planning and model routing
       -> governed model/tool loop with optional bounded execution plans
       -> interruption, resume and settlement
       -> typed events, artifacts and diagnostics
  -> RuntimeResult
Capability Packages -> versioned contributions -> RuntimeGeneration
```

A Host may expose multiple user-facing root Agents from one immutable
`RuntimeGeneration`. A Capability Package declares an `AgentEntrypointSpec`,
and each conversation resolves it to an immutable `CapabilityView`:

```text
installed RuntimeGeneration
  intersect AgentEntrypointSpec package/dependency scope
  intersect active Skill policy
  intersect progressive tool disclosure
  intersect runtime Policy and Budget decisions
  -> effective per-conversation CapabilityView
```

The entrypoint selects a scoped view of the same compiled graph; it does not
build a second runtime. Internal SubAgents remain bounded child tasks and may
only narrow their parent's effective scope.

LangGraph is an internal execution adapter for graph state and checkpoint-aware
control flow. DeepKeel's public contracts do not expose LangGraph types, so a
Host or Capability Package is not coupled to graph construction details.

The integration layers are `runtime_sdk`, `extension_sdk`, `adapter_sdk`,
`memory_sdk`, `mcp_sdk`, `orchestration_sdk`, and experimental `a2a_sdk`.
Internal modules may change without compatibility guarantees.

Each public symbol has exactly one owner:

- `runtime_sdk`: requests, results, state, scope, lifecycle, and execution;
- `extension_sdk`: Capability Packages, Skills, tools, artifacts, and handoffs;
- `adapter_sdk`: Host infrastructure ports, conformance suites, and composition;
- `memory_sdk`: product-neutral memory records and memory ports;
- `mcp_sdk`: governed MCP server and transport integration;
- `orchestration_sdk`: bounded execution-plan, SubAgent, and deliberation contracts;
- `a2a_sdk`: experimental A2A 1.0 remote-specialist interoperability.

`RuntimePorts` remains a flat compatibility contract, while infrastructure
authors can compose it from four cohesive bundles:

- `RuntimePersistencePorts` for state, checkpoints, journals, and replay;
- `RuntimeGovernancePorts` for policy, budget, health, and control;
- `RuntimeObservabilityPorts` for telemetry, traces, and evaluation evidence;
- `RuntimeExecutionPorts` for model, tool, context, reference, and secret edges.

The implementation is deliberately split behind those public facades. Runtime
turn coordination delegates to execution and failure helpers; model and tool
gateways delegate provider calls and settlement; Graph nodes delegate model
steps; context and SubAgent execution use bounded support modules. An automated
AST import-graph test rejects internal dependency cycles, and structural
ratchets prevent these facades from becoming monoliths again.
