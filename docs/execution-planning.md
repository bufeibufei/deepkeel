# Execution planning

[English](execution-planning.md) | [简体中文](execution-planning.zh-CN.md)

DeepKeel supports an optional Plan & Execute layer inside its canonical ReAct
graph. It is intended for requests that require multiple capabilities,
dependency ordering, bounded parallel evidence collection, durable user
handoffs, or explicit result synthesis. It is not a second runtime and should
not be used for greetings, direct answers, or one-tool lookups.

## Composition and Skill policy

The Host enables the internal planning control tool with
`RuntimePorts(planning_enabled=True)`. Each active Skill then declares a
`planning_policy`:

```json
{
  "mode": "preferred",
  "max_steps": 8,
  "max_revisions": 2,
  "max_parallel_steps": 4,
  "max_attempts_per_step": 2
}
```

`disabled` hides the control tool, `allowed` lets the model choose it,
`preferred` instructs the model to plan genuinely multi-step work, and
`required` forces the first model transition through the planning control tool.
The global composition switch remains authoritative: a Skill cannot enable a
control tool that the Host did not install.

## Plan contract

An `ExecutionPlan` is a versioned, durable DAG. Every `PlanStep` has a stable
identity, objective, executor kind, capability reference, arguments,
dependencies, success criteria, bounded attempts, execution status, and result
projection. Executable steps reference ordinary registered tools. Workflows and
SubAgents therefore use their existing tool boundary rather than bypassing
governance.

Before adoption, Core verifies:

- plan and revision bounds;
- unique step identities and acyclic dependencies;
- tool existence and active Skill allowlists;
- runtime-control tool isolation;
- completed-step immutability across revisions;
- read-only and parallel-safety metadata from the authoritative ToolSpec.

## Scheduling

Core schedules only dependency-ready steps. Independent read-only,
parallel-safe tools may run as one bounded batch. Side-effecting tools,
suspending tools, asynchronous workflows, and synthesis run serially. Every
generated tool call carries stable plan, revision, step, attempt, idempotency,
and resource identities, then passes through the existing ToolExecutor, Policy,
Budget, Hook, checkpoint, and event paths.

A retryable failure may repeat a step within its attempt limit. A terminal
failure returns control to the model for a bounded plan revision or an honest
partial answer. Completed steps cannot be removed or rewritten by replanning.

## Interruption and recovery

The active plan is present in both LangGraph state and the portable runtime
checkpoint. A user-action or asynchronous tool result marks its plan step as
waiting. Resume resolves that exact step, schedules newly ready dependents, and
does not replay completed work. The same behavior applies to live LangGraph
resume and cross-worker restoration from a durable checkpoint.

## Events and presentation

Plan lifecycle events use the `plan.*` namespace, including validation, start,
step start/completion/wait/retry/failure, revision, synthesis, completion, and
partial completion. Payloads expose objective, progress, step identity,
capability reference, attempts, and status. They intentionally exclude hidden
chain-of-thought. Hosts should render a compact progress projection and keep
the complete event history in their trace or debug surface.
