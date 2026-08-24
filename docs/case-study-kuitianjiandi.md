# Case study: Kuitianjiandi Host

Kuitianjiandi is the original downstream Host from which DeepKeel's reusable
runtime contracts were extracted. It is a conversational application that lets a
root agent coordinate deterministic domain calculations, retrieval, long-running
generation, user handoffs, specialist agents, and mobile result surfaces.

This document describes architectural validation only. It contains no user data,
credentials, production addresses, private prompts, or domain conclusions.

## Why this Host is a useful runtime test

The application began as a conventional product-specific agent loop. Real usage
exposed problems that were not solved by adding more prompt instructions:

- a model response finished, but the composer stayed locked;
- a browser refresh lost pending actions or result cards;
- a background report completed, while chat still showed it as running;
- a stale Worker could race a resumed run;
- tool and Skill catalogs became too large to expose in every turn;
- frontend components inferred state from prose and drifted from backend state;
- temporary analysis and persistent user identity needed different scopes;
- provider retries and tool replay risked duplicate work;
- traces, logs, metrics, and product events disagreed about one run.

These failures became product-neutral DeepKeel contracts rather than permanent
special cases in the Host.

## Product-to-runtime mapping

| Kuitianjiandi concern | Host or package responsibility | DeepKeel responsibility |
| --- | --- | --- |
| Browser chat, login, SSE, mobile UI | Host | Typed request/result/event contracts and `ui_state` projection |
| Birth profiles and user identity | Host database and policy | `RuntimeScope`, context authority, and fail-closed identity checks |
| Bazi, Liuyao, date selection, naming | Capability Packages | Package lifecycle, Skill/tool visibility, artifacts, and handoffs |
| Long report generation | Domain worker | Async suspension, portable checkpoint, observation, resume, and settlement |
| Classical-text retrieval and web search | Package tools or MCP | Governed tool gateway, permissions, budgets, schema validation, and trace |
| Provider-specific model APIs | Host model adapter | Model roles, routing policy, health, retry accounting, and output contracts |
| Result cards and detail pages | Host frontend renderers | Versioned `Artifact` and `ArtifactView` projection |
| Run debugging | Host operations UI and OTel stack | Ordered events, diagnostics, failure taxonomy, and OTel projection |

## Example lifecycle: long-running domain report

1. The Host sends a `RuntimeRequest` with a stable `RuntimeScope`, conversation,
   run identity, and selected agent entrypoint.
2. Core resolves the immutable `CapabilityView`. The model sees only descriptors
   allowed for that user, entrypoint, policy, and current activation.
3. A domain tool validates structured input. Missing information becomes a typed
   clarification or `PendingAction`, not a red failure card.
4. After deterministic preparation, a delegated task is persisted by the Host
   and the run enters an asynchronous waiting state.
5. The domain Worker claims the task with owner and generation semantics. A stale
   Worker cannot settle work after its lease or fence is replaced.
6. The completed task returns a bounded `Observation` and versioned result
   `Artifact`. Core resumes the original run and gives the artifact to the model.
7. The root agent produces the final answer. The frontend renders the artifact
   separately and unlocks input from canonical `ui_state`, not from text matching.
8. Runtime events project to SSE, the durable journal, OpenTelemetry, and compact
   diagnostics with the same run and event identities.

## Capability topology

Kuitianjiandi uses one general root agent plus domain-specific entrypoints. A
specialist entrypoint is not another runtime process: it is an immutable narrowed
view of the same installed `RuntimeGeneration` and canonical graph.

Capability Packages contribute domain behavior such as:

- deterministic chart and rule calculations;
- long-running report workflows;
- interactive divination handoffs;
- date-selection and naming workflows;
- retrieval, citation, and MCP-backed search;
- typed result artifacts and presentation metadata;
- specialist prompts, context contributors, and SubAgent definitions.

The packages cannot import Host API handlers or ORM models. They receive stable
SDK contracts and Host-injected services through the installation and execution
contexts.

## What stayed in the Host

DeepKeel deliberately did not absorb:

- account registration, authentication, and authorization endpoints;
- PostgreSQL connection ownership and product migrations;
- product-specific profile schemas or record lists;
- browser routing, React state, visual design, and card components;
- model credentials and vendor account configuration;
- domain prompts, rule engines, retrieval corpora, and answer-quality criteria;
- deployment topology and the Grafana, Tempo, Loki, and Prometheus environment.

This boundary is what makes the Core reusable. The Host can redesign its product
without forking the runtime, while runtime reliability improvements can be
released without importing the product.

## Validation contributed back to DeepKeel

The Host drove concrete runtime requirements now covered by package-owned tests:

- canonical status wins over stale frontend or graph projections;
- interruption and resume preserve run, turn, scope, and capability generation;
- answer streaming does not lose text under bounded backpressure;
- terminal states settle once and release the composer;
- tool and model invocations are replay-safe;
- package, user, tenant, and namespace state cannot cross scope boundaries;
- stale workers fail execution-fence checks;
- capability discovery cannot bypass permission filtering;
- result artifacts survive refresh and can be projected independently of chat;
- downstream Host compatibility runs against the installed candidate package.

The detailed commands and owning workflows are listed in the
[verification matrix](verification-matrix.md).

## Lessons for another industrial Host

The reusable part is not the domain prompt. It is the boundary around work:

1. Model uncertainty belongs in planning and interpretation; identity, authority,
   side-effect settlement, and lifecycle state should be deterministic.
2. A long task needs a durable protocol, not a longer HTTP timeout.
3. A tool result should become an observation and artifact, not an ad-hoc block of
   UI-specific prose.
4. Capability scale requires permission-first retrieval and measurable selection,
   not a larger system prompt.
5. The same event identity should connect product state, logs, traces, metrics,
   retries, and user-visible progress.

Those principles are domain-neutral and form the practical value of DeepKeel.
