from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest
import harness_core
import harness_core.adapter_sdk as adapter_sdk
import harness_core.extension_sdk as extension_sdk
import harness_core.mcp_sdk as mcp_sdk
import harness_core.memory_sdk as memory_sdk
import harness_core.orchestration_sdk as orchestration_sdk
import harness_core.runtime_sdk as runtime_sdk

from harness_core.runtime_sdk import (
    Artifact,
    InMemoryRuntimeStateStore,
    PendingAction,
    RuntimeRequest,
    RuntimeStateConflict,
    RuntimeStateMutation,
    ToolCall,
    ToolResult,
)
from harness_core.extension_sdk import (
    ArtifactTypeSpec,
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityPackSpec,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolProviderSpec,
    ToolSpec,
    validate_capability_pack,
)
from harness_core.adapter_sdk import (
    BudgetPolicy,
    BudgetRequest,
    ContextSegment,
    ContextWindowPolicy,
    DeterministicContextWindowManager,
    HarnessRuntimeBuilder,
    InMemoryBudgetLedger,
    InMemoryContextSummaryCache,
    InMemoryTelemetry,
    InMemoryModelInvocationRecorder,
    ModelInvocation,
    ModelProviderInfo,
    ModelTurn,
    RuntimePorts,
    TelemetryRecord,
    UsageReport,
)
from harness_core.handoffs import HandoffSpec
from harness_core.subagents import SubAgentSpec
from harness_core.public_api import PUBLIC_API_BY_LAYER, PUBLIC_API_SYMBOLS, PUBLIC_API_VERSION


def test_package_root_only_exposes_versioned_sdk_entrypoints() -> None:
    assert PUBLIC_API_VERSION == "3.8.0"
    assert tuple(harness_core.__all__) == (
        "HARNESS_CORE_CONTRACT_VERSION",
        "HARNESS_CORE_VERSION",
        "adapter_sdk",
        "extension_sdk",
        "mcp_sdk",
        "memory_sdk",
        "orchestration_sdk",
        "runtime_sdk",
    )
    root_runtime_symbols = {
        "HARNESS_CORE_CONTRACT_VERSION",
        "HARNESS_CORE_VERSION",
    }
    assert (set(harness_core.__all__) & PUBLIC_API_SYMBOLS) == root_runtime_symbols


def test_public_api_matches_the_frozen_v3_snapshot() -> None:
    serialized = json.dumps(
        PUBLIC_API_BY_LAYER,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    actual = hashlib.sha256(serialized).hexdigest()
    expected = (
        Path(__file__).with_name("public_api_v3.sha256").read_text(encoding="ascii").strip()
    )

    assert actual == expected, (
        "public API changed; review compatibility and update the v3 snapshot only "
        "for an intentional contract release"
    )


@pytest.mark.parametrize(
    "module,layer",
    [
        (runtime_sdk, "runtime"),
        (extension_sdk, "extension"),
        (orchestration_sdk, "orchestration"),
        (mcp_sdk, "mcp"),
        (memory_sdk, "memory"),
        (adapter_sdk, "adapter"),
    ],
)
def test_sdk_module_exports_match_the_versioned_manifest(module, layer: str) -> None:
    expected = tuple(PUBLIC_API_BY_LAYER[layer])

    assert tuple(module.__all__) == expected
    assert all(hasattr(module, symbol) for symbol in expected)


def run_runtime(runtime, question: str, **kwargs):
    request = RuntimeRequest(
        question=question,
        user_id=str(kwargs.pop("user_id", "local-device")),
        short_context=dict(kwargs.pop("short_context", {})),
        context_bundle=dict(kwargs.pop("context_bundle", {})),
        skill_activation=dict(kwargs.pop("skill_activation", {})),
        model_policy=dict(kwargs.pop("model_policy", {})),
    )
    return runtime.run(request, **kwargs)


class EchoModelAdapter:
    info = ModelProviderInfo(
        provider_id="example.echo",
        model_id="echo-v1",
        model_role="fast",
    )

    def invoke(self, request: ModelInvocation, *, on_text_delta=None) -> ModelTurn:
        assert request.messages[-1]["role"] == "user"
        if on_text_delta is not None:
            on_text_delta("hello ")
            on_text_delta("back")
        return ModelTurn(
            content="hello back",
            finish_reason="stop",
            model_id=self.info.model_id,
            model_role=self.info.model_role,
        )


@dataclass(frozen=True, slots=True)
class InventoryPack:
    spec = CapabilityPackSpec(
        package_id="example.inventory",
        package_version="1.2.0",
        declared_tools=("inventory.lookup",),
        declared_skills=("inventory-query",),
        required_scopes=("inventory.read",),
    )

    def install(
        self,
        context: CapabilityInstallContext,
    ) -> CapabilityContribution:
        context.register_tool(
            ToolSpec(
                name="inventory.lookup",
                parameters_schema={
                    "type": "object",
                    "properties": {"item": {"type": "string"}},
                    "required": ["item"],
                    "additionalProperties": False,
                },
                required_args=["item"],
                output_schema={
                    "type": "object",
                    "properties": {"quantity": {"type": "integer"}},
                    "required": ["quantity"],
                },
                runtime_policy={"required_scopes": ["inventory.read"]},
            ),
            self._lookup,
        )
        context.register_skill("inventory-query", {"label": "Inventory query"})
        return CapabilityContribution(
            package_id=self.spec.package_id,
            tools=self.spec.declared_tools,
            skills=self.spec.declared_skills,
        )

    @staticmethod
    def _lookup(_call: ToolCall, _context: ToolExecutionContext) -> dict:
        return {"status": "ok", "result": {"quantity": 3}}


def test_public_sdk_installs_declarative_capability_pack() -> None:
    runtime = HarnessRuntimeBuilder().add_capability_pack(InventoryPack()).build()

    assert runtime.tool_registry.get("inventory.lookup").name == "inventory.lookup"
    assert runtime.tool_executor.registered_tools == frozenset({"inventory.lookup"})
    assert runtime.capability_contributions[0].skills == ("inventory-query",)


def test_conformance_uses_pack_declaration_without_duplicate_arguments() -> None:
    report = validate_capability_pack(InventoryPack())

    assert report.passed is True
    assert report.package_version == "1.2.0"
    assert report.declared_tools == ["inventory.lookup"]
    assert report.invalid_tool_contracts == []


def test_conformance_rejects_write_tool_marked_parallel_safe() -> None:
    @dataclass(frozen=True, slots=True)
    class UnsafePack:
        spec = CapabilityPackSpec(
            package_id="example.unsafe",
            declared_tools=("unsafe.write",),
        )

        def install(self, context: CapabilityInstallContext) -> CapabilityContribution:
            context.register_tool(
                ToolSpec(
                    name="unsafe.write",
                    parameters_schema={"type": "object", "properties": {}},
                    read_only=False,
                    parallel_safe=True,
                ),
                lambda *_args: {"status": "ok"},
            )
            return CapabilityContribution(
                package_id=self.spec.package_id,
                tools=("unsafe.write",),
            )

    report = validate_capability_pack(UnsafePack())

    assert report.passed is False
    assert report.invalid_tool_contracts == [
        "unsafe.write: write tools cannot be parallel_safe"
    ]


def test_runtime_ports_reject_unknown_configuration_keys() -> None:
    with pytest.raises(TypeError, match="unknown runtime ports: database"):
        HarnessRuntimeBuilder().configure_ports(database=object())


def test_v1_capability_pack_is_explicitly_rejected() -> None:
    @dataclass(frozen=True, slots=True)
    class LegacyPack:
        package_id = "example.legacy"
        contract_version = "harness-core-v1"
        tool_names = ("legacy.read",)

        def register(self, executor: ToolExecutor) -> None:
            executor.registry.register(
                ToolSpec(
                    name="legacy.read",
                    parameters_schema={"type": "object", "properties": {}},
                )
            )
            executor.register("legacy.read", lambda *_args: {"status": "ok"})

    registry = ToolRegistry()
    with pytest.raises(TypeError, match="must declare a CapabilityPackSpec"):
        HarnessRuntimeBuilder(registry).add_capability_pack(LegacyPack())

    assert registry.list_tools() == []


def test_runtime_ports_dataclass_remains_directly_constructible() -> None:
    assert RuntimePorts().checkpoint_store is None


def test_explicit_model_adapter_and_telemetry_port_run_without_legacy_reflection() -> None:
    telemetry = InMemoryTelemetry()
    runtime = (
        HarnessRuntimeBuilder()
        .with_ports(RuntimePorts(telemetry=telemetry))
        .build()
    )

    result = run_runtime(
        runtime,
        "hello",
        provider=EchoModelAdapter(),
        context_bundle={"agent_session_id": "run-explicit-adapter"},
    )

    assert result.final_answer.markdown == "hello back"
    events = telemetry.snapshot()
    assert events
    assert all(event.run_id == "run-explicit-adapter" for event in events)
    assert any(event.event_name == "answer.delta" for event in events)
    settled = next(event for event in events if event.event_name == "runtime.settled")
    assert settled.status == "completed"
    assert settled.attributes["stop_reason"] == "final_answer"
    assert settled.attributes["recovery_source"] == ""
    assert all("payload" not in event.attributes for event in events)
    assert "hello" not in str([event.attributes for event in events])


def test_forced_tool_contract_falls_back_to_structured_clarification() -> None:
    class IgnoresForcedTool(EchoModelAdapter):
        def invoke(self, request: ModelInvocation, *, on_text_delta=None) -> ModelTurn:
            return ModelTurn(
                content="Please provide the missing dates.",
                finish_reason="stop",
                model_id=self.info.model_id,
                model_role=self.info.model_role,
            )

    registry = ToolRegistry(
        [
            ToolSpec(
                name="plan.build",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string"},
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                    },
                    "required": ["objective", "start_date", "end_date"],
                    "additionalProperties": False,
                },
                required_args=["objective", "start_date", "end_date"],
                argument_contract={
                    "clarification": {
                        "prompt": "Please provide the objective, start date, and end date."
                    }
                },
                read_only=False,
            )
        ]
    )
    runtime = runtime_sdk.HarnessRuntime(registry, ToolExecutor(registry))

    result = run_runtime(
        runtime,
        "Interview",
        provider=IgnoresForcedTool(),
        context_bundle={"agent_session_id": "run-forced-tool-clarification"},
        skill_activation={
            "skill_id": "planning",
            "kind": "workflow",
            "allowed_tools": ["plan.build"],
            "required_tools": ["plan.build"],
            "completion_policy": {
                "required_transition": "plan.build",
                "allow_model_clarification": False,
                "clarification_strategy": "tool_contract",
                "waiting_statuses": ["waiting_user_input"],
            },
        },
    )

    assert result.status.value == "waiting_user_input"
    assert result.pending_action is not None
    assert result.pending_action.action_type == "clarification"
    assert result.pending_action.payload["clarification"]["missing_fields"] == [
        "objective",
        "start_date",
        "end_date",
    ]
    assert result.error is None


def test_telemetry_failure_is_fail_open_and_visible_in_diagnostics() -> None:
    class BrokenTelemetry:
        def record(self, _event: TelemetryRecord) -> None:
            raise RuntimeError("collector unavailable")

    runtime = (
        HarnessRuntimeBuilder()
        .with_ports(RuntimePorts(telemetry=BrokenTelemetry()))
        .build()
    )

    result = run_runtime(
        runtime,
        "hello",
        provider=EchoModelAdapter(),
        context_bundle={"agent_session_id": "run-broken-telemetry"},
    )

    assert result.status.value == "completed"
    diagnostics = result.diagnostics["telemetry"]
    assert diagnostics["status"] == "degraded"
    assert diagnostics["error_count"] > 0


def test_capability_pack_installs_every_extension_kind_from_real_registrations() -> None:
    @dataclass(frozen=True, slots=True)
    class CompletePack:
        spec = CapabilityPackSpec(
            package_id="example.complete",
            declared_tools=("complete.read",),
            declared_skills=("complete-skill",),
            declared_artifact_types=("complete_record",),
            declared_handoffs=("complete.read",),
            declared_tool_providers=("complete-provider",),
            declared_resources=("tool-provider:complete-provider",),
            declared_subagents=("complete-specialist",),
            declared_context_contributors=("complete-context",),
        )

        def install(self, context: CapabilityInstallContext) -> CapabilityContribution:
            context.register_tool(
                ToolSpec(
                    name="complete.read",
                    parameters_schema={"type": "object", "properties": {}},
                ),
                lambda *_args: {"status": "ok"},
            )
            context.register_skill("complete-skill", {"label": "Complete"})
            context.register_artifact_type(
                ArtifactTypeSpec(
                    artifact_type="complete_record",
                    schema={"type": "object", "properties": {}},
                )
            )
            context.register_handoff(
                "complete.read",
                HandoffSpec(
                    action_kind="complete",
                    noun="record",
                    title="Complete record",
                    summary="Complete the record",
                    primary_label="Continue",
                    cancel_label="Cancel",
                    completion_artifact_type="complete_record",
                ),
            )
            class CompleteProvider:
                spec = ToolProviderSpec(provider_id="complete-provider")

                def install(self, *, registry, executor) -> None:
                    return None

                def diagnostics(self) -> list[dict[str, object]]:
                    return []

                def close(self) -> None:
                    return None

            context.register_tool_provider(CompleteProvider())
            context.register_subagent(
                SubAgentSpec(
                    id="complete-specialist",
                    label="Complete specialist",
                    tool_allowlist=["complete.read"],
                )
            )
            context.register_context_contributor(
                "complete-context",
                lambda value: {
                    **value,
                    "runtime_context": {"catalog_context": "installed"},
                },
            )
            return CapabilityContribution(
                package_id=self.spec.package_id,
                metadata={"source": "contract-test"},
            )

    runtime = HarnessRuntimeBuilder().add_capability_pack(CompletePack()).build()
    contribution = runtime.capability_contributions[0]

    assert contribution.tools == ("complete.read",)
    assert contribution.skills == ("complete-skill",)
    assert contribution.artifact_types == ("complete_record",)
    assert contribution.handoffs == ("complete.read",)
    assert contribution.tool_providers == ("complete-provider",)
    assert contribution.subagents == ("complete-specialist",)
    assert contribution.context_contributors == ("complete-context",)
    assert contribution.metadata == {"source": "contract-test"}
    assert validate_capability_pack(CompletePack()).passed is True

    class ContextAwareModel(EchoModelAdapter):
        def invoke(self, request: ModelInvocation, *, on_text_delta=None) -> ModelTurn:
            assert any(
                "catalog_context" in str(message.get("content") or "")
                for message in request.messages
            )
            return super().invoke(request, on_text_delta=on_text_delta)

    result = run_runtime(runtime, "hello", provider=ContextAwareModel())
    assert result.final_answer.markdown == "hello back"


def test_failed_capability_install_rolls_back_all_partial_registrations() -> None:
    @dataclass(frozen=True, slots=True)
    class BrokenPack:
        spec = CapabilityPackSpec(package_id="example.broken")

        def install(self, context: CapabilityInstallContext):
            context.register_tool(
                ToolSpec(
                    name="broken.read",
                    parameters_schema={"type": "object", "properties": {}},
                ),
                lambda *_args: {"status": "ok"},
            )
            context.register_skill("broken-skill", {})
            raise RuntimeError("installation failed")

    builder = HarnessRuntimeBuilder().add_capability_pack(BrokenPack())

    with pytest.raises(RuntimeError, match="installation failed"):
        builder.build()

    assert builder.registry.list_tools() == []
    assert builder.capability_catalog.skills == {}


def test_conformance_rejects_declared_capability_that_was_not_installed() -> None:
    @dataclass(frozen=True, slots=True)
    class IncompletePack:
        spec = CapabilityPackSpec(
            package_id="example.incomplete",
            declared_skills=("missing-skill",),
        )

        def install(self, _context: CapabilityInstallContext):
            return CapabilityContribution(package_id=self.spec.package_id)

    report = validate_capability_pack(IncompletePack())

    assert report.passed is False
    assert report.missing_capabilities == {"skills": ["missing-skill"]}


def test_context_setup_failure_returns_standard_terminal_contract() -> None:
    events: list[dict] = []
    telemetry = InMemoryTelemetry()

    def broken_context_builder(_question, _short, _bundle):
        raise RuntimeError("context backend unavailable")

    runtime = (
        HarnessRuntimeBuilder()
        .with_ports(
            RuntimePorts(
                context_builder=broken_context_builder,
                telemetry=telemetry,
            )
        )
        .build()
    )

    result = run_runtime(
        runtime,
        "hello",
        provider=EchoModelAdapter(),
        context_bundle={
            "agent_session_id": "context-failure-run",
            "thread_id": "context-failure-thread",
        },
        event_sink=events.append,
    )

    assert result.status.value == "failed"
    assert result.final_answer.status == "failed"
    assert events[-1]["event_type"] == "agent.failed"
    assert events[-1]["payload"]["phase"] == "context_setup"
    assert result.ui_state["can_send"] is True
    assert result.ui_state["composer_mode"] == "ready"
    assert result.error["message"] != "context backend unavailable"
    settled = telemetry.snapshot()[-1]
    assert settled.event_name == "runtime.settled"
    assert settled.status == "failed"
    assert settled.attributes["phase"] == "context_setup"


def test_context_window_deduplicates_history_and_enforces_a_deterministic_budget() -> None:
    manager = DeterministicContextWindowManager(
        ContextWindowPolicy(
            max_input_tokens=180,
            reserved_output_tokens=40,
            history_limit=2,
            max_message_tokens=24,
            minimum_section_tokens=8,
        )
    )

    prepared = manager.prepare(
        "summarize",
        {"current_time": {"year": 2026}},
        {
            "runtime_context": {
                "facts": {"document": "fact " * 120},
                "memories": [{"content": "memory " * 40}],
                "recent_messages": [
                    {"role": "user", "content": "old"},
                    {"role": "assistant", "content": "middle " * 40},
                    {"role": "user", "content": "latest " * 40},
                ],
            }
        },
    )

    assert "recent_messages" not in prepared.context_bundle["runtime_context"]
    assert len(prepared.context_bundle["recent_messages"]) == 2
    assert prepared.diagnostics["history_dropped_count"] == 1
    assert prepared.diagnostics["history_truncated_count"] == 2
    assert prepared.diagnostics["final_tokens"] <= 140
    assert prepared.diagnostics["original_tokens"] > prepared.diagnostics["final_tokens"]


def test_runtime_exposes_context_window_diagnostics_without_prompt_payloads() -> None:
    runtime = HarnessRuntimeBuilder().build()

    result = run_runtime(
        runtime,
        "hello",
        provider=EchoModelAdapter(),
        context_bundle={
            "agent_session_id": "context-window-run",
            "runtime_context": {"facts": {"safe": True}},
        },
    )

    diagnostics = result.diagnostics["context_window"]
    assert diagnostics["schema_version"] == "harness-context-window-v2"
    assert diagnostics["final_tokens"] > 0
    assert "facts" not in diagnostics
    assert any(
        event.source_event_type == "budget.usage.recorded"
        for event in result.events
    )


def test_context_window_prefers_required_segments_and_cached_summaries() -> None:
    manager = DeterministicContextWindowManager(
        ContextWindowPolicy(
            max_input_tokens=180,
            reserved_output_tokens=40,
            minimum_section_tokens=8,
        )
    )
    prepared = manager.prepare(
        "review",
        {},
        {
            "runtime_context": {
                "subject": {"id": "subject-1"},
                "facts": {"raw": "important " * 100},
                "optional_notes": "noise " * 100,
            },
            "context_segments": [
                ContextSegment(
                    key="facts",
                    value={"raw": "important " * 100},
                    priority=100,
                    source="fact-store",
                    summary={"summary": "stable fact summary"},
                    summary_version="facts-v3",
                ),
                {
                    "key": "optional_notes",
                    "priority": -10,
                    "source": "transient",
                },
            ],
        },
    )

    context = prepared.context_bundle["runtime_context"]
    diagnostics = prepared.diagnostics
    assert context["subject"] == {"id": "subject-1"}
    assert context["facts"] == {"summary": "stable fact summary"}
    assert "subject" in diagnostics["required_sections_retained"]
    assert diagnostics["summarized_sections"] == ["facts"]
    assert diagnostics["final_tokens"] <= diagnostics["input_budget_tokens"]


def test_context_summary_cache_hits_only_for_the_matching_source_fingerprint() -> None:
    cache = InMemoryContextSummaryCache()
    manager = DeterministicContextWindowManager(
        ContextWindowPolicy(
            max_input_tokens=120,
            reserved_output_tokens=30,
            minimum_section_tokens=8,
        ),
        summary_cache=cache,
    )
    seeded = manager.prepare(
        "review",
        {},
        {
            "runtime_context": {"facts": "raw " * 100},
            "context_segments": [{
                "key": "facts",
                "priority": 100,
                "cache_key": "facts:1",
                "source_fingerprint": "source-v1",
                "summary": "cached summary",
            }],
        },
    )
    assert seeded.context_bundle["runtime_context"]["facts"] == "cached summary"

    hit = manager.prepare(
        "review",
        {},
        {
            "runtime_context": {"facts": "raw " * 100},
            "context_segments": [{
                "key": "facts",
                "priority": 100,
                "cache_key": "facts:1",
                "source_fingerprint": "source-v1",
            }],
        },
    )
    assert hit.context_bundle["runtime_context"]["facts"] == "cached summary"
    assert hit.diagnostics["summary_cache_hits"] == ["facts"]

    stale = manager.prepare(
        "review",
        {},
        {
            "runtime_context": {"facts": "new " * 100},
            "context_segments": [{
                "key": "facts",
                "priority": 100,
                "cache_key": "facts:1",
                "source_fingerprint": "source-v2",
            }],
        },
    )
    assert stale.context_bundle["runtime_context"].get("facts") != "cached summary"
    assert stale.diagnostics["summary_cache_misses"] == ["facts"]


def test_budget_ledger_supports_idempotent_peak_aggregation() -> None:
    ledger = InMemoryBudgetLedger()
    first = ledger.consume(
        BudgetRequest(
            run_id="run-budget-peak",
            metric="tool_concurrency",
            amount=3,
            limit=4,
            operation_id="parallel-batch-1",
            aggregation="max",
        )
    )
    replay = ledger.consume(
        BudgetRequest(
            run_id="run-budget-peak",
            metric="tool_concurrency",
            amount=3,
            limit=4,
            operation_id="parallel-batch-1",
            aggregation="max",
        )
    )
    lower_peak = ledger.consume(
        BudgetRequest(
            run_id="run-budget-peak",
            metric="tool_concurrency",
            amount=2,
            limit=4,
            operation_id="parallel-batch-2",
            aggregation="max",
        )
    )

    assert first.allowed and replay.allowed and lower_peak.allowed
    assert first.used == replay.used == lower_peak.used == 3
    assert ledger.snapshot("run-budget-peak").usage["tool_concurrency"] == 3


def test_budget_policy_role_override_and_provider_usage_contract() -> None:
    policy = BudgetPolicy.from_mapping({
        "max_output_tokens_per_call": 1000,
        "roles": {"fast": {"max_output_tokens_per_call": 256}},
    })
    usage = UsageReport.from_provider(
        {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        estimated_input=999,
        estimated_output=999,
    )

    assert policy.limit("max_output_tokens_per_call", role="fast") == 256
    assert policy.limit("max_output_tokens_per_call", role="reasoning") == 1000
    assert usage.as_dict() == {
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
        "source": "provider",
    }


def test_runtime_state_store_commits_status_event_and_checkpoint_atomically() -> None:
    store = InMemoryRuntimeStateStore()
    mutation = RuntimeStateMutation(
        mutation_id="mutation-1",
        run_id="run-atomic",
        event_type="runtime.checkpoint.committed",
        target_status="waiting_user_input",
        event_payload={"status": "waiting_user_input"},
        checkpoint_state={"schema_version": "harness-durable-checkpoint-v2"},
        expected_version=0,
        expected_sequence=0,
    )

    receipt = store.commit(mutation)
    replay = store.commit(mutation)
    snapshot = store.snapshot("run-atomic")

    assert receipt.version == receipt.sequence == 1
    assert replay.replayed is True
    assert snapshot["status"] == "waiting_user_input"
    assert len(snapshot["events"]) == len(snapshot["checkpoints"]) == 1


def test_runtime_state_store_rolls_back_every_crash_point_and_rejects_stale_writes() -> None:
    def fail_after_checkpoint(stage: str) -> None:
        if stage == "after_checkpoint":
            raise RuntimeError("simulated crash")

    failed_store = InMemoryRuntimeStateStore(fail_after_checkpoint)
    mutation = RuntimeStateMutation(
        mutation_id="mutation-crash",
        run_id="run-crash",
        event_type="runtime.checkpoint.committed",
        target_status="task_running",
        checkpoint_state={"state": "waiting"},
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        failed_store.commit(mutation)
    assert failed_store.snapshot("run-crash") == {
        "version": 0,
        "sequence": 0,
        "status": "preparing",
        "events": [],
        "checkpoints": [],
    }

    store = InMemoryRuntimeStateStore()
    store.commit(mutation)
    with pytest.raises(RuntimeStateConflict):
        store.commit(
            RuntimeStateMutation(
                mutation_id="mutation-stale",
                run_id="run-crash",
                event_type="runtime.checkpoint.committed",
                target_status="task_running",
                expected_version=0,
            )
        )


@pytest.mark.parametrize(
    "crash_point",
    (
        "before_event",
        "after_event",
        "after_checkpoint",
        "after_checkpoint_cleanup",
        "before_commit",
    ),
)
def test_runtime_state_store_has_no_partial_write_at_each_crash_point(
    crash_point: str,
) -> None:
    def fail_at(stage: str) -> None:
        if stage == crash_point:
            raise RuntimeError(f"crash:{stage}")

    store = InMemoryRuntimeStateStore(fail_at)
    mutation = RuntimeStateMutation(
        mutation_id=f"mutation-{crash_point}",
        run_id=f"run-{crash_point}",
        event_type="run.settled",
        target_status="completed",
        event_payload={"status": "completed"},
        checkpoint_type="terminal",
        checkpoint_state={"status": "completed"},
        delete_checkpoint_types=("runtime",),
    )

    with pytest.raises(RuntimeError, match=f"crash:{crash_point}"):
        store.commit(mutation)
    assert store.snapshot(mutation.run_id) == {
        "version": 0,
        "sequence": 0,
        "status": "preparing",
        "events": [],
        "checkpoints": [],
    }


def test_terminal_state_mutation_replaces_resumable_checkpoint_atomically() -> None:
    fail_terminal = False

    def fail_after_cleanup(stage: str) -> None:
        if fail_terminal and stage == "after_checkpoint_cleanup":
            raise RuntimeError("terminal commit interrupted")

    store = InMemoryRuntimeStateStore(fail_after_cleanup)
    waiting = RuntimeStateMutation(
        mutation_id="wait-before-terminal",
        run_id="run-terminal",
        event_type="runtime.checkpoint.committed",
        target_status="waiting_user_action",
        checkpoint_type="runtime",
        checkpoint_state={"status": "waiting_user_action"},
    )
    store.commit(waiting)

    terminal = RuntimeStateMutation(
        mutation_id="terminal-completed",
        run_id="run-terminal",
        event_type="run.settled",
        target_status="completed",
        event_payload={"status": "completed"},
        event_visibility="public",
        checkpoint_type="terminal",
        checkpoint_state={"status": "completed"},
        delete_checkpoint_types=("runtime", "suspended", "resume", "settling"),
        expected_version=1,
        expected_sequence=1,
    )
    fail_terminal = True
    with pytest.raises(RuntimeError, match="terminal commit interrupted"):
        store.commit(terminal)
    interrupted = store.snapshot("run-terminal")
    assert interrupted["status"] == "waiting_user_action"
    assert [item["checkpoint_type"] for item in interrupted["checkpoints"]] == [
        "runtime"
    ]

    fail_terminal = False
    receipt = store.commit(terminal)
    replay = store.commit(terminal)
    completed = store.snapshot("run-terminal")
    assert receipt.status == "completed"
    assert replay.replayed is True
    assert [item["checkpoint_type"] for item in completed["checkpoints"]] == [
        "terminal"
    ]


def test_runtime_persists_one_canonical_settlement_before_cleanup() -> None:
    registry = ToolRegistry()
    store = InMemoryRuntimeStateStore()
    runtime = HarnessRuntimeBuilder(registry, ToolExecutor(registry)).with_ports(
        RuntimePorts(runtime_state_store=store)
    ).build()

    result = run_runtime(
        runtime,
        "hello",
        provider=EchoModelAdapter(),
        context_bundle={"agent_session_id": "run-settled"},
    )
    snapshot = store.load_snapshot("run-settled")
    journal = store.snapshot("run-settled")

    assert result.status.value == "completed"
    assert snapshot.settled is True
    assert snapshot.settlement_status == "completed"
    assert snapshot.last_event_type == "run.settled"
    assert snapshot.can_accept_input is True
    assert [event["event_type"] for event in journal["events"]] == ["run.settled"]
    assert journal["events"][0]["payload"]["status"] == "completed"


def test_model_invocation_envelope_is_replayable_without_leaking_prompt_to_events() -> None:
    registry = ToolRegistry()
    recorder = InMemoryModelInvocationRecorder()
    runtime = HarnessRuntimeBuilder(registry, ToolExecutor(registry)).with_ports(
        RuntimePorts(model_invocation_recorder=recorder)
    ).build()

    result = run_runtime(
        runtime,
        "private question",
        provider=EchoModelAdapter(),
        context_bundle={"agent_session_id": "run-envelope"},
    )
    route = next(item for item in result.trace if item["action"] == "model.route.selected")
    public = route["invocation"]
    envelope = recorder.get(public["invocation_id"])

    assert public["recorded"] is True
    assert public["message_count"] >= 2
    assert "messages" not in public
    assert "private question" not in json.dumps(public, ensure_ascii=False)
    assert envelope is not None
    assert envelope.request.messages[-1]["content"] == "private question"
    assert envelope.request_fingerprint == public["request_fingerprint"]


def test_settled_run_rejects_non_idempotent_follow_up_mutation() -> None:
    store = InMemoryRuntimeStateStore()
    settled = RuntimeStateMutation(
        mutation_id="settled-once",
        run_id="run-immutable",
        event_type="run.settled",
        target_status="failed",
        event_payload={"status": "failed"},
        checkpoint_type="terminal",
    )
    store.commit(settled)

    assert store.commit(settled).replayed is True
    with pytest.raises(RuntimeStateConflict, match="settled run"):
        store.commit(
            RuntimeStateMutation(
                mutation_id="settled-twice",
                run_id="run-immutable",
                event_type="run.settled",
                target_status="completed",
                event_payload={"status": "completed"},
                checkpoint_type="terminal",
            )
        )


def test_runtime_waiting_action_is_committed_through_atomic_state_store() -> None:
    class HandoffModelAdapter:
        info = ModelProviderInfo(
            provider_id="example.handoff",
            model_id="handoff-v1",
            model_role="reasoning",
        )

        def invoke(self, request: ModelInvocation, *, on_text_delta=None) -> ModelTurn:
            return ModelTurn(
                tool_calls=[ToolCall(id="handoff-call", name="workflow.handoff")],
                finish_reason="tool_calls",
                model_id=self.info.model_id,
                model_role=self.info.model_role,
            )

    registry = ToolRegistry(
        [
            ToolSpec(
                name="workflow.handoff",
                parameters_schema={"type": "object", "properties": {}},
                requires_user_action=True,
            )
        ]
    )
    executor = ToolExecutor(registry)

    def require_action(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(
            call=call,
            status="requires_user_action",
            summary="User confirmation is required.",
            pending_action=PendingAction(
                id="pending-1",
                run_id=context.run_id,
                tool_call_id=call.id,
                action_type="confirmation",
                prompt="Continue?",
            ),
        )

    executor.register("workflow.handoff", require_action)
    store = InMemoryRuntimeStateStore()
    runtime = HarnessRuntimeBuilder(registry, executor).with_ports(
        RuntimePorts(runtime_state_store=store)
    ).build()

    result = run_runtime(
        runtime,
        "start workflow",
        provider=HandoffModelAdapter(),
        context_bundle={"agent_session_id": "run-handoff"},
        skill_activation={
            "skill_id": "handoff_workflow",
            "kind": "workflow",
            "allowed_tools": ["workflow.handoff"],
            "required_tools": ["workflow.handoff"],
            "completed_tools": [],
            "completion_policy": {
                "waiting_statuses": ["waiting_user_action"],
            },
        },
    )
    snapshot = store.snapshot("run-handoff")

    assert result.status.value == "waiting_user_action"
    recovery = result.diagnostics["recovery"]
    assert recovery["atomic_checkpoint"] == "persisted"
    assert recovery["state_receipt"]["status"] == "waiting_user_action"
    assert snapshot["status"] == "waiting_user_action"
    assert len(snapshot["events"]) == len(snapshot["checkpoints"]) == 1
    assert result.skill_activation["completed_tools"] == ["workflow.handoff"]


def test_runtime_hides_tools_skipped_after_user_action_suspension() -> None:
    class HandoffWithSiblingModelAdapter:
        info = ModelProviderInfo(
            provider_id="example.handoff-sibling",
            model_id="handoff-sibling-v1",
            model_role="reasoning",
        )

        def invoke(self, request: ModelInvocation, *, on_text_delta=None) -> ModelTurn:
            return ModelTurn(
                tool_calls=[
                    ToolCall(id="handoff-call", name="workflow.handoff"),
                    ToolCall(id="memory-call", name="memory.read"),
                ],
                finish_reason="tool_calls",
                model_id=self.info.model_id,
                model_role=self.info.model_role,
            )

    registry = ToolRegistry(
        [
            ToolSpec(
                name="workflow.handoff",
                parameters_schema={"type": "object", "properties": {}},
                requires_user_action=True,
            ),
            ToolSpec(
                name="memory.read",
                parameters_schema={"type": "object", "properties": {}},
            ),
        ]
    )
    executor = ToolExecutor(registry)
    memory_calls = 0

    def require_action(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(
            call=call,
            status="requires_user_action",
            summary="User confirmation is required.",
            pending_action=PendingAction(
                id="pending-with-sibling",
                run_id=context.run_id,
                tool_call_id=call.id,
                action_type="confirmation",
                prompt="Continue?",
            ),
        )

    def read_memory(call: ToolCall, _context: ToolExecutionContext) -> ToolResult:
        nonlocal memory_calls
        memory_calls += 1
        return ToolResult(call=call, status="succeeded", summary="Memory loaded.")

    executor.register("workflow.handoff", require_action)
    executor.register("memory.read", read_memory)
    result = run_runtime(
        HarnessRuntimeBuilder(registry, executor).build(),
        "start workflow",
        provider=HandoffWithSiblingModelAdapter(),
        context_bundle={"agent_session_id": "run-handoff-with-sibling"},
    )

    assert result.status.value == "waiting_user_action"
    assert memory_calls == 0
    assert all(item.name != "memory.read" for item in result.tool_results)
    assert not any(
        event.source_event_type in {"tool.failed", "tool.call.failed"}
        for event in result.events
    )
    assert any(event.source_event_type == "tool.skipped" for event in result.events)


def test_tool_executor_rejects_artifact_that_violates_registered_schema() -> None:
    @dataclass(frozen=True, slots=True)
    class ArtifactPack:
        spec = CapabilityPackSpec(
            package_id="example.artifact",
            declared_tools=("artifact.create",),
            declared_artifact_types=("inventory_record",),
        )

        def install(self, context: CapabilityInstallContext):
            def create(call: ToolCall, execution: ToolExecutionContext) -> ToolResult:
                return ToolResult(
                    call=call,
                    status="succeeded",
                    artifacts=[
                        Artifact(
                            id="artifact-1",
                            run_id=execution.run_id,
                            artifact_type="inventory_record",
                            data={"quantity": "not-an-integer"},
                        )
                    ],
                )

            context.register_tool(
                ToolSpec(
                    name="artifact.create",
                    parameters_schema={"type": "object", "properties": {}},
                ),
                create,
            )
            context.register_artifact_type(
                ArtifactTypeSpec(
                    artifact_type="inventory_record",
                    schema={
                        "type": "object",
                        "properties": {"quantity": {"type": "integer"}},
                        "required": ["quantity"],
                    },
                )
            )

    runtime = HarnessRuntimeBuilder().add_capability_pack(ArtifactPack()).build()
    result = runtime.tool_executor.execute(
        ToolCall(id="artifact-call", name="artifact.create"),
        ToolExecutionContext(run_id="artifact-run", user_id="user-1"),
    )

    assert result.status == "failed"
    assert result.metadata["artifact_contract_failed"] is True
    assert "does not match its contract" in result.error


def test_failed_pack_install_restores_replaced_handler() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(
                name="existing.read",
                parameters_schema={"type": "object", "properties": {}},
            )
        ]
    )
    executor = ToolExecutor(registry)

    def original_handler(_call, _context):
        return ToolResult(
            call=_call,
            status="succeeded",
            data={"source": "original"},
        )

    executor.register("existing.read", original_handler)

    @dataclass(frozen=True, slots=True)
    class BrokenReplacementPack:
        spec = CapabilityPackSpec(package_id="example.replacement")

        def install(self, context: CapabilityInstallContext):
            context.executor.register(
                "existing.read",
                lambda call, _context: ToolResult(
                    call=call,
                    status="succeeded",
                    data={"source": "replacement"},
                ),
            )
            raise RuntimeError("replacement failed")

    builder = HarnessRuntimeBuilder(registry, executor).add_capability_pack(
        BrokenReplacementPack()
    )
    with pytest.raises(RuntimeError, match="replacement failed"):
        builder.build()

    result = executor.execute(
        ToolCall(id="existing-call", name="existing.read"),
        ToolExecutionContext(run_id="existing-run", user_id="user-1"),
    )
    assert result.data == {"source": "original"}


def test_core_executor_rejects_untyped_handler_payloads() -> None:
    registry = ToolRegistry(
        [ToolSpec(name="typed.only", parameters_schema={"type": "object"})]
    )
    executor = ToolExecutor(registry)
    executor.register("typed.only", lambda *_args: {"status": "ok"})

    result = executor.execute(
        ToolCall(id="typed-call", name="typed.only"),
        ToolExecutionContext(run_id="typed-run", user_id="user-1"),
    )

    assert result.status == "failed"
    assert result.error == "tool handlers must return ToolResult"


def test_capability_resources_close_with_runtime_and_failed_install() -> None:
    class Resource:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    installed_resource = Resource()

    @dataclass(frozen=True, slots=True)
    class ResourcePack:
        spec = CapabilityPackSpec(
            package_id="example.resource",
            declared_resources=("client:primary",),
        )

        def install(self, context: CapabilityInstallContext):
            context.register_resource("client:primary", installed_resource)

    runtime = HarnessRuntimeBuilder().add_capability_pack(ResourcePack()).build()
    assert runtime.capability_contributions[0].resources == ("client:primary",)
    runtime.close()
    runtime.close()
    assert installed_resource.close_count == 1

    failed_resource = Resource()

    @dataclass(frozen=True, slots=True)
    class FailedResourcePack:
        spec = CapabilityPackSpec(package_id="example.failed-resource")

        def install(self, context: CapabilityInstallContext):
            context.register_resource("client:failed", failed_resource)
            raise RuntimeError("resource installation failed")

    with pytest.raises(RuntimeError, match="resource installation failed"):
        HarnessRuntimeBuilder().add_capability_pack(FailedResourcePack()).build()
    assert failed_resource.close_count == 1


def test_runtime_diagnostics_expose_installed_capabilities_without_payloads() -> None:
    runtime = HarnessRuntimeBuilder().add_capability_pack(InventoryPack()).build()

    result = run_runtime(runtime, "hello", provider=EchoModelAdapter())
    capabilities = result.diagnostics["capabilities"]

    assert capabilities["packages"] == [
        {
            "package_id": "example.inventory",
            "tools": 1,
            "skills": 1,
            "artifact_types": 0,
            "subagents": 0,
            "resources": 0,
        }
    ]
    assert capabilities["catalog"]["skills"] == 1
