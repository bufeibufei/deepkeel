from __future__ import annotations

from dataclasses import dataclass

import pytest

from harness_core.composition import HarnessRuntimeBuilder, RuntimePorts
from harness_core.extension_sdk import (
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityManifest,
    CapabilityPackSpec,
    RuntimeGeneration,
)
from harness_core.runtime_sdk import RuntimeRequest
from harness_core.runtime_sdk import InMemoryDurableCheckpointStore
from harness_core.policy import DefaultPolicyEngine
from harness_core.model import ModelTurn
from harness_core.tool_registry import ToolRegistry, ToolSpec
from harness_core.tools import ToolExecutor


class ScriptedNativeProvider:
    model = "scripted-model"
    model_role = "reasoning"

    def __init__(self, turns):
        self.turns = list(turns)

    def complete_chat(self, _messages, **_kwargs):
        content = self.turns.pop(0)
        return {
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
            "model": self.model,
        }


class TruncatingModelAdapter:
    info = type(
        "ProviderInfo",
        (),
        {
            "provider_id": "test.truncating",
            "model_id": "test-long-answer",
            "model_role": "reasoning",
            "supports_native_tools": True,
        },
    )()

    def __init__(self, turns: list[ModelTurn]):
        self.turns = list(turns)
        self.requests = []

    def invoke(self, request, *, on_text_delta=None):
        self.requests.append(request)
        turn = self.turns.pop(0)
        if on_text_delta is not None:
            on_text_delta(turn.content)
        return turn


@dataclass
class DemoCapabilityPack:
    spec = CapabilityPackSpec(
        package_id="example.demo",
        declared_tools=("demo.lookup",),
    )
    registrations: int = 0

    def install(self, context: CapabilityInstallContext) -> CapabilityContribution:
        self.registrations += 1
        context.register_tool(
            ToolSpec(
                name="demo.lookup",
                parameters_schema={"type": "object", "properties": {}},
            ),
            lambda *_args: (_ for _ in ()).throw(AssertionError("not executed")),
        )
        return CapabilityContribution(
            package_id=self.spec.package_id,
            tools=("demo.lookup",),
        )


def test_builder_composes_runtime_and_capability_pack_once():
    pack = DemoCapabilityPack()
    builder = HarnessRuntimeBuilder().add_capability_pack(pack)

    runtime = builder.build()
    result = runtime.run(
        RuntimeRequest(
            question="hello",
            context_bundle={"agent_session_id": "run-builder"},
        ),
        provider=ScriptedNativeProvider(["hello back"]),
    )

    assert result.final_answer.markdown == "hello back"
    assert runtime.tool_registry.get("demo.lookup").name == "demo.lookup"
    assert pack.registrations == 1
    with pytest.raises(RuntimeError, match="cannot be reused"):
        builder.build()


def test_builder_rejects_executor_with_another_registry():
    with pytest.raises(ValueError, match="same ToolRegistry"):
        HarnessRuntimeBuilder(ToolRegistry(), ToolExecutor(ToolRegistry()))


def test_builder_rejects_duplicate_capability_package_identity():
    builder = HarnessRuntimeBuilder().add_capability_pack(DemoCapabilityPack())

    with pytest.raises(ValueError, match="already registered"):
        builder.add_capability_pack(DemoCapabilityPack())


def test_builder_runs_the_generation_selected_by_the_control_plane():
    manifest = CapabilityManifest(
        id="example.demo",
        version="1.0.0",
        core_version="*",
        entrypoint="tests.test_composition:DemoCapabilityPack",
        tools=("demo.lookup",),
    )
    generation = RuntimeGeneration.create(
        (manifest,),
        catalog_version="catalog-release-7",
    )

    runtime = (
        HarnessRuntimeBuilder()
        .add_capability_pack(DemoCapabilityPack(), manifest=manifest)
        .with_runtime_generation(generation)
        .build()
    )

    assert runtime.runtime_generation == generation


def test_builder_rejects_a_generation_that_differs_from_installed_packs():
    manifest = CapabilityManifest(
        id="example.demo",
        version="1.0.0",
        core_version="*",
        entrypoint="tests.test_composition:DemoCapabilityPack",
        tools=("demo.lookup",),
    )
    changed = manifest.model_copy(update={"version": "2.0.0"})

    with pytest.raises(ValueError, match="changed manifests"):
        (
            HarnessRuntimeBuilder()
            .add_capability_pack(DemoCapabilityPack(), manifest=manifest)
            .with_runtime_generation(RuntimeGeneration.create((changed,)))
            .build()
        )


def test_resume_fails_before_model_use_when_generation_is_incompatible():
    run_id = "resume-incompatible-generation"
    checkpoint_store = InMemoryDurableCheckpointStore()
    old_manifest = CapabilityManifest(
        id="example.demo",
        version="1.0.0",
        core_version="*",
        entrypoint="tests.test_composition:DemoCapabilityPack",
        tools=("demo.lookup",),
    )
    checkpoint_store.save(
        run_id,
        {
            "schema_version": "harness-durable-checkpoint-v2",
            "run_id": run_id,
            "thread_id": "thread-generation",
            "runtime": {
                "diagnostics": {
                    "capabilities": {
                        "generation": RuntimeGeneration.create(
                            (old_manifest,)
                        ).model_dump(mode="json")
                    }
                }
            },
        },
        user_id="user-generation",
    )
    current_manifest = old_manifest.model_copy(update={"version": "2.0.0"})
    runtime = (
        HarnessRuntimeBuilder()
        .with_ports(RuntimePorts(checkpoint_store=checkpoint_store))
        .add_capability_pack(DemoCapabilityPack(), manifest=current_manifest)
        .build()
    )

    result = runtime.run(
        RuntimeRequest(
            question="continue",
            user_id="user-generation",
            run_id=run_id,
            thread_id="thread-generation",
            short_context={"resume": True},
        ),
        provider=ScriptedNativeProvider([]),
    )

    assert result.status.value == "failed"
    assert result.error is not None
    assert result.error["code"] == "CHECKPOINT_INCOMPATIBLE"


def test_runtime_continues_and_merges_output_truncated_by_model_limit():
    provider = TruncatingModelAdapter(
        [
            ModelTurn(
                content="The first section ends with shared text",
                finish_reason="length",
                model_id="test-long-answer",
                model_role="reasoning",
            ),
            ModelTurn(
                content="shared text and the answer is complete.",
                finish_reason="stop",
                model_id="test-long-answer",
                model_role="reasoning",
            ),
        ]
    )
    runtime = HarnessRuntimeBuilder().build()

    result = runtime.run(
        RuntimeRequest(
            question="give me a long answer",
            context_bundle={"agent_session_id": "run-output-continuation"},
        ),
        provider=provider,
    )

    assert result.status.value == "completed"
    assert (
        result.final_answer.markdown
        == "The first section ends with shared text and the answer is complete."
    )
    assert len(provider.requests) == 2
    assert provider.requests[1].messages[-1]["role"] == "user"
    assert "Continue exactly where it stopped" in provider.requests[1].messages[-1]["content"]
    assert any(event.event_type == "model.output_truncated.retrying" for event in result.events)


def test_builder_injects_shared_governance_ports_into_executor_and_runtime():
    policy = DefaultPolicyEngine()
    runtime = HarnessRuntimeBuilder().with_ports(RuntimePorts(policy_engine=policy)).build()

    assert runtime.policy_engine is policy
    assert runtime.tool_executor.policy_engine is policy
