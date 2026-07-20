from __future__ import annotations

from dataclasses import dataclass

import pytest

from harness_core.composition import HarnessRuntimeBuilder, RuntimePorts
from harness_core.extension_sdk import (
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityPackSpec,
)
from harness_core.runtime_sdk import RuntimeRequest
from harness_core.policy import DefaultPolicyEngine
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


def test_builder_injects_shared_governance_ports_into_executor_and_runtime():
    policy = DefaultPolicyEngine()
    runtime = HarnessRuntimeBuilder().with_ports(RuntimePorts(policy_engine=policy)).build()

    assert runtime.policy_engine is policy
    assert runtime.tool_executor.policy_engine is policy
