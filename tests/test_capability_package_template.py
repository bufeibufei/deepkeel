from __future__ import annotations

from dataclasses import dataclass

from deepkeel.extension_sdk import (
    CapabilityBudgetSpec,
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityManifest,
    CapabilityPackSpec,
    ToolSpec,
    capability_pack_spec_from_manifest,
    certify_capability_package,
    validate_capability_pack,
)
from deepkeel.runtime_sdk import (
    EvalCase,
    EvalExpectation,
    RuntimeRequest,
)
from deepkeel.composition import HarnessRuntimeBuilder
from deepkeel.model import ModelTurn


MANIFEST = CapabilityManifest(
    id="example.inventory",
    version="1.2.0",
    core_version="*",
    entrypoint="tests.test_capability_package_template:InventoryPack",
    tools=("inventory.lookup",),
    permissions=("inventory.read",),
    tool_permissions={"inventory.lookup": ("inventory.read",)},
    budget=CapabilityBudgetSpec(
        max_model_calls=3,
        max_tool_calls=2,
    ),
    state_schema_version="2",
    resume_compatible_versions=("1.1.0",),
    metadata={"domain": "inventory"},
)


@dataclass
class InventoryPack:
    spec = capability_pack_spec_from_manifest(MANIFEST)

    def install(self, context: CapabilityInstallContext) -> CapabilityContribution:
        context.register_tool(
            ToolSpec(
                name="inventory.lookup",
                read_only=True,
                parameters_schema={
                    "type": "object",
                    "properties": {"item": {"type": "string"}},
                    "required": ["item"],
                },
                runtime_policy={"required_scopes": ["inventory.read"]},
            ),
            lambda *_args: {"status": "succeeded", "result": {"quantity": 12}},
        )
        return CapabilityContribution(
            package_id=self.spec.package_id,
            tools=("inventory.lookup",),
        )


def test_manifest_is_the_single_source_for_pack_spec_and_conformance() -> None:
    spec = capability_pack_spec_from_manifest(MANIFEST)
    report = validate_capability_pack(InventoryPack(), manifest=MANIFEST)

    assert spec.package_id == MANIFEST.id
    assert spec.required_scopes == ("inventory.read",)
    assert spec.metadata["capability_manifest"]["budget"] == {
        "max_model_calls": 3,
        "max_tool_calls": 2,
    }
    assert report.passed is True
    assert report.manifest_validated is True
    assert report.runtime_generation_id.startswith("generation-")
    assert report.declared_permissions == ["inventory.read"]
    assert report.declared_budget["max_tool_calls"] == 2
    assert report.state_schema_version == "2"
    assert report.resume_compatible_versions == ["1.1.0", "1.2.0"]


def test_conformance_rejects_pack_spec_drift_from_manifest() -> None:
    @dataclass
    class DriftedPack(InventoryPack):
        spec = CapabilityPackSpec(
            package_id=MANIFEST.id,
            package_version=MANIFEST.version,
            declared_tools=("inventory.lookup",),
        )

    report = validate_capability_pack(DriftedPack(), manifest=MANIFEST)

    assert report.passed is False
    assert report.manifest_validated is False
    assert report.runtime_generation_id == ""
    assert "manifest and pack spec disagree on required_scopes" in report.issues


class InventoryProvider:
    info = type(
        "ProviderInfo",
        (),
        {
            "provider_id": "test.inventory",
            "model_id": "inventory-model",
            "model_role": "reasoning",
            "supports_native_tools": True,
        },
    )()

    def __init__(self) -> None:
        self.turns = [
            ModelTurn(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "inventory-call",
                        "name": "inventory.lookup",
                        "arguments": {"item": "pen"},
                    }
                ],
            ),
            ModelTurn(content="There are 12 pens in stock.", finish_reason="stop"),
        ]

    def invoke(self, _request, *, on_text_delta=None):
        turn = self.turns.pop(0)
        if on_text_delta and turn.content:
            on_text_delta(turn.content)
        return turn


def test_certification_covers_lifecycle_governance_and_behavior() -> None:
    runtime = (
        HarnessRuntimeBuilder()
        .add_capability_pack(InventoryPack(), manifest=MANIFEST)
        .build()
    )
    cases = [
        EvalCase(
            case_id="inventory-lookup",
            request=RuntimeRequest(
                question="How many pens are available?",
                context_bundle={
                    "governance_scopes": ["inventory.read"],
                },
            ),
            expectation=EvalExpectation(
                required_tools=frozenset({"inventory.lookup"}),
            ),
            tags=frozenset(
                {
                    "tool_selection",
                    "argument_generation",
                    "task_completion",
                    "recovery",
                    "answer_quality",
                }
            ),
        )
    ]

    report = certify_capability_package(
        InventoryPack(),
        manifest=MANIFEST,
        cases=cases,
        execute=lambda request: runtime.run(request, provider=InventoryProvider()),
    )

    assert report.passed is True
    assert report.lifecycle.passed is True
    assert report.conformance.permission_coverage == {
        "inventory.lookup": ["inventory.read"]
    }
    assert report.missing_eval_tags == []


def test_certification_fails_when_required_behavior_is_not_covered() -> None:
    runtime = (
        HarnessRuntimeBuilder()
        .add_capability_pack(InventoryPack(), manifest=MANIFEST)
        .build()
    )

    report = certify_capability_package(
        InventoryPack(),
        manifest=MANIFEST,
        cases=[
            EvalCase(
                case_id="partial",
                request=RuntimeRequest(
                    question="Check pens",
                    context_bundle={"governance_scopes": ["inventory.read"]},
                ),
                expectation=EvalExpectation(
                    required_tools=frozenset({"inventory.lookup"})
                ),
                tags=frozenset({"tool_selection"}),
            )
        ],
        execute=lambda request: runtime.run(request, provider=InventoryProvider()),
    )

    assert report.passed is False
    assert report.missing_eval_tags == [
        "answer_quality",
        "argument_generation",
        "recovery",
        "task_completion",
    ]


def test_conformance_installs_declared_dependency_manifests_first() -> None:
    foundation = CapabilityManifest(
        id="example.foundation",
        version="1.0.0",
        core_version="*",
        entrypoint="example.foundation:Pack",
    )
    dependent_manifest = MANIFEST.model_copy(
        update={
            "id": "example.inventory-dependent",
            "dependencies": {"example.foundation": ">=1.0.0"},
        }
    )

    @dataclass
    class DependentInventoryPack(InventoryPack):
        spec = capability_pack_spec_from_manifest(dependent_manifest)

    report = validate_capability_pack(
        DependentInventoryPack(),
        manifest=dependent_manifest,
        dependency_manifests=(
            foundation,
            CapabilityManifest(
                id="example.unrelated",
                version="1.0.0",
                core_version="*",
                entrypoint="example.unrelated:Pack",
                dependencies={
                    "example.inventory-dependent": ">=1.0.0",
                },
            ),
        ),
    )

    assert report.passed is True
    assert report.runtime_generation_id.startswith("generation-")


def test_conformance_composes_real_dependency_packs_when_provided() -> None:
    foundation_manifest = CapabilityManifest(
        id="example.foundation",
        version="1.0.0",
        core_version="*",
        entrypoint="example.foundation:Pack",
        tools=("foundation.lookup",),
    )
    dependent_manifest = MANIFEST.model_copy(
        update={
            "id": "example.inventory-dependent",
            "dependencies": {"example.foundation": ">=1.0.0"},
        }
    )

    @dataclass
    class FoundationPack:
        spec = capability_pack_spec_from_manifest(foundation_manifest)

        def install(
            self,
            context: CapabilityInstallContext,
        ) -> CapabilityContribution:
            context.register_tool(
                ToolSpec(name="foundation.lookup", read_only=True),
                lambda *_args: {"status": "succeeded"},
            )
            return CapabilityContribution(
                package_id=self.spec.package_id,
                tools=("foundation.lookup",),
            )

    @dataclass
    class DependentInventoryPack(InventoryPack):
        spec = capability_pack_spec_from_manifest(dependent_manifest)

        def install(
            self,
            context: CapabilityInstallContext,
        ) -> CapabilityContribution:
            context.registry.get("foundation.lookup")
            return super().install(context)

    report = validate_capability_pack(
        DependentInventoryPack(),
        manifest=dependent_manifest,
        dependency_manifests=(foundation_manifest,),
        dependency_packs=(FoundationPack(),),
    )

    assert report.passed is True
    assert report.registered_handlers == [
        "foundation.lookup",
        "inventory.lookup",
    ]
