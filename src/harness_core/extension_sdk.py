"""Versioned Extension SDK for tools, Skills, Capability Packs, and artifacts."""

from harness_core.capabilities import (
    ArtifactTypeSpec,
    CapabilityCatalog,
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityPack,
    CapabilityPackSpec,
    capability_pack_spec,
)
from harness_core.clarifications import (
    ClarificationRequest,
    clarification_from_validation_error,
    clarification_tool_result,
)
from harness_core.conformance import (
    CapabilityPackConformanceReport,
    validate_capability_pack,
)
from harness_core.handoffs import (
    HandoffRegistry,
    HandoffSpec,
    standardize_pending_action_payload,
)
from harness_core.prompts import harness_system_prompt
from harness_core.references import (
    DefaultReferenceProjector,
    ReferenceProjection,
    ReferenceProjector,
)
from harness_core.skill_packages import (
    ArtifactPresentationActionSpec,
    ArtifactPresentationFieldSpec,
    ArtifactPresentationSpec,
    CompiledSkillSpec,
    SkillPackageManifest,
    load_skill_packages,
    validate_skill_packages,
)
from harness_core.tool_registry import ToolRegistry, ToolSpec
from harness_core.tool_disclosure import (
    TOOL_DISCOVERY_NAME,
    ToolDescriptor,
    ToolView,
    discover_tools,
    install_tool_discovery,
    resolve_tool_view,
    tool_discovery_spec,
)
from harness_core.tool_providers import ToolProvider, ToolProviderSpec, verify_tool_provider
from harness_core.tools import (
    ToolExecutionClaim,
    ToolExecutionContext,
    ToolExecutionStore,
    ToolExecutor,
    ToolHandler,
    ToolPreflight,
)

EXTENSION_SDK_API = (
    "ArtifactPresentationActionSpec",
    "ArtifactPresentationFieldSpec",
    "ArtifactPresentationSpec",
    "ArtifactTypeSpec",
    "CapabilityCatalog",
    "CapabilityContribution",
    "CapabilityInstallContext",
    "CapabilityPack",
    "CapabilityPackConformanceReport",
    "CapabilityPackSpec",
    "ClarificationRequest",
    "CompiledSkillSpec",
    "DefaultReferenceProjector",
    "HandoffRegistry",
    "HandoffSpec",
    "ReferenceProjection",
    "ReferenceProjector",
    "SkillPackageManifest",
    "ToolExecutionClaim",
    "ToolExecutionContext",
    "ToolExecutionStore",
    "ToolExecutor",
    "ToolHandler",
    "ToolProvider",
    "ToolProviderSpec",
    "ToolPreflight",
    "ToolRegistry",
    "ToolSpec",
    "TOOL_DISCOVERY_NAME",
    "ToolDescriptor",
    "ToolView",
    "capability_pack_spec",
    "clarification_from_validation_error",
    "clarification_tool_result",
    "discover_tools",
    "harness_system_prompt",
    "install_tool_discovery",
    "load_skill_packages",
    "standardize_pending_action_payload",
    "resolve_tool_view",
    "tool_discovery_spec",
    "validate_capability_pack",
    "validate_skill_packages",
    "verify_tool_provider",
)

__all__ = list(EXTENSION_SDK_API)
