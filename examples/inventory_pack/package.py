from __future__ import annotations

from pathlib import Path

from harness_core.extension_sdk import (
    CapabilityContribution,
    CapabilityInstallContext,
    CapabilityManifest,
    ToolSpec,
    capability_pack_spec_from_manifest,
    load_capability_manifest,
)
from harness_core.runtime_sdk import ToolCall, ToolResult
from harness_core.extension_sdk import ToolExecutionContext


INVENTORY_MANIFEST = load_capability_manifest(
    Path(__file__).with_name("manifest.json")
)


class InventoryCapabilityPack:
    """Minimal non-fate package proving that Core has no product dependency."""

    manifest: CapabilityManifest = INVENTORY_MANIFEST
    spec = capability_pack_spec_from_manifest(manifest)

    def install(
        self,
        context: CapabilityInstallContext,
    ) -> CapabilityContribution:
        context.register_tool(
            ToolSpec(
                name="inventory.lookup",
                description="Look up the available quantity for an inventory item.",
                read_only=True,
                parallel_safe=True,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "minLength": 1},
                    },
                    "required": ["item"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "quantity": {"type": "integer", "minimum": 0},
                    },
                    "required": ["item", "quantity"],
                    "additionalProperties": False,
                },
                runtime_policy={
                    "required_scopes": ["inventory.read"],
                },
            ),
            _lookup_inventory,
        )
        return CapabilityContribution(
            package_id=self.spec.package_id,
            tools=("inventory.lookup",),
        )


def _lookup_inventory(
    call: ToolCall,
    _context: ToolExecutionContext,
) -> ToolResult:
    item = str(call.arguments["item"])
    return ToolResult(
        call=call,
        status="succeeded",
        summary=f"{item} inventory loaded",
        data={"item": item, "quantity": 12},
    )
