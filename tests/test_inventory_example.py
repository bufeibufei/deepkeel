from examples.inventory_pack import INVENTORY_MANIFEST, InventoryCapabilityPack
from deepkeel.extension_sdk import validate_capability_pack


def test_product_neutral_inventory_example_passes_conformance() -> None:
    report = validate_capability_pack(
        InventoryCapabilityPack(),
        manifest=INVENTORY_MANIFEST,
    )

    assert report.passed is True
    assert report.package_id == "example.inventory"
    assert report.permission_coverage == {
        "inventory.lookup": ["inventory.read"]
    }
