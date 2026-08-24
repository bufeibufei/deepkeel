# Inventory Capability Package

[English](README.md) | [简体中文](README.zh-CN.md)

该 Package 与任何参考 Host 都无关，用于证明业务 Package 可以只通过公开 Extension
SDK 安装，而无需修改 Core。

使用 `INVENTORY_MANIFEST` 作为声明，安装 `InventoryCapabilityPack`，并提供包含全部
必需认证场景 Tag 的 Eval Case。参见 `tests/test_inventory_example.py`。
