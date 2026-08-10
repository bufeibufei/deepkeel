from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from deepkeel.graph_model_execution import ModelNodeExecution


class GraphModelNodeMixin:
    """Model-step orchestration, disclosure, hooks and completion handling."""

    async def amodel_node(
        self: Any,
        state: dict[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        return await ModelNodeExecution(self, state, config).run()
