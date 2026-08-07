from __future__ import annotations

import asyncio
from typing import Any, Callable

from deepkeel.model_failures import provider_fingerprint
from deepkeel.model_invocations import ModelInvocation, ModelTurn
from deepkeel.model_native_provider import NativeChatProviderAdapter
from deepkeel.model_provider_contracts import AsyncModelProviderAdapter, ModelProviderAdapter


def _as_provider_adapter(provider: Any) -> ModelProviderAdapter | AsyncModelProviderAdapter:
    return (
        provider
        if isinstance(provider, (ModelProviderAdapter, AsyncModelProviderAdapter))
        else NativeChatProviderAdapter(provider)
    )


def _adapter_fingerprint(
    provider: ModelProviderAdapter | AsyncModelProviderAdapter,
) -> tuple[str, str]:
    info = provider.info
    return (info.provider_id, info.model_id)


async def _ainvoke_provider(
    provider: ModelProviderAdapter | AsyncModelProviderAdapter,
    request: ModelInvocation,
    *,
    on_text_delta: Callable[[str], None] | None = None,
) -> ModelTurn:
    if isinstance(provider, AsyncModelProviderAdapter):
        return await asyncio.wait_for(
            provider.ainvoke(request, on_text_delta=on_text_delta),
            timeout=max(1, request.request_timeout),
        )
    return await asyncio.wait_for(
        asyncio.to_thread(
            provider.invoke,
            request,
            on_text_delta=on_text_delta,
        ),
        timeout=max(1, request.request_timeout),
    )
