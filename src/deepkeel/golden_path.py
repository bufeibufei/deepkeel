from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, Mapping

from deepkeel.capabilities import CapabilityPack
from deepkeel.composition import (
    HarnessRuntimeBuilder,
    RuntimePorts,
    RuntimeProfile,
    RuntimeProfileName,
)
from deepkeel.runtime import HarnessRuntime
from deepkeel.runtime_api import RuntimeRequest, RuntimeResult, RuntimeStreamEvent


@dataclass(frozen=True, slots=True)
class AgentDefaults:
    """Typed identity and context defaults for the Golden Path facade."""

    user_id: str = "local-device"
    tenant_id: str = ""
    namespace: str = "default"
    agent_entrypoint_id: str = ""
    agent_entrypoint_version: str = ""
    short_context: Mapping[str, Any] = field(default_factory=dict)
    context_bundle: Mapping[str, Any] = field(default_factory=dict)
    skill_activation: Mapping[str, Any] = field(default_factory=dict)
    model_policy: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentHarness:
    """Small public facade for the common build-and-run integration path.

    Advanced Hosts retain full access through ``runtime`` or can wrap an
    independently composed runtime with :meth:`from_runtime`.
    """

    runtime: HarnessRuntime
    provider: Any = None
    providers: Mapping[str, Any] = field(default_factory=dict)
    defaults: AgentDefaults = field(default_factory=AgentDefaults)

    def __post_init__(self) -> None:
        self.providers = dict(self.providers)

    @classmethod
    def create(
        cls,
        *,
        provider: Any = None,
        providers: Mapping[str, Any] | None = None,
        capability_packs: Iterable[CapabilityPack] = (),
        ports: RuntimePorts | None = None,
        profile: RuntimeProfile | RuntimeProfileName = "development",
        defaults: AgentDefaults | None = None,
        max_steps: int = 12,
        max_parallel_tools: int = 4,
    ) -> "AgentHarness":
        builder = HarnessRuntimeBuilder(profile=profile)
        if ports is not None:
            builder.with_ports(ports)
        builder.with_max_steps(max_steps)
        builder.with_max_parallel_tools(max_parallel_tools)
        for pack in capability_packs:
            builder.add_capability_pack(pack)
        runtime = builder.build_production() if profile == "production" else builder.build()
        return cls(
            runtime=runtime,
            provider=provider,
            providers=dict(providers or {}),
            defaults=defaults or AgentDefaults(),
        )

    @classmethod
    def from_runtime(
        cls,
        runtime: HarnessRuntime,
        *,
        provider: Any = None,
        providers: Mapping[str, Any] | None = None,
        defaults: AgentDefaults | None = None,
    ) -> "AgentHarness":
        return cls(
            runtime=runtime,
            provider=provider,
            providers=dict(providers or {}),
            defaults=defaults or AgentDefaults(),
        )

    def request(self, question: str, **overrides: Any) -> RuntimeRequest:
        values: dict[str, Any] = {
            "question": question,
            "user_id": self.defaults.user_id,
            "tenant_id": self.defaults.tenant_id,
            "namespace": self.defaults.namespace,
            "agent_entrypoint_id": self.defaults.agent_entrypoint_id,
            "agent_entrypoint_version": self.defaults.agent_entrypoint_version,
            "short_context": dict(self.defaults.short_context),
            "context_bundle": dict(self.defaults.context_bundle),
            "skill_activation": dict(self.defaults.skill_activation),
            "model_policy": dict(self.defaults.model_policy),
        }
        values.update(overrides)
        return RuntimeRequest(**values)

    def run(
        self,
        request: str | RuntimeRequest,
        *,
        provider: Any = None,
        providers: Mapping[str, Any] | None = None,
        session: Any = None,
        event_sink: Any = None,
        **request_overrides: Any,
    ) -> RuntimeResult:
        prepared = self._resolve_request(request, request_overrides)
        return self.runtime.run(
            prepared,
            provider=self.provider if provider is None else provider,
            providers=self._resolve_providers(providers),
            session=session,
            event_sink=event_sink,
        )

    async def arun(
        self,
        request: str | RuntimeRequest,
        *,
        provider: Any = None,
        providers: Mapping[str, Any] | None = None,
        session: Any = None,
        event_sink: Any = None,
        **request_overrides: Any,
    ) -> RuntimeResult:
        prepared = self._resolve_request(request, request_overrides)
        return await self.runtime.arun(
            prepared,
            provider=self.provider if provider is None else provider,
            providers=self._resolve_providers(providers),
            session=session,
            event_sink=event_sink,
        )

    async def astream(
        self,
        request: str | RuntimeRequest,
        *,
        provider: Any = None,
        providers: Mapping[str, Any] | None = None,
        session: Any = None,
        **request_overrides: Any,
    ) -> AsyncIterator[RuntimeStreamEvent]:
        prepared = self._resolve_request(request, request_overrides)
        stream = self.runtime.astream(
            prepared,
            provider=self.provider if provider is None else provider,
            providers=self._resolve_providers(providers),
            session=session,
        )
        try:
            async for event in stream:
                yield event
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                await close()

    def _resolve_request(
        self,
        request: str | RuntimeRequest,
        overrides: Mapping[str, Any],
    ) -> RuntimeRequest:
        if isinstance(request, RuntimeRequest):
            if overrides:
                raise ValueError("request overrides are only supported when request is a string")
            return request
        return self.request(request, **dict(overrides))

    def _resolve_providers(self, providers: Mapping[str, Any] | None) -> dict[str, Any] | None:
        resolved = dict(self.providers if providers is None else providers)
        return resolved or None


__all__ = ["AgentDefaults", "AgentHarness"]
