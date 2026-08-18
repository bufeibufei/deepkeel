"""Optional public SDK for A2A 1.0 remote specialist interoperability."""

from deepkeel.contrib.a2a import (
    A2A_PROTOCOL_VERSION,
    A2AAgentCard,
    A2AAgentInterface,
    A2AAgentSkill,
    A2AArtifact,
    A2AClientPort,
    A2ADelegationExecutor,
    A2AError,
    A2AMessage,
    A2APart,
    A2AProtocolError,
    A2ARemoteAgent,
    A2ASendResponse,
    A2ATask,
    A2ATaskStatus,
    A2ATransportError,
    HttpJsonA2AClient,
)

A2A_SDK_API = (
    "A2A_PROTOCOL_VERSION",
    "A2AAgentCard",
    "A2AAgentInterface",
    "A2AAgentSkill",
    "A2AArtifact",
    "A2AClientPort",
    "A2ADelegationExecutor",
    "A2AError",
    "A2AMessage",
    "A2APart",
    "A2AProtocolError",
    "A2ARemoteAgent",
    "A2ASendResponse",
    "A2ATask",
    "A2ATaskStatus",
    "A2ATransportError",
    "HttpJsonA2AClient",
)

__all__ = list(A2A_SDK_API)
