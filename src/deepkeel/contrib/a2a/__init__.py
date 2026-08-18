"""Optional A2A 1.0 adapter for governed remote specialist delegation."""

from deepkeel.contrib.a2a.adapter import A2ADelegationExecutor, A2ARemoteAgent
from deepkeel.contrib.a2a.client import HttpJsonA2AClient
from deepkeel.contrib.a2a.contracts import (
    A2A_PROTOCOL_VERSION,
    A2AAgentCard,
    A2AAgentInterface,
    A2AAgentSkill,
    A2AArtifact,
    A2AClientPort,
    A2AError,
    A2AMessage,
    A2APart,
    A2AProtocolError,
    A2ASendResponse,
    A2ATask,
    A2ATaskStatus,
    A2ATransportError,
)

__all__ = [
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
]
