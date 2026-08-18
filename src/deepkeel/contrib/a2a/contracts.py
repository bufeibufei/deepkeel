from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


A2A_PROTOCOL_VERSION = "1.0"
A2ATaskState = Literal[
    "TASK_STATE_UNSPECIFIED",
    "TASK_STATE_SUBMITTED",
    "TASK_STATE_WORKING",
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_REJECTED",
    "TASK_STATE_AUTH_REQUIRED",
]


class A2APart(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    text: str | None = None
    raw: str | None = None
    url: str | None = None
    data: Any = None
    filename: str = ""
    media_type: str = Field(default="", alias="mediaType")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_one_content_kind(self) -> "A2APart":
        present = sum(
            value is not None for value in (self.text, self.raw, self.url, self.data)
        )
        if present != 1:
            raise ValueError("A2A part requires exactly one content field")
        return self


class A2AMessage(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    message_id: str = Field(alias="messageId", min_length=1)
    role: Literal["ROLE_USER", "ROLE_AGENT"]
    parts: list[A2APart] = Field(min_length=1)
    context_id: str = Field(default="", alias="contextId")
    task_id: str = Field(default="", alias="taskId")
    reference_task_ids: list[str] = Field(default_factory=list, alias="referenceTaskIds")
    extensions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2AArtifact(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    artifact_id: str = Field(alias="artifactId", min_length=1)
    name: str = ""
    description: str = ""
    parts: list[A2APart] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ATaskStatus(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    state: A2ATaskState
    message: A2AMessage | None = None
    timestamp: str = ""


class A2ATask(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str = Field(min_length=1)
    context_id: str = Field(default="", alias="contextId")
    status: A2ATaskStatus
    artifacts: list[A2AArtifact] = Field(default_factory=list)
    history: list[A2AMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ASendResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    task: A2ATask | None = None
    message: A2AMessage | None = None

    @model_validator(mode="after")
    def validate_single_response(self) -> "A2ASendResponse":
        if (self.task is None) == (self.message is None):
            raise ValueError("A2A response requires exactly one task or message")
        return self


class A2AAgentInterface(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    url: str = Field(min_length=1)
    protocol_binding: str = Field(alias="protocolBinding", min_length=1)
    protocol_version: str = Field(default=A2A_PROTOCOL_VERSION, alias="protocolVersion")
    tenant: str = ""


class A2AAgentSkill(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=list, alias="inputModes")
    output_modes: list[str] = Field(default_factory=list, alias="outputModes")


class A2AAgentCard(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    supported_interfaces: list[A2AAgentInterface] = Field(
        min_length=1,
        alias="supportedInterfaces",
    )
    version: str = Field(min_length=1)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    default_input_modes: list[str] = Field(default_factory=list, alias="defaultInputModes")
    default_output_modes: list[str] = Field(default_factory=list, alias="defaultOutputModes")
    skills: list[A2AAgentSkill] = Field(default_factory=list)
    security_schemes: dict[str, Any] = Field(default_factory=dict, alias="securitySchemes")
    security_requirements: list[dict[str, Any]] = Field(
        default_factory=list,
        alias="securityRequirements",
    )


class A2AClientPort(Protocol):
    def send_message(
        self,
        message: A2AMessage,
        *,
        accepted_output_modes: list[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> A2ASendResponse: ...

    def get_task(
        self,
        task_id: str,
        *,
        history_length: int = 5,
        timeout_seconds: float | None = None,
    ) -> A2ATask: ...

    def cancel_task(
        self,
        task_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> A2ATask: ...

    def close(self) -> None: ...


class A2AError(RuntimeError):
    pass


class A2ATransportError(A2AError):
    pass


class A2AProtocolError(A2AError):
    pass
