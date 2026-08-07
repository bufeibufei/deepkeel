from deepkeel.adapter_sdk import (
    HarnessRuntimeBuilder,
    ModelInvocation,
    ModelProviderInfo,
    ModelTurn,
    RuntimePorts,
)
from deepkeel.extension_sdk import (
    EntryToolActivationDecision,
    EntryToolActivationRequest,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolSpec,
)
from deepkeel.runtime_sdk import PendingAction, RuntimeRequest, ToolCall, ToolResult


class EntryToolModel:
    info = ModelProviderInfo(
        provider_id="test.entry-tool",
        model_id="entry-tool-v1",
        model_role="reasoning",
    )

    def invoke(self, request: ModelInvocation, *, on_text_delta=None) -> ModelTurn:
        return ModelTurn(
            tool_calls=[
                ToolCall(
                    id="cast-call",
                    name="workflow.start",
                    arguments={"question": "model rewrite"},
                )
            ],
            finish_reason="tool_calls",
            model_id=self.info.model_id,
            model_role=self.info.model_role,
        )


class WorkflowEntryActivator:
    def activate(
        self,
        request: EntryToolActivationRequest,
    ) -> EntryToolActivationDecision | None:
        call = request.tool_calls[0]
        if call.name != "workflow.start":
            return None
        activation = {
            "skill_id": "workflow",
            "version": "1.0",
            "kind": "workflow",
            "label": "Workflow",
            "source": "model",
            "invocation_id": f"model:{request.run_id}:{call.id}",
            "explicit": False,
            "phase": "activating",
            "allowed_tools": ["workflow.start"],
            "required_tools": ["workflow.start"],
            "completed_tools": [],
            "completion_policy": {
                "waiting_statuses": ["waiting_user_action"],
            },
        }
        normalized = call.model_copy(
            update={"arguments": {"question": request.question}}
        )
        return EntryToolActivationDecision(
            skill_activation=activation,
            tool_calls=(normalized,),
        )


def test_entry_tool_promotes_skill_inside_first_react_turn() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(
                name="workflow.start",
                parameters_schema={
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
                requires_user_action=True,
            )
        ]
    )
    executor = ToolExecutor(registry)

    def start(call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        assert call.arguments["question"] == "original user question"
        assert context.metadata["skill_activation"]["skill_id"] == "workflow"
        return ToolResult(
            call=call,
            status="requires_user_action",
            summary="Continue in the workflow UI.",
            pending_action=PendingAction(
                id="pending-1",
                run_id=context.run_id,
                tool_call_id=call.id,
                action_type="workflow",
                prompt="Continue?",
            ),
        )

    executor.register("workflow.start", start)
    runtime = HarnessRuntimeBuilder(registry, executor).with_ports(
        RuntimePorts(entry_tool_skill_activator=WorkflowEntryActivator())
    ).build()

    result = runtime.run(
        RuntimeRequest(
            question="original user question",
            user_id="user-1",
            run_id="run-1",
            thread_id="thread-1",
        ),
        provider=EntryToolModel(),
    )

    assert result.status.value == "waiting_user_action"
    assert result.skill_activation["skill_id"] == "workflow"
    assert result.skill_activation["source"] == "model"
    assert result.skill_activation["completed_tools"] == ["workflow.start"]
    activation_event = next(
        event for event in result.events if event.event_type == "skill.activated"
    )
    assert activation_event.payload["entry_tool_names"] == ["workflow.start"]
