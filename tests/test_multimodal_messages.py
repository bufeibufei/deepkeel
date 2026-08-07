import pytest
from pydantic import ValidationError

from deepkeel.budget import InMemoryBudgetLedger
from deepkeel.context import build_initial_messages
from deepkeel.model import RoutedModelGateway, provider_messages_from_agent
from deepkeel.model_capabilities import ModelCapabilities
from deepkeel.model_routing import ModelStepContext
from deepkeel.policy import DefaultPolicyEngine
from deepkeel.runtime_sdk import AgentMessage, MessageContentPart, RuntimeRequest


def _image_part(reference_id: str = "attachment-1") -> MessageContentPart:
    return MessageContentPart(
        type="image",
        uri=f"attachment://{reference_id}",
        reference_id=reference_id,
        media_type="image/jpeg",
        width=1280,
        height=960,
    )


def test_message_content_part_rejects_inline_or_local_binary_payloads() -> None:
    with pytest.raises(ValidationError, match="inline or local-file"):
        MessageContentPart(
            type="image",
            uri="data:image/png;base64,AAAA",
            media_type="image/png",
        )

    with pytest.raises(ValidationError, match="inline or local-file"):
        MessageContentPart(type="image", uri="file:///tmp/private.jpg")

    with pytest.raises(ValidationError, match="inline binary"):
        MessageContentPart(
            type="image",
            uri="attachment://attachment-1",
            metadata={"payload": b"private-image"},
        )


def test_provider_messages_translate_opaque_image_references_without_duplication() -> None:
    message = AgentMessage(
        id="message-1",
        role="user",
        content="请看这张图片",
        content_parts=[
            MessageContentPart(type="text", text="请看这张图片"),
            _image_part(),
        ],
    )

    payload = provider_messages_from_agent([message])

    assert payload == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请看这张图片"},
                {
                    "type": "image_url",
                    "image_url": {"url": "attachment://attachment-1"},
                },
            ],
        }
    ]


def test_initial_messages_restore_history_media_and_merge_current_input_parts() -> None:
    image = _image_part()
    messages = build_initial_messages(
        "分析一下",
        {},
        {
            "recent_messages": [
                {
                    "id": "persisted-message",
                    "role": "user",
                    "content": "分析一下",
                }
            ]
        },
        input_parts=[image],
    )

    assert len(messages) == 1
    assert messages[0].id == "persisted-message"
    assert messages[0].content_parts == [image]


def test_image_only_history_is_preserved_in_initial_messages() -> None:
    messages = build_initial_messages(
        "继续分析上一张图片",
        {},
        {
            "recent_messages": [
                {
                    "id": "image-turn",
                    "role": "user",
                    "content": "",
                    "content_parts": [_image_part("image-only")],
                }
            ]
        },
    )

    assert messages[0].id == "image-turn"
    assert messages[0].content == ""
    assert messages[0].content_parts[0].uri == "attachment://image-only"
    assert messages[-1].content == "继续分析上一张图片"


def test_runtime_request_serializes_media_references_without_binary_data() -> None:
    request = RuntimeRequest(question="看看布局", input_parts=[_image_part()])

    serialized = request.model_dump_json()

    assert "attachment://attachment-1" in serialized
    assert "data:image" not in serialized
    assert ModelCapabilities(supports_image_input=True).supports_image_input is True


class _Provider:
    def __init__(self, model: str, role: str, *, supports_images: bool) -> None:
        self.model = model
        self.model_role = role
        self.base_url = "https://provider.example/v1"
        self.calls = 0
        self.model_capabilities = {
            "supports_image_input": supports_images,
            "source": "test_catalog",
        }

    def complete_chat(self, *_args, **_kwargs):
        self.calls += 1
        return {
            "message": {"role": "assistant", "content": self.model},
            "finish_reason": "stop",
            "model": self.model,
        }


def test_multimodal_routing_skips_models_without_declared_image_support() -> None:
    reasoning = _Provider("text-only", "reasoning", supports_images=False)
    vision = _Provider("vision", "fast", supports_images=True)
    routes: list[dict] = []
    gateway = RoutedModelGateway(
        {"reasoning": reasoning, "fast": vision},
        router=None,
        policy_engine=DefaultPolicyEngine(),
        budget_ledger=InMemoryBudgetLedger(),
    )
    message = AgentMessage(
        id="message-image",
        role="user",
        content="看看图片",
        content_parts=[_image_part()],
    )

    result = gateway.run_turn(
        [message],
        tools=[],
        step_context=ModelStepContext(
            run_id="run-image",
            user_id="user-image",
            thread_id="thread-image",
            turn_id="turn-image",
            step_index=0,
            message_count=1,
            observation_count=0,
            tool_result_count=0,
            available_roles=("reasoning", "fast"),
            model_policy={"mode": "adaptive", "failure_policy": "auto_fallback"},
        ),
        on_route=routes.append,
    )

    assert result.content == "vision"
    assert reasoning.calls == 0
    assert vision.calls == 1
    assert routes[-1]["required_capabilities"]["image_input"] is True
