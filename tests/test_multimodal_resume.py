from deepkeel.graph_state import _apply_resume_payload


def test_clarification_resume_appends_provider_neutral_image_parts():
    state = {
        "run_id": "run-vision-resume",
        "pending_action": {"action_type": "clarification", "tool_name": "vision.read_palm"},
        "pending_async": None,
        "messages": [],
        "observations": [],
        "events": [],
        "metadata": {},
        "missing_requirements": {"tools": ["vision.read_palm"], "artifacts": []},
    }

    resumed = _apply_resume_payload(
        state,
        {
            "status": "succeeded",
            "summary": "I uploaded a clearer palm photo.",
            "data": {
                "clarification_answer": "Please use this clearer photo.",
                "content_parts": [{
                    "type": "image",
                    "uri": "attachment://image-clear-palm",
                    "reference_id": "image-clear-palm",
                    "media_type": "image/jpeg",
                    "detail": "high",
                }],
            },
        },
        {},
        source="user_action",
    )

    user_message = next(message for message in resumed["messages"] if message["role"] == "user")
    assert user_message["content"] == "Please use this clearer photo."
    assert user_message["content_parts"] == [{
        "type": "image",
        "text": "",
        "uri": "attachment://image-clear-palm",
        "reference_id": "image-clear-palm",
        "media_type": "image/jpeg",
        "detail": "high",
        "width": None,
        "height": None,
        "metadata": {},
    }]
