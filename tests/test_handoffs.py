from harness_core.handoffs import HandoffRegistry, HandoffSpec, standardize_pending_action_payload


def test_capability_can_register_handoff_without_changing_runtime_core():
    registry = HandoffRegistry()
    registry.register(
        "marketing.review",
        HandoffSpec(
            action_kind="marketing_review",
            noun="复盘",
            title="等待完成接待复盘",
            summary="完成后继续生成跟进建议",
            primary_label="继续复盘",
            cancel_label="取消复盘",
            handoff_view="marketing_review",
            completion_artifact_type="marketing_review_report",
        ),
    )

    action_type, payload = standardize_pending_action_payload(
        tool_name="marketing.review",
        action_type="user_action",
        payload={"customer_id": "customer-1"},
        registry=registry,
    )

    assert action_type == "tool_handoff"
    assert payload["action_kind"] == "marketing_review"
    assert payload["handoff"]["view"] == "marketing_review"
    assert payload["presentation"]["primary_label"] == "继续复盘"
    assert payload["completion"]["artifact_type"] == "marketing_review_report"
    assert payload["customer_id"] == "customer-1"


def test_unknown_handoff_uses_business_neutral_fallback_contract():
    action_type, payload = standardize_pending_action_payload(
        tool_name="unknown.external_tool",
        action_type="user_action",
        payload={"title": "确认外部操作"},
    )

    assert action_type == "tool_handoff"
    assert payload["action_kind"] == "tool_handoff"
    assert payload["presentation"]["title"] == "确认外部操作"
    assert payload["presentation"]["primary_label"] == "Continue"
