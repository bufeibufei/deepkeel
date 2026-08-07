from deepkeel.skills import SkillPolicy
from deepkeel.workflow_policy import evaluate_workflow_completion


def test_workflow_skill_policy_preserves_durable_constraints():
    policy = SkillPolicy.from_snapshot(
        {
            "skill_id": "liuyao_divination",
            "version": "1",
            "kind": "workflow",
            "explicit": True,
            "allowed_tools": ["liuyao.start_cast", "liuyao.read_result"],
            "required_tools": ["liuyao.start_cast"],
            "prompt_instructions": "将问题整理成一事一占。",
        }
    )

    assert policy.active is True
    assert policy.durable is True
    assert policy.allows_tool("liuyao.start_cast") is True
    assert policy.allows_tool("bazi.generate_reading") is False
    assert policy.runtime_snapshot()["prompt_instructions"] == "将问题整理成一事一占。"


def test_unselected_skill_does_not_restrict_general_agent_tools():
    policy = SkillPolicy.from_snapshot(None)

    assert policy.active is False
    assert policy.allows_tool("rag.search_literature") is True


def test_skill_policy_exposes_bounded_delegation_contract():
    policy = SkillPolicy.from_snapshot(
        {
            "skill_id": "review_workflow",
            "delegation_policy": {
                "enabled": True,
                "allowed_agents": ["review.risk", "review.facts"],
                "max_tasks": 2,
                "max_concurrency": 2,
                "max_model_calls": 5,
                "max_tool_calls": 3,
            },
        }
    )

    assert policy.delegation.enabled is True
    assert policy.delegation.allows_agent("review.risk") is True
    assert policy.delegation.allows_agent("review.unknown") is False
    assert policy.delegation.runtime_snapshot()["max_model_calls"] == 5


def test_workflow_completion_requires_every_tool_and_artifact():
    policy = SkillPolicy.from_snapshot(
        {
            "skill_id": "report_workflow",
            "kind": "workflow",
            "required_tools": ["demo.build_report"],
            "output_contract": {"requires_artifact": "report"},
        }
    )

    missing_artifact = evaluate_workflow_completion(
        policy,
        {
            "tool_results": [{"name": "demo.build_report", "status": "succeeded"}],
            "artifacts": [],
        },
    )
    missing_tool = evaluate_workflow_completion(
        policy,
        {
            "tool_results": [],
            "artifacts": [{"artifact_type": "report"}],
        },
    )
    complete = evaluate_workflow_completion(
        policy,
        {
            "tool_results": [{"name": "demo.build_report", "status": "succeeded"}],
            "artifacts": [{"artifact_type": "report"}],
        },
    )

    assert policy.required_artifacts == frozenset({"report"})
    assert missing_artifact.allowed is False
    assert missing_artifact.missing_tools == ()
    assert missing_artifact.missing_artifacts == ("report",)
    assert missing_tool.missing_tools == ("demo.build_report",)
    assert missing_tool.missing_artifacts == ()
    assert complete.allowed is True


def test_workflow_completion_rejects_pending_tools_and_non_terminal_artifacts():
    policy = SkillPolicy.from_snapshot(
        {
            "skill_id": "report_workflow",
            "kind": "workflow",
            "required_tools": ["demo.build_report"],
            "output_contract": {"requires_artifact": "report"},
        }
    )

    decision = evaluate_workflow_completion(
        policy,
        {
            "tool_results": [
                {"name": "demo.build_report", "status": "waiting_async"}
            ],
            "artifacts": [
                {
                    "artifact_type": "report",
                    "data": {"status": "running"},
                }
            ],
        },
    )

    assert decision.allowed is False
    assert decision.missing_tools == ("demo.build_report",)
    assert decision.missing_artifacts == ("report",)


def test_workflow_completion_accepts_one_tool_from_each_required_group():
    policy = SkillPolicy.from_snapshot(
        {
            "skill_id": "bazi_reading",
            "kind": "workflow",
            "required_tool_groups": [
                ["bazi.generate_reading", "bazi.generate_temporary_reading"]
            ],
            "output_contract": {"requires_artifact": "bazi_report"},
        }
    )

    complete = evaluate_workflow_completion(
        policy,
        {
            "tool_results": [
                {"name": "bazi.generate_reading", "status": "succeeded"}
            ],
            "artifacts": [{"artifact_type": "bazi_report"}],
        },
    )
    missing = evaluate_workflow_completion(
        policy,
        {
            "tool_results": [],
            "artifacts": [{"artifact_type": "bazi_report"}],
        },
    )

    assert complete.allowed is True
    assert missing.allowed is False
    assert missing.missing_tools == ()
    assert missing.missing_tool_groups == (
        ("bazi.generate_reading", "bazi.generate_temporary_reading"),
    )
