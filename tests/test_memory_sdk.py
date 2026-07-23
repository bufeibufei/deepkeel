from harness_core.memory_sdk import (
    MemoryClaim,
    MemoryEvidence,
    MemoryMutation,
    MemoryQuery,
    MemorySearchHit,
    MemorySearchPage,
)


def test_memory_contracts_round_trip_without_business_types() -> None:
    claim = MemoryClaim(
        user_id="user-1",
        domain="career",
        predicate="job_search_target",
        value="优先寻找 Agent 工程岗位",
    )
    evidence = MemoryEvidence(source_type="conversation", text="我正在找 Agent 工程岗位")
    mutation = MemoryMutation(action="create", claim=claim, evidence=[evidence])
    page = MemorySearchPage(
        hits=[MemorySearchHit(claim=claim, lexical_score=0.8, score=0.8)],
        trace={"candidate_count": 1},
    )

    assert mutation.claim is not None
    assert mutation.claim.predicate == "job_search_target"
    assert MemoryQuery(user_id="user-1", text="工作").limit == 8
    assert page.hits[0].claim.value == "优先寻找 Agent 工程岗位"
