from harness_core.memory_sdk import MemoryQuery


def test_memory_query_can_scope_retrieval_to_a_subject() -> None:
    query = MemoryQuery(
        user_id="user-1",
        subject_type="person",
        subject_id="subject-1",
        profile_id="profile-1",
        text="career",
    )

    assert query.subject_type == "person"
    assert query.subject_id == "subject-1"
