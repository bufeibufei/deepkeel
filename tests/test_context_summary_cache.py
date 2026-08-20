from deepkeel.adapter_conformance import verify_context_summary_cache_contract
from deepkeel.context_window import InMemoryContextSummaryCache
from deepkeel.scope import RuntimeScope


def test_in_memory_context_summary_cache_isolates_runtime_scopes() -> None:
    verify_context_summary_cache_contract(InMemoryContextSummaryCache())


def test_runtime_scope_digest_uses_the_complete_ownership_boundary() -> None:
    base = RuntimeScope(tenant_id="tenant-a", namespace="default", user_id="user-1")

    assert len(base.scope_digest) == 64
    assert base.scope_digest != RuntimeScope(
        tenant_id="tenant-b",
        namespace="default",
        user_id="user-1",
    ).scope_digest
    assert base.scope_digest != RuntimeScope(
        tenant_id="tenant-a",
        namespace="private",
        user_id="user-1",
    ).scope_digest
    assert base.scope_digest != RuntimeScope(
        tenant_id="tenant-a",
        namespace="default",
        user_id="user-2",
    ).scope_digest
