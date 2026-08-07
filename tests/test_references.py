from deepkeel.references import DefaultReferenceProjector


def test_default_reference_projector_is_tool_and_domain_neutral():
    projection = DefaultReferenceProjector()(
        [
            {
                "name": "external.lookup",
                "data": {
                    "query": "inventory",
                    "evidence": [{"id": "record-1", "title": "Inventory record"}],
                    "results": [{"url": "https://example.com", "title": "Public source"}],
                },
            }
        ],
        {},
    )

    assert [item["kind"] for item in projection.references] == ["record", "web"]
    assert projection.references[0]["source_tool"] == "external.lookup"
    assert projection.evidence == [projection.references[0]]
