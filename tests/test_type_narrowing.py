from collections import UserDict

from deepkeel.type_narrowing import (
    as_dict,
    as_dict_list,
    as_list,
    as_optional_dict,
)


def test_mapping_boundaries_return_detached_string_keyed_dicts() -> None:
    source = UserDict({"name": "alpha", 7: "numeric"})

    narrowed = as_dict(source)

    assert narrowed == {"name": "alpha", "7": "numeric"}
    assert narrowed is not source
    assert as_optional_dict(source) == narrowed
    assert as_optional_dict("invalid") is None


def test_sequence_boundaries_filter_only_when_contract_requires_dicts() -> None:
    source = [{"id": 1}, "ignored", UserDict({"id": 2}), 3]

    assert as_list(source) == source
    assert as_list((1, 2)) == [1, 2]
    assert as_list("not-a-sequence-payload") == []
    assert as_dict_list(source) == [{"id": 1}, {"id": 2}]
    assert as_dict_list(None) == []
