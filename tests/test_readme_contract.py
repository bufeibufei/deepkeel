from pathlib import Path

from verification.readme_contract import (
    _bilingual_document_errors,
    _invalid_local_links,
    verify_readme_contract,
)


def test_english_and_chinese_readmes_share_the_release_contract() -> None:
    verify_readme_contract(Path(__file__).resolve().parents[1])


def test_local_link_validation_rejects_missing_and_escaping_targets(tmp_path: Path) -> None:
    document = tmp_path / "README.md"

    errors = _invalid_local_links(
        tmp_path,
        document,
        "[missing](docs/missing.md) [escape](../outside.md) [web](https://example.com)",
    )

    assert errors == [
        "README.md has missing local link: 'docs/missing.md'",
        "README.md link escapes repository: '../outside.md'",
    ]


def test_bilingual_document_validation_requires_linked_reciprocal_translation(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "# Guide\n\n[English](guide.md) | [简体中文](guide.zh-CN.md)\n",
        encoding="utf-8",
    )
    (docs / "guide.zh-CN.md").write_text(
        "# 指南\n\n[English](guide.md) | [简体中文](guide.zh-CN.md)\n",
        encoding="utf-8",
    )

    assert _bilingual_document_errors(
        tmp_path,
        "[Guide](docs/guide.md)",
        "[指南](docs/guide.zh-CN.md)",
    ) == []

    assert _bilingual_document_errors(
        tmp_path,
        "[Guide](docs/guide.md)",
        "No translated guide link",
    ) == ["README.zh-CN.md does not link 'docs/guide.zh-CN.md'"]
