from pathlib import Path

from verification.readme_contract import verify_readme_contract


def test_english_and_chinese_readmes_share_the_release_contract() -> None:
    verify_readme_contract(Path(__file__).resolve().parents[1])
