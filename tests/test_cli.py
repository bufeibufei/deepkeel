from __future__ import annotations

import json

from deepkeel.cli import _safe_error, main
from deepkeel.public_api import PUBLIC_API_VERSION
from deepkeel.version import PACKAGE_VERSION


def test_doctor_reports_machine_readable_runtime_status(capsys: object) -> None:
    assert main(["doctor"]) == 0

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["ok"] is True
    assert output["deepkeel_version"] == PACKAGE_VERSION
    assert output["public_api_version"] == PUBLIC_API_VERSION
    assert all(output["mandatory_dependencies"].values())


def test_postgres_command_fails_closed_when_dsn_is_missing(
    monkeypatch: object,
    capsys: object,
) -> None:
    monkeypatch.delenv("DEEPKEEL_TEST_MISSING_DSN", raising=False)  # type: ignore[attr-defined]

    assert (
        main(
            [
                "postgres",
                "status",
                "--dsn-env",
                "DEEPKEEL_TEST_MISSING_DSN",
            ]
        )
        == 2
    )

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["ok"] is False
    assert output["error"]["code"] == "POSTGRES_DSN_MISSING"


def test_cli_error_output_redacts_connection_secrets() -> None:
    error = RuntimeError(
        "postgresql://operator:secret@example.test/runtime password=secret"
    )

    message = _safe_error(error)

    assert "secret" not in message
    assert "postgresql://***" in message
    assert "password=***" in message
