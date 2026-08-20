from __future__ import annotations

import json
from pathlib import Path

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


def test_pack_cli_scaffolds_inspects_and_validates_manifest(
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    target = tmp_path / "weather_pack"

    assert main(["pack", "init", str(target), "--package-id", "demo.weather"]) == 0
    created = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert created["files"] == ["manifest.json", "package.py", "README.md"]

    manifest = target / "manifest.json"
    assert main(["pack", "inspect", str(manifest)]) == 0
    inspected = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert inspected["manifest"]["id"] == "demo.weather"

    assert main(["pack", "validate", str(manifest)]) == 0
    validated = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert validated["ok"] is True

    monkeypatch.syspath_prepend(str(target))  # type: ignore[attr-defined]
    assert main(
        [
            "pack",
            "validate",
            str(manifest),
            "--factory",
            "package:DemoWeatherPack",
        ]
    ) == 0
    conformance = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert conformance["report"]["manifest_validated"] is True

    assert main(["pack", "digest", str(manifest), str(target / "package.py")]) == 0
    digest = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert len(digest["sha256"]) == 64


def test_pack_cli_refuses_to_replace_existing_skeleton(
    tmp_path: Path,
    capsys: object,
) -> None:
    target = tmp_path / "existing"
    assert main(["pack", "init", str(target), "--package-id", "demo.existing"]) == 0
    capsys.readouterr()  # type: ignore[attr-defined]

    assert main(["pack", "init", str(target), "--package-id", "demo.existing"]) == 2
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["error"]["code"] == "CAPABILITY_PACK_EXISTS"
