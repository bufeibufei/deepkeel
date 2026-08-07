from examples.durable_approval.main import run_approval
from examples.quickstart.main import run_quickstart


def test_quickstart_runs_from_public_sdk() -> None:
    assert run_quickstart() == "DeepKeel received: Is the runtime ready?"


def test_durable_approval_suspends_and_resumes() -> None:
    assert run_approval() == ("waiting_user_action", "completed")
