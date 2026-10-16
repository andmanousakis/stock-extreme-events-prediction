import pytest

@pytest.fixture(autouse = True)
def no_remote_services(monkeypatch):
    """
    Switch MLflow tracking and the remote storage bucket off in every test, so tests never log runs, register models
    or write tables on the real servers; tables are written locally instead.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest helper for temporary changes.
    """

    # Remove the MLflow and storage credentials from the test's environment.
    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "STORAGE_BUCKET", "AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):

        # Remove the variable if it is set.
        monkeypatch.delenv(name, raising = False)
