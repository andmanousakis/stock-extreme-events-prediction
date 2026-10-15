import pytest

@pytest.fixture(autouse = True)
def no_mlflow(monkeypatch):
    """
    Switch MLflow tracking off in every test, so tests never log runs or register models on the real tracking server.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest helper for temporary changes.
    """

    # Remove the MLflow credentials from the test's environment.
    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):

        # Remove the variable if it is set.
        monkeypatch.delenv(name, raising = False)
