import importlib
import pandas as pd
import pytest

model_monitoring = importlib.import_module("src.pipeline.7_model_monitoring")

def fake_scores(candidate: float = None, deployed: float = None) -> pd.DataFrame:
    """
    Build a small comparison table with each role's val and test MCC.

    Args:
        candidate (float): The candidate's val MCC.
        deployed (float): The deployed version's val MCC, or None if nothing comparable is deployed.

    Returns:
        pd.DataFrame: Fake comparison table, one row per role and split.
    """

    # Add the candidate's rows.
    rows = [{"role": "candidate", "split": "val", "mcc": candidate}, {"role": "candidate", "split": "test", "mcc": 0.0}]

    # Add the deployed version's rows if it exists.
    if deployed is not None:

        # Add its val and test rows.
        rows += [{"role": "deployed", "split": "val", "mcc": deployed}, {"role": "deployed", "split": "test", "mcc": 0.9}]

    # Return the table.
    return pd.DataFrame(rows)

@pytest.fixture(autouse = True)
def run_in_tmp_path(tmp_path, monkeypatch):
    """
    Run every test from a temporary folder so log files and outputs don't land in the project.

    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Pytest helper for temporary changes.
    """

    # Change the working directory to the temporary folder.
    monkeypatch.chdir(tmp_path)

class TestDecide:
    """
    Unit tests for ModelMonitoring.decide.
    """

    @pytest.mark.parametrize("candidate, deployed, expected", [
        (0.10, None, True),
        (0.30, 0.20, True),
        (0.10, 0.20, False),
        (0.20, 0.20, False),
    ])
    def test_promotes_only_when_better(self, candidate, deployed, expected):
        """
        Promote when nothing is deployed, otherwise only on a strictly higher val MCC. Test scores never decide.

        Args:
            candidate (float): The candidate's val MCC.
            deployed (float): The deployed version's val MCC, or None.
            expected (bool): Expected decision.
        """

        # Create the step.
        monitoring = model_monitoring.ModelMonitoring()

        # Check the decision.
        assert monitoring.decide(versions = {"candidate": "2", "deployed": "1" if deployed is not None else None}, table = fake_scores(candidate = candidate, deployed = deployed)) is expected
