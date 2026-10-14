import shutil
import tempfile
from pathlib import Path
import pytest

@pytest.fixture
def workdir(monkeypatch):
    """
    Run the test from a fresh temporary folder and delete it, with everything written to it, afterwards.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest helper for temporary changes.

    Yields:
        Path: The temporary folder.
    """

    # Create the temporary folder.
    path = Path(tempfile.mkdtemp())

    # Run from the temporary folder so outputs are written there.
    monkeypatch.chdir(path)

    # Hand the folder to the test.
    yield path

    # Move back to the original working directory.
    monkeypatch.undo()

    # Delete the folder and all results.
    shutil.rmtree(path)
