from utils.config_loader import ConfigLoader

class TestConfigLoader:
    """
    Unit tests for ConfigLoader environment selection.
    """

    def test_reads_app_env(self, monkeypatch):
        """
        Load the environment named by APP_ENV.

        Args:
            monkeypatch (pytest.MonkeyPatch): Pytest helper for temporary changes.
        """

        # Set APP_ENV to prod.
        monkeypatch.setenv("APP_ENV", "prod")

        # Create the config loader.
        loader = ConfigLoader()

        # Check the environment.
        assert loader.env == "prod"

        # Check the prod start date is loaded.
        assert loader.load("config-retrieval")["request"]["start"] == "2015-01-01"
