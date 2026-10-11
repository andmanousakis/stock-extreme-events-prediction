from utils.config_loader import ConfigLoader

class TestConfigLoader:
    """
    Unit tests for ConfigLoader environment selection.
    """

    def test_defaults_to_dev(self, monkeypatch):
        """
        Load dev configs when APP_ENV is not set.

        Args:
            monkeypatch (pytest.MonkeyPatch): Pytest helper for temporary changes.
        """

        # Remove APP_ENV.
        monkeypatch.delenv("APP_ENV", raising = False)

        # Create the config loader.
        loader = ConfigLoader()

        # Check the environment.
        assert loader.env == "dev"

        # Check the dev log level is loaded.
        assert loader.load("config-logger")["level"] == "DEBUG"

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

    def test_env_argument_overrides_app_env(self, monkeypatch):
        """
        Prefer an explicit env argument over APP_ENV.

        Args:
            monkeypatch (pytest.MonkeyPatch): Pytest helper for temporary changes.
        """

        # Set APP_ENV to prod.
        monkeypatch.setenv("APP_ENV", "prod")

        # Create the config loader for uat.
        loader = ConfigLoader(env = "uat")

        # Check the environment.
        assert loader.env == "uat"

        # Check configs are read from the uat folder.
        assert loader.config_dir.name == "uat"
