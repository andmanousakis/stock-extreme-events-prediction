import os
from pathlib import Path
import yaml

class ConfigLoader:
    """
    Load YAML config files for the current environment (dev, uat or prod) from the project's configs directory.
    """

    def __init__(self, config_dir: str = None, env: str = None):
        """
        Set the environment and the directory that config files are read from.

        Args:
            config_dir (str): Path to the configs directory. Defaults to <project_root>/configs/<env>.
            env (str): Environment name. Defaults to the APP_ENV environment variable, or dev if it is not set.
        """

        # Use the given environment, then APP_ENV, then dev.
        self.env = env or os.environ.get("APP_ENV", "dev")

        # Default to the environment's folder in the configs directory at the project root.
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parents[1] / "configs" / self.env

    def load(self, name: str = None) -> dict:
        """
        Read a config file and return its contents.

        Args:
            name (str): Config file name without the .yaml extension.

        Returns:
            dict: Parsed config values.
        """

        # Build the config file path.
        path = self.config_dir / f"{name}.yaml"

        # Read and parse the YAML file.
        with open(path, "r") as file:

            # Parse file contents.
            config = yaml.safe_load(file)

        # Return config.
        return config
