from pathlib import Path
import yaml

class ConfigLoader:
    """
    Load YAML config files from the project's configs directory.
    """

    def __init__(self, config_dir: str = None):
        """
        Set the directory that config files are read from.

        Args:
            config_dir (str): Path to the configs directory. Defaults to <project_root>/configs.
        """

        # Default to the configs directory at the project root.
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parents[1] / "configs"

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
