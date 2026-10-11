import logging
from pathlib import Path
from rich.logging import RichHandler
from utils.config_loader import ConfigLoader

class Logger:
    """
    Build loggers that write colored messages to the console and plain messages to a log file.
    """

    def __init__(self):
        """
        Load the logger configs.
        """

        # Load configs.
        self.config = ConfigLoader().load("config-logger")

        # Get the console configs.
        self.console_config = self.config["console"]

        # Get the file configs.
        self.file_config = self.config["file"]

    def console_handler(self) -> RichHandler:
        """
        Create a handler that prints colored messages to the console.

        Returns:
            RichHandler: Console handler.
        """

        # Create the rich console handler.
        handler = RichHandler(show_path = False)

        # Set the console message format.
        handler.setFormatter(logging.Formatter(self.console_config["format"]))

        # Return handler.
        return handler

    def file_handler(self) -> logging.FileHandler:
        """
        Create a handler that appends messages to the log file.

        Returns:
            logging.FileHandler: File handler.
        """

        # Create the log directory if it does not exist.
        Path(self.file_config["path"]).parent.mkdir(parents = True, exist_ok = True)

        # Create the file handler.
        handler = logging.FileHandler(self.file_config["path"])

        # Set the file message format.
        handler.setFormatter(logging.Formatter(self.file_config["format"]))

        # Return handler.
        return handler

    def add_step(self, record: logging.LogRecord = None) -> bool:
        """
        Set the record's step name to the logging method's name unless the error handler already set it.

        Args:
            record (logging.LogRecord): Log record being processed.

        Returns:
            bool: Always True, so the record is kept.
        """

        # Fall back to the name of the method that logged.
        record.step = getattr(record, "step", record.funcName)

        # Keep the record.
        return True

    def get_logger(self, name: str = None) -> logging.Logger:
        """
        Get a named logger with console and file handlers attached.

        Args:
            name (str): Logger name, usually the class name.

        Returns:
            logging.Logger: Configured logger.
        """

        # Get the logger by name.
        logger = logging.getLogger(name)

        # Attach handlers only once per logger.
        if not logger.handlers:

            # Fill in the step name on every record.
            logger.addFilter(self.add_step)

            # Attach the console handler.
            logger.addHandler(self.console_handler())

            # Attach the file handler.
            logger.addHandler(self.file_handler())

        # Set the minimum level to log.
        logger.setLevel(self.config["level"])

        # Stop messages from being printed twice by the root logger.
        logger.propagate = False

        # Return logger.
        return logger
