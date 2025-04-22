# utils/logger.py
import logging
import os

def setup_logger(log_file: str) -> logging.Logger:
    """Sets up a logger to record metrics and messages.

    Configures a logging system that writes to a specified file with a standard
    format including timestamp, log level, and message. Creates the directory for
    the log file if it does not exist.

    Args:
        log_file (str): Path to the log file where messages will be written.

    Returns:
        logging.Logger: Configured logger instance for logging messages.
    """
    # Create the directory for the log file if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure the logging system
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Get and return the logger instance
    logger = logging.getLogger()
    return logger