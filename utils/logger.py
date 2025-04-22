# utils/logger.py
import logging
import os

def setup_logger(log_file):
    """Configura un logger para registrar métricas."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    return logger