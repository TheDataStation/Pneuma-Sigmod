import logging
import os

from .storage_config import get_storage_path


def configure_logging():
    os.makedirs(get_storage_path(), exist_ok=True)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(get_storage_path(), "pneuma.log")),
            logging.StreamHandler(),
        ],
    )
