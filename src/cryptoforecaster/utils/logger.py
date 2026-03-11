"""Centralised logger configuration using loguru."""

import sys
from loguru import logger
from cryptoforecaster.config import settings


def setup_logger(level: str = settings.log_level):
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        settings.log_file,
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
    )
    return logger
