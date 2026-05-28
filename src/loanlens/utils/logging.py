"""
Centralized logging configuration using loguru.
Import logger from here — never use print() in production code.
"""

import sys
from loguru import logger
from loanlens.config import get_settings


def setup_logging() -> None:
    """Configure loguru for the application."""
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — rotates daily, keeps 7 days
    logger.add(
        "logs/loanlens_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function} | {message}",
    )


# Export configured logger
__all__ = ["logger", "setup_logging"]
