"""Logging configuration boundary."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging once the service starts."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
