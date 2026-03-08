"""Centralized logging configuration for the medical agentic system.

Usage:
    from backend.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting step: %s", step_name)
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_root_logger() -> None:
    """Set up root logger once on first call."""
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name, ensuring root is configured."""
    _configure_root_logger()
    return logging.getLogger(name)
