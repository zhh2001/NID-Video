"""Loguru wrapper. Centralizes sink configuration so callers just `from ... import logger`."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logger(
    level: str = "INFO",
    log_file: str | Path | None = None,
    rotation: str = "50 MB",
    retention: int = 5,
) -> None:
    """Install stderr (and optionally file) sinks with a project-wide format.

    Idempotent: subsequent calls reset the sinks rather than stacking them.
    """
    global _CONFIGURED
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=fmt, colorize=True, enqueue=False)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level=level,
            format=fmt,
            rotation=rotation,
            retention=retention,
            enqueue=True,
            colorize=False,
        )
    _CONFIGURED = True


def get_logger():
    """Return the project logger, configuring with INFO defaults on first use."""
    if not _CONFIGURED:
        setup_logger()
    return logger


__all__ = ["logger", "setup_logger", "get_logger"]
