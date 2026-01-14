"""
Structured logging configuration for AdTech LLM.

Uses structlog for consistent, structured logging across all components.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from src.config.settings import settings


def setup_logging() -> None:
    """Configure structured logging for the application."""
    # Determine processors based on environment
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if settings.monitoring.log_format == "json":
        # JSON format for production
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Human-readable format for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.monitoring.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.monitoring.log_level.upper()),
    )


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually module name)
        **initial_context: Initial context values to bind

    Returns:
        Configured structlog logger
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


class LoggerMixin:
    """Mixin class to add logging capability to any class."""

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger bound to class name."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
