"""Structured logging via structlog — JSON to stderr.

stderr is critical: stdout is reserved for the CLI's ScreeningResult JSON
output, so scripts can pipe `retina-screen screen ... | jq` without log
lines polluting the data stream.
"""

import logging
import sys

import structlog

__all__ = ["setup_logging"]


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog to emit JSON log lines to stderr at the given level.

    Idempotent: safe to call once per process or once per test.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # stdlib logging is configured too, for any third-party libs (boto3, etc.)
    # that use the standard logger.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
        force=True,
    )

    # structlog has its own LoggerFactory; the default writes to stdout, which
    # would corrupt the CLI's stdout-as-data output contract. Pin it to stderr.
    structlog.reset_defaults()
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )
