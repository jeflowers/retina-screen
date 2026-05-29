"""Smoke tests for structlog setup.

Capture is done by monkeypatching sys.stderr to a StringIO and re-invoking
setup_logging — that way the PrintLoggerFactory binds to our capture buffer
at config time.
"""

import io
import json
import logging

import pytest
import structlog

from retina_screen.logging_setup import setup_logging


def test_setup_logging_is_idempotent() -> None:
    setup_logging()
    setup_logging()  # second call must not raise


def test_setup_logging_produces_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """A structlog log line is valid JSON with the expected keys.

    Capture by monkeypatching sys.stderr BEFORE setup_logging, so the
    PrintLoggerFactory binds to our buffer.
    """
    buf = io.StringIO()
    monkeypatch.setattr("sys.stderr", buf)

    setup_logging()
    structlog.get_logger().info("test_event", foo="bar", n=42)

    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert lines, "expected at least one log line on stderr"
    parsed = json.loads(lines[-1])
    assert parsed["event"] == "test_event"
    assert parsed["foo"] == "bar"
    assert parsed["n"] == 42
    assert parsed["level"] == "info"


def test_setup_logging_respects_level() -> None:
    setup_logging(level="WARNING")
    assert logging.getLogger().level == logging.WARNING
