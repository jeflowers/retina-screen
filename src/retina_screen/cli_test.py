"""Tests for the CLI + run_pipeline orchestration.

Acceptance #1: `uv run retina-screen screen --image samples/fundus.jpg
--encounter samples/encounter_example.yaml` runs end-to-end with mock ML
and no AWS, prints a ScreeningResult, and exits 0. The CliRunner test below
covers exactly that path.

Acceptance #2: NO network and NO AWS credentials in mock-default mode — the
autouse socket-block fixture proves this; if any code path tried to reach
the network, the test would fail with NetworkBlockedError.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from retina_screen.cli import PipelineDeps, app, run_pipeline
from retina_screen.config import Config
from retina_screen.dto import EncounterMeta, ScreeningResult
from retina_screen.ml.classifier import MockClassifier
from retina_screen.ml.summarizer import MockSummarizer

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"


def _meta() -> EncounterMeta:
    return EncounterMeta(
        encounter_id="enc_001",
        patient_pseudo_id="pt_001",
        site_id="site_A",
        eye="L",
        capture_device="topcon_nw400",
        captured_at=datetime(2026, 5, 28, 14, 30, tzinfo=UTC),
    )


def _config(
    *,
    bucket: str | None = None,
    table: str | None = None,
    bus: str | None = None,
    threshold: float = 0.7,
) -> Config:
    return Config(
        classifier="mock",
        summarizer="mock",
        confidence_threshold=threshold,
        aws_region="us-west-2",
        raw_image_bucket=bucket,
        results_table=table,
        eventbridge_bus=bus,
        sagemaker_endpoint=None,
        bedrock_classifier_model_id=None,
        bedrock_summarizer_model_id=None,
        anthropic_api_key=None,
        anthropic_model="claude-opus-4-7",
    )


# ─── run_pipeline (unit) ─────────────────────────────────────────────────────


def test_run_pipeline_no_aws_returns_result_and_skips_side_effects() -> None:
    """Mock-default mode: no clients injected, pipeline still produces a valid result."""
    deps = PipelineDeps(
        config=_config(),
        classifier=MockClassifier(),
        summarizer=MockSummarizer(),
    )
    result = run_pipeline(deps, image_bytes=b"FAKE-JPEG", meta=_meta())

    assert isinstance(result, ScreeningResult)
    assert result.meta.encounter_id == "enc_001"
    assert result.final_referable in ("yes", "no", "review")
    assert result.classification.model_name == "mock-classifier"
    assert "screening aid" in result.summary.disclaimer


def test_run_pipeline_with_all_aws_clients_invokes_each_step() -> None:
    s3 = MagicMock()
    table = MagicMock()
    events = MagicMock()
    events.put_events.return_value = {"FailedEntryCount": 0, "Entries": [{"EventId": "x"}]}

    deps = PipelineDeps(
        config=_config(bucket="raw-bucket", table="results-table", bus="bus-main"),
        classifier=MockClassifier(),
        summarizer=MockSummarizer(),
        s3_client=s3,
        dynamodb_table=table,
        events_client=events,
    )

    result = run_pipeline(deps, image_bytes=b"FAKE-JPEG", meta=_meta())

    assert isinstance(result, ScreeningResult)
    s3.put_object.assert_called_once()
    table.put_item.assert_called_once()
    events.put_events.assert_called_once()


def test_run_pipeline_applies_confidence_floor() -> None:
    """Below-threshold confidence is forced to review regardless of what the model said."""

    class _LowConfidenceClassifier:
        @property
        def name(self) -> str:
            return "low-confidence-mock"

        def classify(self, image_bytes: bytes, meta: EncounterMeta) -> Any:
            del image_bytes, meta
            from retina_screen.dto import Classification, ConditionProbability

            return Classification(
                conditions=(
                    ConditionProbability(name="normal", probability=0.9),
                    ConditionProbability(name="other", probability=0.1),
                ),
                referable="no",  # model says no, but confidence is low
                model_name="low-confidence-mock",
                model_version="0.1.0",
                confidence=0.2,  # well below threshold 0.7
            )

    deps = PipelineDeps(
        config=_config(threshold=0.7),
        classifier=_LowConfidenceClassifier(),
        summarizer=MockSummarizer(),
    )
    result = run_pipeline(deps, image_bytes=b"x", meta=_meta())

    assert result.classification.referable == "no"  # what the model said
    assert result.final_referable == "review"  # what the safety rule produced


def test_run_pipeline_skips_emit_when_bus_unset_even_if_client_provided() -> None:
    """Defense in depth: if config has no bus name, even an injected client doesn't fire."""
    events = MagicMock()
    deps = PipelineDeps(
        config=_config(bus=None),  # no bus configured
        classifier=MockClassifier(),
        summarizer=MockSummarizer(),
        events_client=events,
    )
    run_pipeline(deps, image_bytes=b"x", meta=_meta())
    events.put_events.assert_not_called()


# ─── CLI (end-to-end, acceptance #1 + #2) ────────────────────────────────────


def test_cli_screen_command_runs_end_to_end_with_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance #1: real sample files, mock backends, exits 0, prints valid JSON.

    The autouse socket-block fixture also proves acceptance #2 (no network).
    """
    # Ensure clean env so we get mock/mock with no AWS resources.
    for var in (
        "RETINA_CLASSIFIER",
        "RETINA_SUMMARIZER",
        "RETINA_RAW_BUCKET",
        "RETINA_RESULTS_TABLE",
        "RETINA_EVENTBRIDGE_BUS",
        "ANTHROPIC_API_KEY",
        "SAGEMAKER_ENDPOINT",
        "BEDROCK_CLASSIFIER_MODEL_ID",
        "BEDROCK_SUMMARIZER_MODEL_ID",
    ):
        monkeypatch.delenv(var, raising=False)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "screen",
            "--image",
            str(SAMPLES_DIR / "fundus.jpg"),
            "--encounter",
            str(SAMPLES_DIR / "encounter_example.yaml"),
        ],
    )

    assert result.exit_code == 0, f"stderr was:\n{result.stderr}"

    # stdout is the JSON ScreeningResult; stderr holds the structlog lines.
    parsed = json.loads(result.stdout)
    assert parsed["meta"]["encounter_id"] == "enc_2026_05_28_0001"
    assert parsed["meta"]["site_id"] == "site_guam_clinic_01"
    assert parsed["classification"]["model_name"] == "mock-classifier"
    assert "screening aid" in parsed["summary"]["disclaimer"]
    assert parsed["final_referable"] in ("yes", "no", "review")


def test_cli_screen_command_exits_nonzero_on_missing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance #3 surfaced through the CLI: clear error, exit code 1, no traceback."""
    monkeypatch.setenv("RETINA_SUMMARIZER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "screen",
            "--image",
            str(SAMPLES_DIR / "fundus.jpg"),
            "--encounter",
            str(SAMPLES_DIR / "encounter_example.yaml"),
        ],
    )

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.stderr


def test_cli_screen_command_exits_nonzero_on_unimplemented_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting a real backend (Phase F territory) fails loud, not silently."""
    monkeypatch.setenv("RETINA_CLASSIFIER", "sagemaker")
    monkeypatch.setenv("SAGEMAKER_ENDPOINT", "fake-endpoint")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "screen",
            "--image",
            str(SAMPLES_DIR / "fundus.jpg"),
            "--encounter",
            str(SAMPLES_DIR / "encounter_example.yaml"),
        ],
    )

    assert result.exit_code == 1
    assert "not yet implemented" in result.stderr.lower() or "phase f" in result.stderr.lower()
