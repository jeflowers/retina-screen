"""Tests for env-driven config + the confidence-threshold rule.

Acceptance #3: switching to a backend without its required secret must fail
LOUDLY at startup with MissingSecretError — never mid-pipeline.
"""

from datetime import UTC, datetime

import pytest

from retina_screen.config import (
    CLASSIFIER_OPTIONS,
    SUMMARIZER_OPTIONS,
    MissingSecretError,
    apply_confidence_floor,
    load_config,
)
from retina_screen.dto import Classification, ConditionProbability


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every RETINA_* and backend-secret env var so each test starts clean."""
    for var in (
        "RETINA_CLASSIFIER",
        "RETINA_SUMMARIZER",
        "RETINA_CONFIDENCE_THRESHOLD",
        "RETINA_RAW_BUCKET",
        "RETINA_RESULTS_TABLE",
        "RETINA_EVENTBRIDGE_BUS",
        "SAGEMAKER_ENDPOINT",
        "BEDROCK_CLASSIFIER_MODEL_ID",
        "BEDROCK_SUMMARIZER_MODEL_ID",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "AWS_REGION",
    ):
        monkeypatch.delenv(var, raising=False)


def _classification(confidence: float, referable: str = "yes") -> Classification:
    return Classification(
        conditions=(ConditionProbability(name="x", probability=0.5),),
        referable=referable,  # type: ignore[arg-type]
        model_name="m",
        model_version="1",
        confidence=confidence,
    )


# ─── Defaults ────────────────────────────────────────────────────────────────


def test_default_config_is_mock_mock(clean_env: None) -> None:
    config = load_config()
    assert config.classifier == "mock"
    assert config.summarizer == "mock"
    assert config.confidence_threshold == 0.7
    assert config.anthropic_api_key is None
    assert config.sagemaker_endpoint is None


def test_classifier_and_summarizer_option_lists() -> None:
    assert "mock" in CLASSIFIER_OPTIONS
    assert "mock" in SUMMARIZER_OPTIONS
    assert "anthropic" in SUMMARIZER_OPTIONS
    assert "sagemaker" in CLASSIFIER_OPTIONS


# ─── Unknown values fail loud ────────────────────────────────────────────────


def test_unknown_classifier_raises(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETINA_CLASSIFIER", "nonsense")
    with pytest.raises(MissingSecretError, match="RETINA_CLASSIFIER"):
        load_config()


def test_unknown_summarizer_raises(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETINA_SUMMARIZER", "openai")
    with pytest.raises(MissingSecretError, match="RETINA_SUMMARIZER"):
        load_config()


# ─── Required-when-selected secrets ──────────────────────────────────────────


def test_anthropic_summarizer_without_key_raises(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RETINA_SUMMARIZER", "anthropic")
    with pytest.raises(MissingSecretError, match="ANTHROPIC_API_KEY"):
        load_config()


def test_anthropic_summarizer_with_key_loads(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RETINA_SUMMARIZER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    config = load_config()
    assert config.summarizer == "anthropic"
    assert config.anthropic_api_key == "sk-test"


def test_sagemaker_classifier_without_endpoint_raises(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RETINA_CLASSIFIER", "sagemaker")
    with pytest.raises(MissingSecretError, match="SAGEMAKER_ENDPOINT"):
        load_config()


def test_bedrock_summarizer_without_model_raises(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RETINA_SUMMARIZER", "bedrock")
    with pytest.raises(MissingSecretError, match="BEDROCK_SUMMARIZER_MODEL_ID"):
        load_config()


# ─── Threshold parsing ───────────────────────────────────────────────────────


def test_threshold_custom_value(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETINA_CONFIDENCE_THRESHOLD", "0.55")
    assert load_config().confidence_threshold == 0.55


def test_threshold_non_numeric_raises(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETINA_CONFIDENCE_THRESHOLD", "high")
    with pytest.raises(MissingSecretError, match="not a valid float"):
        load_config()


def test_threshold_out_of_range_raises(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETINA_CONFIDENCE_THRESHOLD", "1.5")
    with pytest.raises(MissingSecretError, match=r"\[0\.0, 1\.0\]"):
        load_config()


# ─── apply_confidence_floor — safety rule ────────────────────────────────────


def test_floor_below_threshold_forces_review() -> None:
    c = _classification(confidence=0.5, referable="no")
    assert apply_confidence_floor(c, threshold=0.7) == "review"


def test_floor_above_threshold_passes_through() -> None:
    c = _classification(confidence=0.85, referable="yes")
    assert apply_confidence_floor(c, threshold=0.7) == "yes"


def test_floor_at_threshold_is_not_below() -> None:
    # threshold is exclusive of equality — at-threshold passes through.
    c = _classification(confidence=0.7, referable="no")
    assert apply_confidence_floor(c, threshold=0.7) == "no"


def test_floor_never_auto_clears_low_confidence_referable() -> None:
    # A low-confidence "no" must still go to review — that's the whole point.
    c = _classification(confidence=0.1, referable="no")
    assert apply_confidence_floor(c, threshold=0.7) == "review"
    _ = datetime.now(UTC)  # touch datetime import for completeness
