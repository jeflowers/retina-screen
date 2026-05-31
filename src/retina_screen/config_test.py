"""Tests for env-driven config + the confidence-threshold rule.

Acceptance #3: switching to a backend without its required secret must fail
LOUDLY at startup with MissingSecretError — never mid-pipeline.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from retina_screen.config import (
    CLASSIFIER_OPTIONS,
    SUMMARIZER_OPTIONS,
    Config,
    MissingSecretError,
    apply_confidence_floor,
    build_classifier,
    build_summarizer,
    load_config,
)
from retina_screen.dto import Classification, ConditionProbability
from retina_screen.ml.classifier import Classifier, MockClassifier
from retina_screen.ml.summarizer import MockSummarizer, Summarizer


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


# ─── Factory: build_classifier / build_summarizer ────────────────────────────


def _mock_config(
    classifier: str = "mock",
    summarizer: str = "mock",
    *,
    sagemaker_endpoint: str | None = None,
    bedrock_classifier_model_id: str | None = None,
    bedrock_summarizer_model_id: str | None = None,
    anthropic_api_key: str | None = None,
) -> Config:
    return Config(
        classifier=classifier,  # type: ignore[arg-type]
        summarizer=summarizer,  # type: ignore[arg-type]
        confidence_threshold=0.7,
        aws_region="us-west-2",
        raw_image_bucket=None,
        results_table=None,
        eventbridge_bus=None,
        sagemaker_endpoint=sagemaker_endpoint,
        bedrock_classifier_model_id=bedrock_classifier_model_id,
        bedrock_summarizer_model_id=bedrock_summarizer_model_id,
        anthropic_api_key=anthropic_api_key,
        anthropic_model="claude-opus-4-7",
    )


def test_build_classifier_returns_mock_when_selected() -> None:
    classifier = build_classifier(_mock_config(classifier="mock"))
    assert isinstance(classifier, MockClassifier)
    assert isinstance(classifier, Classifier)


def test_build_summarizer_returns_mock_when_selected() -> None:
    summarizer = build_summarizer(_mock_config(summarizer="mock"))
    assert isinstance(summarizer, MockSummarizer)
    assert isinstance(summarizer, Summarizer)


def test_build_classifier_returns_sagemaker_when_selected(
    mocker: MockerFixture,
) -> None:
    """boto3.client patched so no real client / socket is created."""
    from retina_screen.ml.classifier import SageMakerClassifier

    fake_client = MagicMock()
    mocker.patch("retina_screen.config.boto3.client", return_value=fake_client)

    classifier = build_classifier(
        _mock_config(classifier="sagemaker", sagemaker_endpoint="my-endpoint")
    )
    assert isinstance(classifier, SageMakerClassifier)
    assert classifier.endpoint_name == "my-endpoint"
    assert classifier.client is fake_client


def test_build_classifier_returns_bedrock_when_selected(
    mocker: MockerFixture,
) -> None:
    from retina_screen.ml.classifier import BedrockImageClassifier

    fake_client = MagicMock()
    mocker.patch("retina_screen.config.boto3.client", return_value=fake_client)

    classifier = build_classifier(
        _mock_config(
            classifier="bedrock",
            bedrock_classifier_model_id="anthropic.claude-3",
        )
    )
    assert isinstance(classifier, BedrockImageClassifier)
    assert classifier.model_id == "anthropic.claude-3"


def test_build_summarizer_returns_bedrock_when_selected(
    mocker: MockerFixture,
) -> None:
    from retina_screen.ml.summarizer import BedrockSummarizer

    fake_client = MagicMock()
    mocker.patch("retina_screen.config.boto3.client", return_value=fake_client)

    summarizer = build_summarizer(
        _mock_config(
            summarizer="bedrock",
            bedrock_summarizer_model_id="anthropic.claude-3-haiku",
        )
    )
    assert isinstance(summarizer, BedrockSummarizer)
    assert summarizer.model_id == "anthropic.claude-3-haiku"


def test_build_summarizer_returns_anthropic_when_selected(
    mocker: MockerFixture,
) -> None:
    """anthropic.Anthropic patched so no real API client is created."""
    from retina_screen.ml.summarizer import AnthropicSummarizer

    fake_client = MagicMock()
    mocker.patch("anthropic.Anthropic", return_value=fake_client)

    summarizer = build_summarizer(_mock_config(summarizer="anthropic", anthropic_api_key="sk-test"))
    assert isinstance(summarizer, AnthropicSummarizer)
    assert summarizer.model == "claude-opus-4-7"
    assert summarizer.client is fake_client
