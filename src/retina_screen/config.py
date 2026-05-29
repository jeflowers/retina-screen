"""Env-driven configuration and the confidence-threshold safety rule.

`load_config()` reads environment variables, validates that any secret required
by the selected backend is present, and fails loud at startup if a value is
unknown or missing. The confidence-threshold rule lives here because the
threshold itself is config — applying it returns a final `referable` decision.
"""

import os
from dataclasses import dataclass
from typing import Literal, cast

from retina_screen.dto import Classification, Referable

__all__ = [
    "CLASSIFIER_OPTIONS",
    "SUMMARIZER_OPTIONS",
    "ClassifierName",
    "Config",
    "MissingSecretError",
    "SummarizerName",
    "apply_confidence_floor",
    "load_config",
]


ClassifierName = Literal["mock", "sagemaker", "bedrock"]
SummarizerName = Literal["mock", "bedrock", "anthropic"]

CLASSIFIER_OPTIONS: tuple[ClassifierName, ...] = ("mock", "sagemaker", "bedrock")
SUMMARIZER_OPTIONS: tuple[SummarizerName, ...] = ("mock", "bedrock", "anthropic")


class MissingSecretError(RuntimeError):
    """Raised when a required env var or secret is missing or invalid at startup.

    Always raised at startup (in `load_config`), never mid-pipeline — by the time
    the pipeline starts running, every selected backend's prerequisites are known
    to be present.
    """


@dataclass(frozen=True)
class Config:
    classifier: ClassifierName
    summarizer: SummarizerName
    confidence_threshold: float

    aws_region: str
    raw_image_bucket: str | None
    results_table: str | None
    eventbridge_bus: str | None

    sagemaker_endpoint: str | None
    bedrock_classifier_model_id: str | None
    bedrock_summarizer_model_id: str | None
    anthropic_api_key: str | None
    anthropic_model: str


def load_config() -> Config:
    """Load and validate configuration from the environment.

    Raises `MissingSecretError` immediately if a backend is selected without its
    required secrets (e.g. `RETINA_SUMMARIZER=anthropic` without
    `ANTHROPIC_API_KEY`), or if a selection value is unknown.
    """
    classifier = _require_choice(
        "RETINA_CLASSIFIER",
        os.environ.get("RETINA_CLASSIFIER", "mock"),
        CLASSIFIER_OPTIONS,
    )
    summarizer = _require_choice(
        "RETINA_SUMMARIZER",
        os.environ.get("RETINA_SUMMARIZER", "mock"),
        SUMMARIZER_OPTIONS,
    )

    threshold_raw = os.environ.get("RETINA_CONFIDENCE_THRESHOLD", "0.7")
    try:
        confidence_threshold = float(threshold_raw)
    except ValueError as e:
        raise MissingSecretError(
            f"RETINA_CONFIDENCE_THRESHOLD={threshold_raw!r} is not a valid float"
        ) from e
    if not 0.0 <= confidence_threshold <= 1.0:
        raise MissingSecretError(
            f"RETINA_CONFIDENCE_THRESHOLD={confidence_threshold} must be in [0.0, 1.0]"
        )

    sagemaker_endpoint = _require_env("SAGEMAKER_ENDPOINT") if classifier == "sagemaker" else None
    bedrock_classifier_model_id = (
        _require_env("BEDROCK_CLASSIFIER_MODEL_ID") if classifier == "bedrock" else None
    )
    bedrock_summarizer_model_id = (
        _require_env("BEDROCK_SUMMARIZER_MODEL_ID") if summarizer == "bedrock" else None
    )
    anthropic_api_key = _require_env("ANTHROPIC_API_KEY") if summarizer == "anthropic" else None

    return Config(
        classifier=cast(ClassifierName, classifier),
        summarizer=cast(SummarizerName, summarizer),
        confidence_threshold=confidence_threshold,
        aws_region=os.environ.get("AWS_REGION", "us-west-2"),
        raw_image_bucket=os.environ.get("RETINA_RAW_BUCKET"),
        results_table=os.environ.get("RETINA_RESULTS_TABLE"),
        eventbridge_bus=os.environ.get("RETINA_EVENTBRIDGE_BUS"),
        sagemaker_endpoint=sagemaker_endpoint,
        bedrock_classifier_model_id=bedrock_classifier_model_id,
        bedrock_summarizer_model_id=bedrock_summarizer_model_id,
        anthropic_api_key=anthropic_api_key,
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7"),
    )


def apply_confidence_floor(classification: Classification, threshold: float) -> Referable:
    """Force `referable=review` when confidence is below the threshold.

    Safety rule: never auto-clear a low-confidence prediction. A human must look
    at any case where the model wasn't confident, regardless of what the model
    itself thought the answer was.
    """
    if classification.confidence < threshold:
        return "review"
    return classification.referable


def _require_choice(name: str, value: str, options: tuple[str, ...]) -> str:
    if value not in options:
        raise MissingSecretError(
            f"{name}={value!r} is not one of {options}. "
            f"Set {name} to one of these values or unset it for the default."
        )
    return value


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise MissingSecretError(f"{name} is required by the selected backend but is not set.")
    return value
