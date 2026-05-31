"""Classifier Protocol + implementations.

The Protocol is the architectural seam: the pipeline depends only on it, so
real backends (SageMaker, Bedrock) can be substituted by changing config alone.

DEFENSIVE PARSING (acceptance #6): every real backend wraps its network call
and response parsing in a broad except. On ANY failure — JSON decode error,
missing field, unexpected type, validation error, boto3 ClientError — the
implementation returns a low-confidence Classification with referable="review"
and logs a warning. The pipeline keeps running; a human reviews the result.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog
from botocore.exceptions import ClientError
from pydantic import ValidationError

from retina_screen.dto import (
    Classification,
    ConditionProbability,
    EncounterMeta,
    Referable,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_sagemaker_runtime import SageMakerRuntimeClient

__all__ = [
    "BedrockImageClassifier",
    "Classifier",
    "MockClassifier",
    "SageMakerClassifier",
]


_MOCK_CONDITIONS: tuple[str, ...] = (
    "diabetic_retinopathy_grade_2",
    "glaucoma_suspect",
    "amd_intermediate",
    "normal",
)

# Bedrock body version for Anthropic-format models.
_BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Errors that a real backend must degrade gracefully on, not crash from.
_PARSE_FAILURE_ERRORS = (
    json.JSONDecodeError,
    KeyError,
    TypeError,
    ValueError,
    ValidationError,
)


@runtime_checkable
class Classifier(Protocol):
    """Image-to-classification protocol.

    Implementations: MockClassifier (default), SageMakerClassifier, BedrockImageClassifier.
    `name` is a read-only property so frozen-dataclass implementations conform.
    """

    @property
    def name(self) -> str: ...

    def classify(self, image_bytes: bytes, meta: EncounterMeta) -> Classification: ...


# ─── Mock ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MockClassifier:
    """Deterministic mock classifier — no network, repeatable for tests.

    Output is derived from sha256(image_bytes) so identical inputs always produce
    identical Classifications. Useful both as a default backend and as the
    in-test substitute for the real SageMaker/Bedrock clients.
    """

    name: str = "mock-classifier"
    version: str = field(default="0.1.0")

    def classify(self, image_bytes: bytes, meta: EncounterMeta) -> Classification:
        del meta  # part of the Protocol but unused by the mock
        digest = hashlib.sha256(image_bytes).digest()

        primary_prob = 0.5 + (digest[0] / 255.0) * 0.49  # 0.50..0.99
        confidence = 0.3 + (digest[1] / 255.0) * 0.7  # 0.30..1.00
        primary_name = _MOCK_CONDITIONS[digest[2] % len(_MOCK_CONDITIONS)]

        secondary_name = "other" if primary_name == "normal" else "normal"
        conditions = (
            ConditionProbability(name=primary_name, probability=round(primary_prob, 4)),
            ConditionProbability(name=secondary_name, probability=round(1.0 - primary_prob, 4)),
        )

        referable: Referable = _derive_referable(primary_name, primary_prob)

        return Classification(
            conditions=conditions,
            referable=referable,
            model_name=self.name,
            model_version=self.version,
            confidence=round(confidence, 4),
        )


def _derive_referable(primary_name: str, primary_prob: float) -> Referable:
    """Map the mock's primary prediction to a referable decision."""
    if primary_name == "normal":
        return "no"
    if primary_prob >= 0.7:
        return "yes"
    return "review"


# ─── SageMaker ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SageMakerClassifier:
    """SageMaker runtime classifier (client injected).

    Expects the endpoint to return JSON of the shape:
        {
          "conditions": [{"name": str, "probability": float}, ...],
          "referable": "yes" | "no" | "review",
          "confidence": float in [0, 1]
        }

    Adapt `_parse_classification` for your endpoint's response shape if
    different. Any parse failure or AWS error degrades to a low-confidence
    fallback (referable="review") and a logged warning — never crashes.
    """

    client: SageMakerRuntimeClient
    endpoint_name: str
    name: str = field(default="sagemaker-classifier")

    def classify(self, image_bytes: bytes, meta: EncounterMeta) -> Classification:
        del meta  # could be passed via CustomAttributes; left out for the demo
        log = structlog.get_logger().bind(backend=self.name, endpoint=self.endpoint_name)
        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="image/jpeg",
                Accept="application/json",
                Body=image_bytes,
            )
            body_bytes = response["Body"].read()
            data: Any = json.loads(body_bytes)
            return _parse_classification(
                data, model_name=self.name, model_version=self.endpoint_name
            )
        except ClientError as e:
            log.warning("sagemaker_invoke_failed", error=str(e))
            return _fallback_classification(self.name, self.endpoint_name)
        except _PARSE_FAILURE_ERRORS as e:
            log.warning("sagemaker_response_parse_failed", error=str(e))
            return _fallback_classification(self.name, self.endpoint_name)


# ─── Bedrock (image) ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BedrockImageClassifier:
    """Bedrock multimodal classifier (Claude-on-Bedrock format).

    Adapt `_build_bedrock_image_body` for other Bedrock models (Llama,
    Titan, etc.) — the body and response shape are model-specific.

    Same defensive-parse contract as SageMakerClassifier.
    """

    client: BedrockRuntimeClient
    model_id: str
    name: str = field(default="bedrock-image-classifier")

    def classify(self, image_bytes: bytes, meta: EncounterMeta) -> Classification:
        del meta
        log = structlog.get_logger().bind(backend=self.name, model_id=self.model_id)
        try:
            body = _build_bedrock_image_body(image_bytes)
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            raw = response["body"].read()
            envelope: Any = json.loads(raw)
            # Claude-on-Bedrock returns content as [{type: text, text: "..."}].
            text = envelope["content"][0]["text"]
            data: Any = json.loads(text)
            return _parse_classification(data, model_name=self.name, model_version=self.model_id)
        except ClientError as e:
            log.warning("bedrock_invoke_failed", error=str(e))
            return _fallback_classification(self.name, self.model_id)
        except _PARSE_FAILURE_ERRORS as e:
            log.warning("bedrock_response_parse_failed", error=str(e))
            return _fallback_classification(self.name, self.model_id)


def _build_bedrock_image_body(image_bytes: bytes) -> bytes:
    """Claude-on-Bedrock multimodal request body for classification."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "anthropic_version": _BEDROCK_ANTHROPIC_VERSION,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Classify this fundus image. Output ONLY JSON with this "
                            'schema: {"conditions": [{"name": string, "probability": '
                            'number 0..1}], "referable": "yes"|"no"|"review", '
                            '"confidence": number 0..1}. Use only image evidence. '
                            "Never imply a definitive diagnosis."
                        ),
                    },
                ],
            }
        ],
    }
    return json.dumps(payload).encode("utf-8")


# ─── Shared parsing helpers ──────────────────────────────────────────────────


def _parse_classification(data: Any, *, model_name: str, model_version: str) -> Classification:
    """Build a Classification from a parsed JSON payload.

    Raises any of `_PARSE_FAILURE_ERRORS` on bad shape — the caller catches
    these and falls back. Centralizing the parse means every backend that
    promises this shape gets the same validation.
    """
    return Classification(
        conditions=tuple(
            ConditionProbability(name=str(c["name"]), probability=float(c["probability"]))
            for c in data["conditions"]
        ),
        referable=data["referable"],
        model_name=model_name,
        model_version=model_version,
        confidence=float(data["confidence"]),
    )


def _fallback_classification(model_name: str, model_version: str) -> Classification:
    """Low-confidence fallback for any backend failure.

    referable="review" + confidence=0.0 ensures both the safety floor and
    the routing rule (review → ReferralNeeded) send this to a human.
    """
    return Classification(
        conditions=(ConditionProbability(name="parse_failure_low_confidence", probability=1.0),),
        referable="review",
        model_name=model_name,
        model_version=model_version,
        confidence=0.0,
    )
