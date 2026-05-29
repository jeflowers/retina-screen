"""Classifier Protocol + implementations.

The Protocol is the architectural seam: the pipeline depends only on it, so
real backends (SageMaker, Bedrock) can be substituted by changing config alone.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from retina_screen.dto import (
    Classification,
    ConditionProbability,
    EncounterMeta,
    Referable,
)

__all__ = ["Classifier", "MockClassifier"]


_MOCK_CONDITIONS: tuple[str, ...] = (
    "diabetic_retinopathy_grade_2",
    "glaucoma_suspect",
    "amd_intermediate",
    "normal",
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

        # First three bytes seed primary probability, confidence, and which condition.
        primary_prob = 0.5 + (digest[0] / 255.0) * 0.49  # 0.50..0.99
        confidence = 0.3 + (digest[1] / 255.0) * 0.7  # 0.30..1.00
        primary_name = _MOCK_CONDITIONS[digest[2] % len(_MOCK_CONDITIONS)]

        # Secondary condition holds the remainder so probabilities sum to 1.0.
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
    """Map the mock's primary prediction to a referable decision.

    Mock-only logic: clearly-non-normal primaries become referable; the
    confidence-threshold safety rule (applied separately in the pipeline) may
    still flip the result to "review".
    """
    if primary_name == "normal":
        return "no"
    if primary_prob >= 0.7:
        return "yes"
    return "review"
