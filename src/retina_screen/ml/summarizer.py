"""Summarizer Protocol + implementations.

Mock summarizer ALWAYS includes the screening-aid disclaimer — real backends
will too, but the type system (ClinicalSummary's model_validator) is the
ultimate enforcement, not any individual implementation's prompt.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from retina_screen.dto import Classification, ClinicalSummary, EncounterMeta, Urgency

__all__ = ["MOCK_DISCLAIMER", "MockSummarizer", "Summarizer"]


MOCK_DISCLAIMER = (
    "This is an automated screening aid, not a clinical diagnosis. "
    "A qualified specialist must review any referral."
)


@runtime_checkable
class Summarizer(Protocol):
    """Classification-to-summary protocol.

    Implementations: MockSummarizer (default), BedrockSummarizer, AnthropicSummarizer.
    `name` is a read-only property so frozen-dataclass implementations conform.
    """

    @property
    def name(self) -> str: ...

    def summarize(self, classification: Classification, meta: EncounterMeta) -> ClinicalSummary: ...


@dataclass(frozen=True)
class MockSummarizer:
    """Deterministic templated summarizer — no network."""

    name: str = "mock-summarizer"

    def summarize(self, classification: Classification, meta: EncounterMeta) -> ClinicalSummary:
        primary = classification.conditions[0]
        findings, recommended, urgency = _template_for(
            classification.referable,
            primary.name,
            primary.probability,
            classification.confidence,
            meta.eye,
        )

        return ClinicalSummary(
            findings=findings,
            recommended_action=recommended,
            urgency=urgency,
            disclaimer=MOCK_DISCLAIMER,
        )


def _template_for(
    referable: str,
    primary_name: str,
    primary_prob: float,
    confidence: float,
    eye: str,
) -> tuple[str, str, Urgency]:
    """Pick template strings + urgency for the (referable, primary) combination."""
    eye_label = "left" if eye == "L" else "right"

    if referable == "yes":
        return (
            f"Model flagged probable referable disease in the {eye_label} eye: "
            f"{primary_name} (p={primary_prob:.2f}, confidence={confidence:.2f}).",
            "Refer to ophthalmology within 4 weeks.",
            "soon",
        )
    if referable == "review":
        return (
            f"Borderline finding in the {eye_label} eye: {primary_name} "
            f"(p={primary_prob:.2f}, confidence={confidence:.2f}). "
            f"Below the configured confidence threshold for an auto-decision.",
            "Route to grader queue for manual review before disposition.",
            "soon",
        )
    return (
        f"No referable disease detected in the {eye_label} eye "
        f"({primary_name}, p={primary_prob:.2f}, confidence={confidence:.2f}).",
        "Continue routine screening per site protocol.",
        "routine",
    )
