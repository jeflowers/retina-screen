"""Tests for the Summarizer Protocol and MockSummarizer.

The single most important property: ClinicalSummary's disclaimer validator
must NEVER trip — every code path through MockSummarizer produces a valid
summary. This is the implementation-side complement to the type-level
enforcement in dto.py.
"""

from datetime import UTC, datetime

from retina_screen.dto import (
    Classification,
    ClinicalSummary,
    ConditionProbability,
    EncounterMeta,
    Referable,
)
from retina_screen.ml.summarizer import MOCK_DISCLAIMER, MockSummarizer, Summarizer


def _meta(eye: str = "L") -> EncounterMeta:
    return EncounterMeta(
        encounter_id="enc_001",
        patient_pseudo_id="pt_001",
        site_id="site_A",
        eye=eye,  # type: ignore[arg-type]
        capture_device="topcon_nw400",
        captured_at=datetime(2026, 5, 28, 14, 30, tzinfo=UTC),
    )


def _classification(
    referable: Referable, primary_name: str = "diabetic_retinopathy_grade_2"
) -> Classification:
    return Classification(
        conditions=(
            ConditionProbability(name=primary_name, probability=0.78),
            ConditionProbability(name="normal", probability=0.22),
        ),
        referable=referable,
        model_name="mock-classifier",
        model_version="0.1.0",
        confidence=0.85,
    )


def test_mock_summarizer_implements_protocol() -> None:
    summarizer = MockSummarizer()
    assert isinstance(summarizer, Summarizer)
    assert summarizer.name == "mock-summarizer"


def test_mock_summarizer_always_includes_disclaimer() -> None:
    """Every branch (yes/no/review) must produce a non-empty disclaimer."""
    summarizer = MockSummarizer()
    meta = _meta()

    for referable in ("yes", "no", "review"):
        summary = summarizer.summarize(_classification(referable), meta)
        assert summary.disclaimer == MOCK_DISCLAIMER
        assert "screening aid" in summary.disclaimer


def test_mock_summarizer_is_deterministic() -> None:
    summarizer = MockSummarizer()
    meta = _meta()
    classification = _classification("yes")

    assert summarizer.summarize(classification, meta) == summarizer.summarize(classification, meta)


def test_referable_yes_yields_soon_urgency_and_refer_action() -> None:
    summary = MockSummarizer().summarize(_classification("yes"), _meta())
    assert summary.urgency == "soon"
    assert "refer" in summary.recommended_action.lower()


def test_referable_review_yields_grader_queue_action() -> None:
    summary = MockSummarizer().summarize(_classification("review"), _meta())
    assert summary.urgency == "soon"
    assert (
        "grader" in summary.recommended_action.lower()
        or "review" in summary.recommended_action.lower()
    )


def test_referable_no_yields_routine_urgency() -> None:
    summary = MockSummarizer().summarize(_classification("no", primary_name="normal"), _meta())
    assert summary.urgency == "routine"
    assert "routine" in summary.recommended_action.lower()


def test_summary_mentions_eye_side() -> None:
    """Findings text should reflect which eye was examined (left/right)."""
    s_left = MockSummarizer().summarize(_classification("yes"), _meta(eye="L"))
    s_right = MockSummarizer().summarize(_classification("yes"), _meta(eye="R"))
    assert "left" in s_left.findings.lower()
    assert "right" in s_right.findings.lower()


def test_summary_returns_clinical_summary_type() -> None:
    summary = MockSummarizer().summarize(_classification("yes"), _meta())
    assert isinstance(summary, ClinicalSummary)
