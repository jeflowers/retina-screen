"""Tests for the screening DTOs.

Disclaimer enforcement is a load-bearing safety property — verify it cannot be
bypassed by direct construction, model_validate, or copy-with-empty-disclaimer.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from retina_screen.dto import (
    Classification,
    ClinicalSummary,
    EncounterMeta,
    ReferralNeeded,
    ScreeningCompleted,
    ScreeningResult,
)


def _meta() -> EncounterMeta:
    return EncounterMeta.model_validate(
        {
            "encounter_id": "enc_001",
            "patient_pseudo_id": "pt_001",
            "site_id": "site_A",
            "eye": "L",
            "capture_device": "topcon_nw400",
            "captured_at": "2026-05-28T14:30:00Z",
        }
    )


def _classification(confidence: float = 0.9) -> Classification:
    return Classification.model_validate(
        {
            "conditions": [{"name": "diabetic_retinopathy_grade_2", "probability": 0.71}],
            "referable": "yes",
            "model_name": "mock-classifier",
            "model_version": "0.1",
            "confidence": confidence,
        }
    )


def _summary(disclaimer: str = "This is a screening aid, not a diagnosis.") -> ClinicalSummary:
    return ClinicalSummary.model_validate(
        {
            "findings": "Suspected referable diabetic retinopathy.",
            "recommended_action": "Refer for ophthalmologist review.",
            "urgency": "soon",
            "disclaimer": disclaimer,
        }
    )


# ─── EncounterMeta ───────────────────────────────────────────────────────────


def test_encounter_meta_happy_path() -> None:
    meta = _meta()
    assert meta.eye == "L"
    assert meta.captured_at.tzinfo is not None


def test_encounter_meta_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EncounterMeta.model_validate(
            {
                "encounter_id": "enc_001",
                "patient_pseudo_id": "pt_001",
                "site_id": "site_A",
                "eye": "L",
                "capture_device": "topcon_nw400",
                "captured_at": "2026-05-28T14:30:00Z",
                "smuggled_field": "should fail",
            }
        )


def test_encounter_meta_rejects_invalid_eye() -> None:
    with pytest.raises(ValidationError):
        EncounterMeta.model_validate(
            {
                "encounter_id": "enc_001",
                "patient_pseudo_id": "pt_001",
                "site_id": "site_A",
                "eye": "X",
                "capture_device": "topcon_nw400",
                "captured_at": "2026-05-28T14:30:00Z",
            }
        )


def test_encounter_meta_rejects_empty_strings() -> None:
    with pytest.raises(ValidationError):
        EncounterMeta.model_validate(
            {
                "encounter_id": "",
                "patient_pseudo_id": "pt_001",
                "site_id": "site_A",
                "eye": "L",
                "capture_device": "topcon_nw400",
                "captured_at": "2026-05-28T14:30:00Z",
            }
        )


# ─── Classification ──────────────────────────────────────────────────────────


def test_classification_happy_path() -> None:
    c = _classification()
    assert c.referable == "yes"
    assert c.confidence == 0.9


def test_classification_rejects_out_of_range_probability() -> None:
    with pytest.raises(ValidationError):
        Classification.model_validate(
            {
                "conditions": [{"name": "x", "probability": 1.5}],
                "referable": "yes",
                "model_name": "m",
                "model_version": "1",
                "confidence": 0.9,
            }
        )


def test_classification_rejects_invalid_referable() -> None:
    with pytest.raises(ValidationError):
        Classification.model_validate(
            {
                "conditions": [{"name": "x", "probability": 0.1}],
                "referable": "maybe",
                "model_name": "m",
                "model_version": "1",
                "confidence": 0.9,
            }
        )


# ─── ClinicalSummary — disclaimer enforcement is load-bearing ────────────────


def test_clinical_summary_with_disclaimer_succeeds() -> None:
    s = _summary()
    assert "screening aid" in s.disclaimer


def test_clinical_summary_with_empty_disclaimer_raises() -> None:
    with pytest.raises(ValidationError):
        _summary(disclaimer="")


def test_clinical_summary_with_whitespace_only_disclaimer_raises() -> None:
    with pytest.raises(ValidationError):
        _summary(disclaimer="   \n\t  ")


def test_clinical_summary_is_immutable() -> None:
    s = _summary()
    with pytest.raises(ValidationError):
        s.disclaimer = ""


# ─── ScreeningResult ─────────────────────────────────────────────────────────


def test_screening_result_composition() -> None:
    result = ScreeningResult.model_validate(
        {
            "meta": _meta().model_dump(),
            "classification": _classification().model_dump(),
            "summary": _summary().model_dump(),
            "final_referable": "yes",
            "created_at": datetime.now(UTC),
        }
    )
    assert result.final_referable == "yes"
    assert result.meta.encounter_id == "enc_001"


# ─── Events ──────────────────────────────────────────────────────────────────


def test_referral_needed_event() -> None:
    e = ReferralNeeded.model_validate(
        {
            "encounter_id": "enc_001",
            "site_id": "site_A",
            "occurred_at": datetime.now(UTC),
            "urgency": "soon",
            "model_name": "mock-classifier",
        }
    )
    assert e.event_type == "ReferralNeeded"


def test_screening_completed_event() -> None:
    e = ScreeningCompleted.model_validate(
        {
            "encounter_id": "enc_001",
            "site_id": "site_A",
            "occurred_at": datetime.now(UTC),
        }
    )
    assert e.event_type == "ScreeningCompleted"
