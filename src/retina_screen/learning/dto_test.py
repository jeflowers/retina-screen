"""Tests for the learning-loop DTOs: construction, forbid-extra, immutability, aliases."""

from datetime import UTC, datetime
from typing import get_args

import pytest
from pydantic import ValidationError

from retina_screen.learning.dto import (
    DrSeverity,
    EvalMetrics,
    GraderAdjudication,
    Intervention,
    InterventionType,
    LabeledEncounter,
    OutcomeRecord,
    ProgramMetrics,
)

_WHEN = datetime(2026, 1, 1, tzinfo=UTC)


def _adjudication() -> GraderAdjudication:
    return GraderAdjudication.model_validate(
        {
            "encounter_id": "synthENC0001",
            "grader_id": "synth_grader_1",
            "dr_severity": "pdr",
            "csme": True,
            "icd10_codes": ("E11.359",),
            "confirmed_at": _WHEN,
        }
    )


def test_grader_adjudication_happy_path() -> None:
    adj = _adjudication()
    assert adj.dr_severity == "pdr"
    assert adj.csme is True
    assert adj.icd10_codes == ("E11.359",)


def test_grader_adjudication_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        GraderAdjudication.model_validate(
            {
                "encounter_id": "e",
                "grader_id": "g",
                "dr_severity": "none",
                "csme": False,
                "confirmed_at": _WHEN,
                "surprise": "x",
            }
        )


def test_grader_adjudication_rejects_invalid_severity() -> None:
    with pytest.raises(ValidationError):
        GraderAdjudication.model_validate(
            {
                "encounter_id": "e",
                "grader_id": "g",
                "dr_severity": "terminal",
                "csme": False,
                "confirmed_at": _WHEN,
            }
        )


def test_intervention_happy_path_and_optional_codes() -> None:
    iv = Intervention.model_validate(
        {
            "encounter_id": "e",
            "type": "prp",
            "performed_at": _WHEN,
            "site_id": "synth_site_a",
        }
    )
    assert iv.type == "prp"
    assert iv.cpt_code is None


def test_outcome_record_rejects_negative_oct() -> None:
    with pytest.raises(ValidationError):
        OutcomeRecord.model_validate(
            {
                "encounter_id": "e",
                "follow_up_at": _WHEN,
                "va_lines_change": 2,
                "oct_cmt_um": -5.0,
                "oct_improved": True,
                "stabilized": True,
            }
        )


def test_outcome_record_allows_negative_va_change() -> None:
    rec = OutcomeRecord.model_validate(
        {
            "encounter_id": "e",
            "follow_up_at": _WHEN,
            "va_lines_change": -2,
            "oct_improved": False,
            "stabilized": False,
        }
    )
    assert rec.va_lines_change == -2
    assert rec.oct_cmt_um is None


def test_eval_metrics_rejects_out_of_range_rate() -> None:
    with pytest.raises(ValidationError):
        EvalMetrics.model_validate(
            {
                "n": 1,
                "true_positive": 1,
                "false_positive": 0,
                "true_negative": 0,
                "false_negative": 0,
                "sensitivity": 1.5,
                "specificity": 1.0,
                "accuracy": 1.0,
            }
        )


def test_program_metrics_happy_path() -> None:
    pm = ProgramMetrics.model_validate(
        {
            "screened": 10,
            "dr_prevalence": 0.4,
            "pdr_rate": 0.1,
            "csme_rate": 0.2,
            "prp_rate": 0.1,
            "anti_vegf_rate": 0.2,
            "vitrectomy_rate": 0.0,
        }
    )
    assert pm.screened == 10


def test_labeled_encounter_is_immutable() -> None:
    le = LabeledEncounter.model_validate(
        {"result": _screening_result(), "adjudication": _adjudication()}
    )
    with pytest.raises(ValidationError):
        le.adjudication = _adjudication()


def test_type_aliases_have_expected_values() -> None:
    assert set(get_args(DrSeverity)) == {
        "none",
        "mild_npdr",
        "moderate_npdr",
        "severe_npdr",
        "pdr",
    }
    assert set(get_args(InterventionType)) == {"prp", "anti_vegf", "vitrectomy", "none"}


def _screening_result() -> object:
    # Local import keeps the core-DTO construction next to its only user here.
    from retina_screen.dto import ScreeningResult

    return ScreeningResult.model_validate(
        {
            "meta": {
                "encounter_id": "synthENC0001",
                "patient_pseudo_id": "synthPT0001",
                "site_id": "synth_site_a",
                "eye": "L",
                "capture_device": "synthetic_camera",
                "captured_at": _WHEN,
            },
            "classification": {
                "conditions": ({"name": "normal", "probability": 1.0},),
                "referable": "no",
                "model_name": "mock-classifier",
                "model_version": "0.1.0",
                "confidence": 0.9,
            },
            "summary": {
                "findings": "synthetic findings",
                "recommended_action": "synthetic action",
                "urgency": "routine",
                "disclaimer": "This is a screening aid, not a diagnosis.",
            },
            "final_referable": "no",
            "created_at": _WHEN,
        }
    )
