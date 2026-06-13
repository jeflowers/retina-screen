"""Tests for the production learning-loop DTOs.

Covers the hardening that distinguishes these from a loose sketch: ICD-10
pattern validation, tz-aware datetime enforcement, forbid-extra, immutability,
and the PHI governance tags.
"""

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
_NAIVE = datetime(2026, 1, 1)  # intentionally naive (no tzinfo), to prove rejection


def _adjudication() -> GraderAdjudication:
    return GraderAdjudication.model_validate(
        {
            "encounter_id": "enc_0001",
            "grader_id": "grader_1",
            "dr_severity": "pdr",
            "csme": True,
            "icd10_codes": ("E11.359",),
            "confirmed_at": _WHEN,
        }
    )


def test_grader_adjudication_happy_path() -> None:
    adj = _adjudication()
    assert adj.dr_severity == "pdr"
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


def test_grader_adjudication_rejects_malformed_icd10() -> None:
    with pytest.raises(ValidationError):
        GraderAdjudication.model_validate(
            {
                "encounter_id": "e",
                "grader_id": "g",
                "dr_severity": "pdr",
                "csme": False,
                "icd10_codes": ("NOT-A-CODE",),
                "confirmed_at": _WHEN,
            }
        )


def test_grader_adjudication_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError):
        GraderAdjudication.model_validate(
            {
                "encounter_id": "e",
                "grader_id": "g",
                "dr_severity": "none",
                "csme": False,
                "confirmed_at": _NAIVE,
            }
        )


def test_intervention_optional_codes_default_none() -> None:
    iv = Intervention.model_validate(
        {
            "encounter_id": "e",
            "type": "prp",
            "performed_at": _WHEN,
            "site_id": "site_a",
        }
    )
    assert iv.cpt_code is None


def test_outcome_record_rejects_negative_oct() -> None:
    with pytest.raises(ValidationError):
        OutcomeRecord.model_validate(
            {
                "encounter_id": "e",
                "follow_up_at": _WHEN,
                "va_lines_change": 1,
                "oct_cmt_um": -1.0,
                "oct_improved": True,
                "stabilized": True,
            }
        )


def test_outcome_record_allows_negative_va_change() -> None:
    rec = OutcomeRecord.model_validate(
        {
            "encounter_id": "e",
            "follow_up_at": _WHEN,
            "va_lines_change": -3,
            "oct_improved": False,
            "stabilized": False,
        }
    )
    assert rec.va_lines_change == -3


def test_eval_metrics_rejects_out_of_range_rate() -> None:
    with pytest.raises(ValidationError):
        EvalMetrics.model_validate(
            {
                "n": 1,
                "true_positive": 1,
                "false_positive": 0,
                "true_negative": 0,
                "false_negative": 0,
                "sensitivity": 2.0,
                "specificity": 1.0,
                "accuracy": 1.0,
            }
        )


def test_program_metrics_happy_path() -> None:
    pm = ProgramMetrics.model_validate(
        {
            "screened": 5,
            "dr_prevalence": 0.4,
            "pdr_rate": 0.2,
            "csme_rate": 0.0,
            "prp_rate": 0.2,
            "anti_vegf_rate": 0.0,
            "vitrectomy_rate": 0.0,
        }
    )
    assert pm.screened == 5


def test_labeled_encounter_is_immutable() -> None:
    le = LabeledEncounter.model_validate(
        {"result": _screening_result(), "adjudication": _adjudication()}
    )
    with pytest.raises(ValidationError):
        le.outcome = None


def test_phi_fields_are_tagged_for_governance() -> None:
    # Clinical fields must advertise phi=True so redaction/governance can find them.
    assert GraderAdjudication.model_fields["dr_severity"].json_schema_extra == {"phi": True}
    assert OutcomeRecord.model_fields["va_lines_change"].json_schema_extra == {"phi": True}
    # Staff identifier is NOT PHI.
    assert GraderAdjudication.model_fields["grader_id"].json_schema_extra is None


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
    from retina_screen.dto import ScreeningResult

    return ScreeningResult.model_validate(
        {
            "meta": {
                "encounter_id": "enc_0001",
                "patient_pseudo_id": "pt_0001",
                "site_id": "site_a",
                "eye": "L",
                "capture_device": "topcon_nw400",
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
                "findings": "f",
                "recommended_action": "a",
                "urgency": "routine",
                "disclaimer": "This is a screening aid, not a diagnosis.",
            },
            "final_referable": "no",
            "created_at": _WHEN,
        }
    )
