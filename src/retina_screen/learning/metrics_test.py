"""Tests for the eval / QI / active-learning metric computations."""

from datetime import UTC, datetime

from retina_screen.dto import Referable, ScreeningResult
from retina_screen.learning.dto import (
    DrSeverity,
    GraderAdjudication,
    Intervention,
    InterventionType,
    LabeledEncounter,
)
from retina_screen.learning.metrics import (
    active_learning_queue,
    evaluate,
    program_metrics,
)

_WHEN = datetime(2026, 1, 1, tzinfo=UTC)


def _result(final_referable: Referable, confidence: float = 0.9) -> ScreeningResult:
    return ScreeningResult.model_validate(
        {
            "meta": {
                "encounter_id": "e",
                "patient_pseudo_id": "p",
                "site_id": "synth_site_a",
                "eye": "L",
                "capture_device": "synthetic_camera",
                "captured_at": _WHEN,
            },
            "classification": {
                "conditions": ({"name": "x", "probability": 1.0},),
                "referable": "yes",
                "model_name": "mock-classifier",
                "model_version": "0.1.0",
                "confidence": confidence,
            },
            "summary": {
                "findings": "f",
                "recommended_action": "a",
                "urgency": "routine",
                "disclaimer": "This is a screening aid, not a diagnosis.",
            },
            "final_referable": final_referable,
            "created_at": _WHEN,
        }
    )


def _labeled(
    final_referable: Referable,
    severity: DrSeverity,
    *,
    csme: bool = False,
    interventions: tuple[Intervention, ...] = (),
) -> LabeledEncounter:
    adjudication = GraderAdjudication.model_validate(
        {
            "encounter_id": "e",
            "grader_id": "g",
            "dr_severity": severity,
            "csme": csme,
            "confirmed_at": _WHEN,
        }
    )
    return LabeledEncounter.model_validate(
        {
            "result": _result(final_referable),
            "adjudication": adjudication,
            "interventions": interventions,
        }
    )


def _intervention(kind: InterventionType) -> Intervention:
    return Intervention.model_validate(
        {
            "encounter_id": "e",
            "type": kind,
            "performed_at": _WHEN,
            "site_id": "synth_site_a",
        }
    )


def test_evaluate_confusion_matrix_and_rates() -> None:
    labeled = [
        _labeled("yes", "pdr"),  # truth+, pred+  -> TP
        _labeled("no", "severe_npdr"),  # truth+, pred-  -> FN
        _labeled("yes", "none"),  # truth-, pred+  -> FP
        _labeled("no", "none"),  # truth-, pred-  -> TN
    ]
    m = evaluate(labeled)
    assert (m.true_positive, m.false_negative, m.false_positive, m.true_negative) == (
        1,
        1,
        1,
        1,
    )
    assert m.n == 4
    assert m.sensitivity == 0.5
    assert m.specificity == 0.5
    assert m.accuracy == 0.5


def test_evaluate_csme_counts_as_referable_truth() -> None:
    # Mild NPDR is not referable on its own, but CSME makes it truth-positive.
    m = evaluate([_labeled("yes", "mild_npdr", csme=True)])
    assert m.true_positive == 1
    assert m.sensitivity == 1.0


def test_evaluate_review_counts_as_predicted_positive() -> None:
    m = evaluate([_labeled("review", "pdr")])
    assert m.true_positive == 1


def test_evaluate_empty_is_zeroed_not_crash() -> None:
    m = evaluate([])
    assert m.n == 0
    assert m.sensitivity == 0.0
    assert m.accuracy == 0.0


def test_program_metrics_rates() -> None:
    labeled = [
        _labeled("no", "none"),
        _labeled("no", "mild_npdr"),
        _labeled("yes", "pdr", interventions=(_intervention("prp"),)),
        _labeled("yes", "moderate_npdr", csme=True, interventions=(_intervention("anti_vegf"),)),
    ]
    pm = program_metrics(labeled)
    assert pm.screened == 4
    assert pm.dr_prevalence == 0.75
    assert pm.pdr_rate == 0.25
    assert pm.csme_rate == 0.25
    assert pm.prp_rate == 0.25
    assert pm.anti_vegf_rate == 0.25
    assert pm.vitrectomy_rate == 0.0


def test_active_learning_queue_selects_low_confidence() -> None:
    low = _result("review", confidence=0.5)
    high = _result("no", confidence=0.95)
    queue = active_learning_queue([low, high], threshold=0.7)
    assert queue == (low,)
