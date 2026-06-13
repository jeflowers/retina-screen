"""Eval / QI / active-learning computations over the learning-loop contracts.

Pure functions, no I/O and no network: `evaluate` scores model predictions
against grader ground truth, `program_metrics` aggregates the QI feed, and
`active_learning_queue` surfaces the low-confidence cases worth labeling next
(the same confidence-floor idea the pipeline already enforces).
"""

from collections.abc import Sequence

from retina_screen.dto import ScreeningResult
from retina_screen.learning.dto import (
    EvalMetrics,
    GraderAdjudication,
    InterventionType,
    LabeledEncounter,
    ProgramMetrics,
)

__all__ = ["active_learning_queue", "evaluate", "program_metrics"]


# Clinically referable disease (moderate NPDR or worse, or any CSME).
_REFERABLE_SEVERITIES = frozenset({"moderate_npdr", "severe_npdr", "pdr"})


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _truth_positive(adjudication: GraderAdjudication) -> bool:
    """Ground-truth referable: grader severity is referable, or CSME present."""
    return adjudication.dr_severity in _REFERABLE_SEVERITIES or adjudication.csme


def _pred_positive(result: ScreeningResult) -> bool:
    """Model flagged the case: any final disposition other than a clean 'no'."""
    return result.final_referable != "no"


def evaluate(labeled: Sequence[LabeledEncounter]) -> EvalMetrics:
    """Score model predictions against grader ground truth (referable as binary)."""
    tp = fp = tn = fn = 0
    for item in labeled:
        truth = _truth_positive(item.adjudication)
        pred = _pred_positive(item.result)
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif pred:
            fp += 1
        else:
            tn += 1

    return EvalMetrics(
        n=len(labeled),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        sensitivity=_safe_div(tp, tp + fn),
        specificity=_safe_div(tn, tn + fp),
        accuracy=_safe_div(tp + tn, len(labeled)),
    )


def program_metrics(labeled: Sequence[LabeledEncounter]) -> ProgramMetrics:
    """Aggregate program / QI metrics over a labeled set."""
    n = len(labeled)
    any_dr = sum(1 for item in labeled if item.adjudication.dr_severity != "none")
    pdr = sum(1 for item in labeled if item.adjudication.dr_severity == "pdr")
    csme = sum(1 for item in labeled if item.adjudication.csme)

    def _intervention_rate(kind: InterventionType) -> float:
        count = sum(1 for item in labeled if any(iv.type == kind for iv in item.interventions))
        return _safe_div(count, n)

    return ProgramMetrics(
        screened=n,
        dr_prevalence=_safe_div(any_dr, n),
        pdr_rate=_safe_div(pdr, n),
        csme_rate=_safe_div(csme, n),
        prp_rate=_intervention_rate("prp"),
        anti_vegf_rate=_intervention_rate("anti_vegf"),
        vitrectomy_rate=_intervention_rate("vitrectomy"),
    )


def active_learning_queue(
    results: Sequence[ScreeningResult], threshold: float
) -> tuple[ScreeningResult, ...]:
    """Cases below the confidence threshold — the highest-value labels to collect."""
    return tuple(r for r in results if r.classification.confidence < threshold)
