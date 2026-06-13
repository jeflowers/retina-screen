"""Tests for the synthetic data generator and the end-to-end demo."""

from retina_screen.learning.demo import run_demo
from retina_screen.learning.metrics import evaluate, program_metrics
from retina_screen.learning.synthetic import make_synthetic_labeled_encounters


def test_generator_is_deterministic_for_same_n() -> None:
    a = make_synthetic_labeled_encounters(25)
    b = make_synthetic_labeled_encounters(25)
    assert a == b


def test_generator_count_matches_n() -> None:
    assert len(make_synthetic_labeled_encounters(13)) == 13


def test_all_identifiers_are_obviously_synthetic() -> None:
    for item in make_synthetic_labeled_encounters(20):
        assert item.result.meta.encounter_id.startswith("synthENC")
        assert item.result.meta.patient_pseudo_id.startswith("synthPT")
        assert item.result.meta.site_id.startswith("synth_site_")
        assert item.adjudication.grader_id.startswith("synth_")


def test_referable_truth_cases_get_intervention_and_outcome() -> None:
    for item in make_synthetic_labeled_encounters(60):
        needs_treatment = (
            item.adjudication.dr_severity in ("severe_npdr", "pdr") or item.adjudication.csme
        )
        if needs_treatment:
            assert item.interventions
            assert item.outcome is not None
        else:
            assert item.interventions == ()
            assert item.outcome is None


def test_metrics_run_over_synthetic_data() -> None:
    labeled = make_synthetic_labeled_encounters(100)
    pm = program_metrics(labeled)
    ev = evaluate(labeled)
    assert pm.screened == 100
    assert 0.0 <= pm.dr_prevalence <= 1.0
    assert ev.n == 100
    assert ev.true_positive + ev.false_positive + ev.true_negative + ev.false_negative == 100


def test_run_demo_returns_both_metric_dtos() -> None:
    program, evaluation = run_demo(30)
    assert program.screened == 30
    assert evaluation.n == 30
