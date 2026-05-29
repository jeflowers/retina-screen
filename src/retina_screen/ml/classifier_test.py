"""Tests for the Classifier Protocol and MockClassifier.

Determinism is the load-bearing property — the same image bytes must always
produce the same Classification, both as a sanity check and as the substrate
that makes test fixtures stable.
"""

from datetime import UTC, datetime

from retina_screen.dto import EncounterMeta
from retina_screen.ml.classifier import Classifier, MockClassifier


def _meta(eye: str = "L") -> EncounterMeta:
    return EncounterMeta(
        encounter_id="enc_001",
        patient_pseudo_id="pt_001",
        site_id="site_A",
        eye=eye,  # type: ignore[arg-type]
        capture_device="topcon_nw400",
        captured_at=datetime(2026, 5, 28, 14, 30, tzinfo=UTC),
    )


def test_mock_classifier_implements_protocol() -> None:
    """MockClassifier satisfies the Classifier Protocol at runtime."""
    classifier = MockClassifier()
    assert isinstance(classifier, Classifier)
    assert classifier.name == "mock-classifier"


def test_mock_classifier_is_deterministic_for_same_bytes() -> None:
    classifier = MockClassifier()
    bytes_a = b"fundus-image-bytes-aaa"
    meta = _meta()

    result1 = classifier.classify(bytes_a, meta)
    result2 = classifier.classify(bytes_a, meta)

    assert result1 == result2


def test_mock_classifier_differs_for_different_bytes() -> None:
    classifier = MockClassifier()
    meta = _meta()

    result_a = classifier.classify(b"image-aaa", meta)
    result_b = classifier.classify(b"image-zzz-totally-different", meta)

    assert result_a != result_b


def test_mock_classifier_probabilities_in_range() -> None:
    classifier = MockClassifier()
    meta = _meta()

    # Sweep a range of inputs and assert probability invariants hold for all.
    for seed in range(50):
        result = classifier.classify(f"image-{seed}".encode(), meta)
        for condition in result.conditions:
            assert 0.0 <= condition.probability <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        total = sum(c.probability for c in result.conditions)
        assert abs(total - 1.0) < 0.001  # primary + secondary sum to 1


def test_mock_classifier_reports_model_identity() -> None:
    result = MockClassifier().classify(b"x", _meta())
    assert result.model_name == "mock-classifier"
    assert result.model_version == "0.1.0"


def test_mock_classifier_normal_primary_is_not_referable() -> None:
    """When the mock lands on 'normal', referable should be 'no'."""
    classifier = MockClassifier()
    meta = _meta()

    # Find an input whose digest[2] % 4 == 3 (= "normal" — last in _MOCK_CONDITIONS).
    found_normal = False
    for seed in range(200):
        result = classifier.classify(f"hunting-{seed}".encode(), meta)
        if result.conditions[0].name == "normal":
            assert result.referable == "no"
            found_normal = True
            break
    assert found_normal, "expected at least one normal primary in 200 samples"
