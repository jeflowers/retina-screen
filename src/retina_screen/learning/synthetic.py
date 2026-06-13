"""Deterministic SYNTHETIC data generator for the learning-loop demo — no real PHI.

Produces fabricated `LabeledEncounter`s by running the default mock ML backends
over fake image bytes, then attaching a hash-derived grader label, interventions,
and outcome. Deterministic (seeded by index) so tests and the demo are
repeatable. Every identifier is obviously fabricated (``synth*`` prefixes).
"""

import hashlib
from datetime import UTC, datetime, timedelta

from retina_screen.config import apply_confidence_floor
from retina_screen.dto import EncounterMeta, ScreeningResult
from retina_screen.learning.dto import (
    DrSeverity,
    GraderAdjudication,
    Intervention,
    InterventionType,
    LabeledEncounter,
    OutcomeRecord,
)
from retina_screen.ml.classifier import MockClassifier
from retina_screen.ml.summarizer import MockSummarizer

__all__ = ["make_synthetic_labeled_encounters"]


_SITES = ("synth_site_a", "synth_site_b", "synth_site_c")
_REFERABLE_SEVERITIES: tuple[DrSeverity, ...] = ("moderate_npdr", "severe_npdr", "pdr")
_NONREFERABLE_SEVERITIES: tuple[DrSeverity, ...] = ("none", "mild_npdr")
_INTERVENTIONS: tuple[InterventionType, ...] = ("prp", "anti_vegf", "vitrectomy")
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_CONFIDENCE_THRESHOLD = 0.7
# % of encounters where the synthetic grader is made to DISAGREE with the model,
# so the demo produces believable (imperfect) metrics rather than perfect ones.
_LABEL_NOISE_PCT = 15


def _seed_byte(label: str, index: int) -> int:
    """Deterministic pseudo-random byte for (label, index) — no network, repeatable."""
    return hashlib.sha256(f"{label}-{index}".encode()).digest()[0]


def make_synthetic_labeled_encounters(n: int = 200) -> tuple[LabeledEncounter, ...]:
    """Build `n` fabricated labeled encounters for the demo. Deterministic in `n`."""
    classifier = MockClassifier()
    summarizer = MockSummarizer()
    out: list[LabeledEncounter] = []

    for i in range(n):
        meta = EncounterMeta.model_validate(
            {
                "encounter_id": f"synthENC{i:04d}",
                "patient_pseudo_id": f"synthPT{i:04d}",
                "site_id": _SITES[i % len(_SITES)],
                "eye": "L" if i % 2 == 0 else "R",
                "capture_device": "synthetic_camera",
                "captured_at": _BASE + timedelta(days=i),
            }
        )
        image_bytes = f"synthetic-fundus-{i}".encode()
        classification = classifier.classify(image_bytes, meta)
        summary = summarizer.summarize(classification, meta)
        final_referable = apply_confidence_floor(classification, _CONFIDENCE_THRESHOLD)
        result = ScreeningResult.model_validate(
            {
                "meta": meta,
                "classification": classification,
                "summary": summary,
                "final_referable": final_referable,
                "created_at": _BASE + timedelta(days=i, hours=1),
            }
        )

        # Generate the grader's ground truth to AGREE with the model most of the
        # time (with `_LABEL_NOISE_PCT` injected disagreement), so the demo shows
        # believable sensitivity/specificity. This is FABRICATED agreement for
        # demonstration — NOT a measurement of real model performance.
        model_positive = result.final_referable != "no"
        disagrees = _seed_byte("noise", i) % 100 < _LABEL_NOISE_PCT
        truth_positive = model_positive != disagrees  # XOR
        if truth_positive:
            pool: tuple[DrSeverity, ...] = _REFERABLE_SEVERITIES
            csme = _seed_byte("csme", i) % 3 == 0
        else:
            pool = _NONREFERABLE_SEVERITIES
            csme = False
        severity = pool[_seed_byte("severity", i) % len(pool)]
        adjudication = GraderAdjudication.model_validate(
            {
                "encounter_id": meta.encounter_id,
                "grader_id": "synth_grader_1",
                "dr_severity": severity,
                "csme": csme,
                "icd10_codes": ("E11.359",) if severity == "pdr" else (),
                "confirmed_at": _BASE + timedelta(days=i, hours=2),
            }
        )

        interventions: tuple[Intervention, ...] = ()
        outcome: OutcomeRecord | None = None
        if severity in ("severe_npdr", "pdr") or csme:
            kind = _INTERVENTIONS[_seed_byte("intervention", i) % len(_INTERVENTIONS)]
            interventions = (
                Intervention.model_validate(
                    {
                        "encounter_id": meta.encounter_id,
                        "type": kind,
                        "cpt_code": "67XXX",
                        "telehealth_code": "GT",
                        "performed_at": _BASE + timedelta(days=i + 30),
                        "site_id": meta.site_id,
                    }
                ),
            )
            outcome = OutcomeRecord.model_validate(
                {
                    "encounter_id": meta.encounter_id,
                    "follow_up_at": _BASE + timedelta(days=i + 120),
                    "va_lines_change": (_seed_byte("va", i) % 5) - 1,
                    "oct_cmt_um": 250.0 + (_seed_byte("oct", i) % 200),
                    "oct_improved": _seed_byte("oct_improved", i) % 2 == 0,
                    "stabilized": _seed_byte("stabilized", i) % 4 != 0,
                }
            )

        out.append(
            LabeledEncounter.model_validate(
                {
                    "result": result,
                    "adjudication": adjudication,
                    "interventions": interventions,
                    "outcome": outcome,
                }
            )
        )

    return tuple(out)
