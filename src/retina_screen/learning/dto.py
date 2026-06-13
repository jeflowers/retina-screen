"""SYNTHETIC demo DTOs for the metrics learning loop — no real PHI.

These model the ground-truth labels and outcomes that close the screening
learning loop: a model prediction (`ScreeningResult`) is paired with a grader's
adjudication (the label), the interventions performed, and the longitudinal
outcome. All data flowing through these in this package is fabricated — see
`retina_screen.learning.synthetic`.

Same contract as the core DTOs: pydantic v2, `extra="forbid"`, frozen.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from retina_screen.dto import ScreeningResult

__all__ = [
    "DrSeverity",
    "EvalMetrics",
    "GraderAdjudication",
    "Intervention",
    "InterventionType",
    "LabeledEncounter",
    "OutcomeRecord",
    "ProgramMetrics",
]


# International Clinical Diabetic Retinopathy (ICDR) severity scale.
DrSeverity = Literal["none", "mild_npdr", "moderate_npdr", "severe_npdr", "pdr"]
InterventionType = Literal["prp", "anti_vegf", "vitrectomy", "none"]


class _DTO(BaseModel):
    """Base: forbid unknown fields, immutable after construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GraderAdjudication(_DTO):
    """Human grader's ground-truth read for an encounter — the training label."""

    encounter_id: str = Field(min_length=1)
    grader_id: str = Field(min_length=1)
    dr_severity: DrSeverity
    csme: bool
    icd10_codes: tuple[str, ...] = ()
    confirmed_at: datetime


class Intervention(_DTO):
    """A treatment performed after screening (carries the billing codes)."""

    encounter_id: str = Field(min_length=1)
    type: InterventionType
    cpt_code: str | None = None
    telehealth_code: str | None = None
    performed_at: datetime
    site_id: str = Field(min_length=1)


class OutcomeRecord(_DTO):
    """Longitudinal outcome after intervention — the outcome label."""

    encounter_id: str = Field(min_length=1)
    follow_up_at: datetime
    va_lines_change: int  # ETDRS lines; positive = improvement
    oct_cmt_um: float | None = Field(default=None, ge=0.0)
    oct_improved: bool
    stabilized: bool


class LabeledEncounter(_DTO):
    """One training example: prediction joined to label + interventions + outcome."""

    result: ScreeningResult
    adjudication: GraderAdjudication
    interventions: tuple[Intervention, ...] = ()
    outcome: OutcomeRecord | None = None


class EvalMetrics(_DTO):
    """Model-vs-grader evaluation over a labeled set (referable as binary)."""

    n: int = Field(ge=0)
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    sensitivity: float = Field(ge=0.0, le=1.0)
    specificity: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)


class ProgramMetrics(_DTO):
    """Aggregate program / QI metrics over a labeled set."""

    screened: int = Field(ge=0)
    dr_prevalence: float = Field(ge=0.0, le=1.0)
    pdr_rate: float = Field(ge=0.0, le=1.0)
    csme_rate: float = Field(ge=0.0, le=1.0)
    prp_rate: float = Field(ge=0.0, le=1.0)
    anti_vegf_rate: float = Field(ge=0.0, le=1.0)
    vitrectomy_rate: float = Field(ge=0.0, le=1.0)
