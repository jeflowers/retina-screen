"""Production data contracts for the screening metrics learning loop.

These are the *real* contracts a deployment would populate to close the loop:
a model prediction (`ScreeningResult`) is paired with a grader's adjudication
(the label), the interventions performed, and the longitudinal outcome — then
`metrics.py` computes eval (model vs grader), the program/QI feed, and the
active-learning queue.

HONESTY / PHI. This is part of an original reference implementation that carries
**no real PHI** — these are real *contracts*, not real *data*. Fields that hold
PHI in production are tagged ``json_schema_extra={"phi": True}`` so governance /
redaction tooling can discover them; in a real deployment the PHI-bearing
records require the controls in BUILD.md Appendix B (KMS CMKs, CloudTrail data
events, RLS / least-privilege, retention). `patient_pseudo_id` (on the joined
`ScreeningResult.meta`) is the de-identification boundary.

Contract: pydantic v2, `extra="forbid"`, frozen, tz-aware datetimes enforced.
"""

from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
)

from retina_screen.dto import ScreeningResult

__all__ = [
    "DrSeverity",
    "EvalMetrics",
    "GraderAdjudication",
    "Icd10Code",
    "Intervention",
    "InterventionType",
    "LabeledEncounter",
    "OutcomeRecord",
    "ProgramMetrics",
]


# International Clinical Diabetic Retinopathy (ICDR) severity scale.
DrSeverity = Literal["none", "mild_npdr", "moderate_npdr", "severe_npdr", "pdr"]
InterventionType = Literal["prp", "anti_vegf", "vitrectomy", "none"]

# ICD-10-CM diagnosis code, e.g. "E11.359": a letter (not U), digit,
# digit/A/B, optional dotted 1-4 char subclassification.
_ICD10_PATTERN = r"^[A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?$"
Icd10Code = Annotated[str, StringConstraints(pattern=_ICD10_PATTERN)]
_NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]

# Marks a field as PHI-bearing in production (for governance / redaction tooling).
_PHI: dict[str, JsonValue] = {"phi": True}


class _DTO(BaseModel):
    """Base: forbid unknown fields, immutable after construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GraderAdjudication(_DTO):
    """Human grader's ground-truth read for an encounter — the training label."""

    encounter_id: str = Field(min_length=1)
    grader_id: str = Field(min_length=1)
    dr_severity: DrSeverity = Field(json_schema_extra=_PHI)
    csme: bool = Field(json_schema_extra=_PHI)
    icd10_codes: tuple[Icd10Code, ...] = Field(default=(), json_schema_extra=_PHI)
    confirmed_at: AwareDatetime


class Intervention(_DTO):
    """A treatment performed after screening (carries the billing codes)."""

    encounter_id: str = Field(min_length=1)
    type: InterventionType = Field(json_schema_extra=_PHI)
    cpt_code: _NonEmptyStr | None = None
    telehealth_code: _NonEmptyStr | None = None
    performed_at: AwareDatetime
    site_id: str = Field(min_length=1)


class OutcomeRecord(_DTO):
    """Longitudinal outcome after intervention — the outcome label."""

    encounter_id: str = Field(min_length=1)
    follow_up_at: AwareDatetime
    va_lines_change: int = Field(json_schema_extra=_PHI)  # ETDRS lines; + = improvement
    oct_cmt_um: float | None = Field(default=None, ge=0.0, json_schema_extra=_PHI)
    oct_improved: bool = Field(json_schema_extra=_PHI)
    stabilized: bool = Field(json_schema_extra=_PHI)


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
