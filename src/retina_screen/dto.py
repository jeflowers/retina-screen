"""Pydantic v2 DTOs for the screening pipeline.

All DTOs forbid extra fields. ClinicalSummary enforces a non-empty disclaimer at
the type level — a summary that drops the screening-aid disclaimer cannot exist.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Classification",
    "ClinicalSummary",
    "ConditionProbability",
    "EncounterMeta",
    "Eye",
    "Referable",
    "ReferralNeeded",
    "ScreeningCompleted",
    "ScreeningResult",
    "Urgency",
]


Eye = Literal["L", "R"]
Referable = Literal["yes", "no", "review"]
Urgency = Literal["routine", "soon", "urgent"]


class _DTO(BaseModel):
    """Base for every DTO: forbid unknown fields, immutable after construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EncounterMeta(_DTO):
    encounter_id: str = Field(min_length=1)
    patient_pseudo_id: str = Field(min_length=1)
    site_id: str = Field(min_length=1)
    eye: Eye
    capture_device: str = Field(min_length=1)
    captured_at: datetime


class ConditionProbability(_DTO):
    name: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)


class Classification(_DTO):
    conditions: tuple[ConditionProbability, ...]
    referable: Referable
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class ClinicalSummary(_DTO):
    findings: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
    urgency: Urgency
    disclaimer: str

    @model_validator(mode="after")
    def _require_disclaimer(self) -> "ClinicalSummary":
        if not self.disclaimer.strip():
            raise ValueError(
                "ClinicalSummary requires a non-empty disclaimer "
                "(this is a screening aid, never a diagnosis)."
            )
        return self


class ScreeningResult(_DTO):
    meta: EncounterMeta
    classification: Classification
    summary: ClinicalSummary
    final_referable: Referable
    created_at: datetime


class _BaseEvent(_DTO):
    encounter_id: str = Field(min_length=1)
    site_id: str = Field(min_length=1)
    occurred_at: datetime


class ReferralNeeded(_BaseEvent):
    event_type: Literal["ReferralNeeded"] = "ReferralNeeded"
    urgency: Urgency
    model_name: str = Field(min_length=1)


class ScreeningCompleted(_BaseEvent):
    event_type: Literal["ScreeningCompleted"] = "ScreeningCompleted"
