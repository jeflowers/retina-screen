"""Tests for EventBridge emission.

Acceptance: referable=yes/review → ReferralNeeded; referable=no → ScreeningCompleted.
A non-zero FailedEntryCount must surface as an error, not silently succeed.
"""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from retina_screen.dto import (
    Classification,
    ClinicalSummary,
    ConditionProbability,
    EncounterMeta,
    Referable,
    ScreeningResult,
)
from retina_screen.events import EVENT_SOURCE, EventEmitFailedError, emit_screening_event


def _result(final_referable: Referable = "yes") -> ScreeningResult:
    return ScreeningResult(
        meta=EncounterMeta(
            encounter_id="enc_001",
            patient_pseudo_id="pt_001",
            site_id="site_A",
            eye="L",
            capture_device="topcon_nw400",
            captured_at=datetime(2026, 5, 28, 14, 30, tzinfo=UTC),
        ),
        classification=Classification(
            conditions=(
                ConditionProbability(name="diabetic_retinopathy_grade_2", probability=0.78),
                ConditionProbability(name="normal", probability=0.22),
            ),
            referable="yes",
            model_name="mock-classifier",
            model_version="0.1.0",
            confidence=0.85,
        ),
        summary=ClinicalSummary(
            findings="Probable referable disease.",
            recommended_action="Refer to ophthalmology.",
            urgency="soon",
            disclaimer="Screening aid, not a diagnosis.",
        ),
        final_referable=final_referable,
        created_at=datetime(2026, 5, 28, 14, 35, tzinfo=UTC),
    )


def _ok_response() -> dict[str, object]:
    return {"FailedEntryCount": 0, "Entries": [{"EventId": "abc"}]}


def test_referable_yes_emits_referral_needed() -> None:
    client = MagicMock()
    client.put_events.return_value = _ok_response()

    event_type = emit_screening_event(client, "bus-main", _result(final_referable="yes"))

    assert event_type == "ReferralNeeded"
    entry = client.put_events.call_args.kwargs["Entries"][0]
    assert entry["EventBusName"] == "bus-main"
    assert entry["Source"] == EVENT_SOURCE
    assert entry["DetailType"] == "ReferralNeeded"


def test_referable_review_also_emits_referral_needed() -> None:
    """Low-confidence cases still need human attention."""
    client = MagicMock()
    client.put_events.return_value = _ok_response()

    event_type = emit_screening_event(client, "bus", _result(final_referable="review"))

    assert event_type == "ReferralNeeded"


def test_referable_no_emits_screening_completed() -> None:
    client = MagicMock()
    client.put_events.return_value = _ok_response()

    event_type = emit_screening_event(client, "bus", _result(final_referable="no"))

    assert event_type == "ScreeningCompleted"
    entry = client.put_events.call_args.kwargs["Entries"][0]
    assert entry["DetailType"] == "ScreeningCompleted"


def test_detail_payload_is_valid_json_with_event_fields() -> None:
    client = MagicMock()
    client.put_events.return_value = _ok_response()

    emit_screening_event(client, "bus", _result(final_referable="yes"))

    detail_str = client.put_events.call_args.kwargs["Entries"][0]["Detail"]
    detail = json.loads(detail_str)
    assert detail["event_type"] == "ReferralNeeded"
    assert detail["encounter_id"] == "enc_001"
    assert detail["site_id"] == "site_A"
    assert detail["urgency"] == "soon"
    assert detail["model_name"] == "mock-classifier"


def test_failed_entries_raise_event_emit_failed_error() -> None:
    """A non-zero FailedEntryCount must surface — silent failure is the worst outcome."""
    client = MagicMock()
    client.put_events.return_value = {"FailedEntryCount": 1, "Entries": [{}]}

    with pytest.raises(EventEmitFailedError, match="encounter_id=enc_001"):
        emit_screening_event(client, "bus", _result())


def test_referral_event_carries_model_identity() -> None:
    """Downstream consumers need to know which model flagged this case."""
    client = MagicMock()
    client.put_events.return_value = _ok_response()

    emit_screening_event(client, "bus", _result())

    detail = json.loads(client.put_events.call_args.kwargs["Entries"][0]["Detail"])
    assert detail["model_name"] == "mock-classifier"
