"""Tests for the DynamoDB results adapter (acceptance #5: idempotency).

Verifies the conditional-write pattern: a duplicate encounter_id is detected,
returns False, and is treated as success (safe). Any other DynamoDB error is
propagated, since "already exists" is normal but "throttled" is not.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from retina_screen.dto import (
    Classification,
    ClinicalSummary,
    ConditionProbability,
    EncounterMeta,
    ScreeningResult,
)
from retina_screen.storage.results_store import put_screening_result


def _result(encounter_id: str = "enc_001", site_id: str = "site_A") -> ScreeningResult:
    return ScreeningResult(
        meta=EncounterMeta(
            encounter_id=encounter_id,
            patient_pseudo_id="pt_001",
            site_id=site_id,
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
            disclaimer="This is an automated screening aid, not a clinical diagnosis.",
        ),
        final_referable="yes",
        created_at=datetime(2026, 5, 28, 14, 35, tzinfo=UTC),
    )


def _conditional_failure() -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": "ConditionalCheckFailedException", "Message": "exists"}},
        operation_name="PutItem",
    )


def _other_failure(code: str = "ProvisionedThroughputExceededException") -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": "throttled"}},
        operation_name="PutItem",
    )


def test_put_screening_result_writes_when_new() -> None:
    table = MagicMock()
    written = put_screening_result(table, _result())

    assert written is True
    table.put_item.assert_called_once()
    call_kwargs = table.put_item.call_args.kwargs
    assert call_kwargs["ConditionExpression"] == "attribute_not_exists(encounter_id)"

    item = call_kwargs["Item"]
    assert item["site_id"] == "site_A"
    assert item["encounter_id"] == "enc_001"
    assert item["patient_pseudo_id"] == "pt_001"
    assert item["final_referable"] == "yes"


def test_put_screening_result_returns_false_on_duplicate() -> None:
    """Idempotency: a second write for the same encounter_id is a no-op."""
    table = MagicMock()
    table.put_item.side_effect = _conditional_failure()

    written = put_screening_result(table, _result())

    assert written is False  # already exists — safe


def test_put_screening_result_propagates_other_client_errors() -> None:
    """Throttling, throttled-table, or service errors must propagate."""
    table = MagicMock()
    table.put_item.side_effect = _other_failure()

    with pytest.raises(ClientError):
        put_screening_result(table, _result())


def test_put_screening_result_converts_floats_to_decimals() -> None:
    """DynamoDB rejects native floats — pydantic emits them, the adapter must convert."""
    table = MagicMock()
    put_screening_result(table, _result())

    item = table.put_item.call_args.kwargs["Item"]
    nested = item["result"]
    assert isinstance(nested["classification"]["confidence"], Decimal)
    assert isinstance(nested["classification"]["conditions"][0]["probability"], Decimal)


def test_put_screening_result_idempotent_double_call() -> None:
    """Twice in a row: first writes, second hits the conditional failure path."""
    table = MagicMock()

    # First call: succeeds. Second call: ConditionalCheckFailedException.
    table.put_item.side_effect = [{"ResponseMetadata": {}}, _conditional_failure()]

    first = put_screening_result(table, _result())
    second = put_screening_result(table, _result())

    assert first is True
    assert second is False
    assert table.put_item.call_count == 2


def test_top_level_attributes_support_per_site_and_per_patient_queries() -> None:
    """The flattened keys make site/patient queries possible without unpacking `result`."""
    table = MagicMock()
    put_screening_result(table, _result(site_id="site_demo_01"))

    item: dict[str, Any] = table.put_item.call_args.kwargs["Item"]
    # PK (site_id) — per-site query partition
    assert item["site_id"] == "site_demo_01"
    # GSI partition — per-patient query
    assert "patient_pseudo_id" in item
    # Sortable by created_at within a partition
    assert "created_at" in item
