"""DynamoDB results storage with idempotent writes.

KEY DESIGN
  PK = site_id
  SK = encounter_id
  GSI: patient_pseudo_id (PK) + encounter_id (SK)

The composite key supports per-site queries by partition; the GSI supports
per-patient lookups across sites. Idempotency uses a conditional write keyed
on (site_id, encounter_id) so re-processing the same encounter persists
exactly once.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError

from retina_screen.dto import ScreeningResult

if TYPE_CHECKING:
    from mypy_boto3_dynamodb.service_resource import Table

__all__ = ["put_screening_result"]


def put_screening_result(table: Table, result: ScreeningResult) -> bool:
    """Idempotent put. Returns True if newly written, False if already exists.

    Any DynamoDB error other than ConditionalCheckFailedException propagates —
    "already exists" is a normal outcome, "throttled" or "service unavailable"
    are not.
    """
    item = _to_item(result)
    try:
        table.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(encounter_id)",
        )
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise
    return True


def _to_item(result: ScreeningResult) -> dict[str, Any]:
    """Project a ScreeningResult into a DynamoDB-shaped item.

    Top-level attributes pull out the keys and the GSI partition so the
    table can be queried without unpacking the nested `result` blob.
    """
    data = result.model_dump(mode="json")
    return {
        "site_id": result.meta.site_id,
        "encounter_id": result.meta.encounter_id,
        "patient_pseudo_id": result.meta.patient_pseudo_id,
        "created_at": data["created_at"],
        "final_referable": result.final_referable,
        "result": _floats_to_decimals(data),
    }


def _floats_to_decimals(obj: Any) -> Any:
    """Recursively convert floats to Decimals — DynamoDB rejects native floats."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _floats_to_decimals(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_floats_to_decimals(v) for v in obj]
    return obj
