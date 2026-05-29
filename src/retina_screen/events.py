"""EventBridge event emission.

Routes each completed screening to one of two events:
  - ReferralNeeded     — final_referable in {yes, review}; downstream consumers
                         (clinician notification, work-queue) act on these
  - ScreeningCompleted — final_referable == no; for analytics/audit only

The "review" case emits ReferralNeeded because it still needs human attention,
even though the model itself didn't auto-clear or auto-refer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from retina_screen.dto import ReferralNeeded, ScreeningCompleted, ScreeningResult

if TYPE_CHECKING:
    from mypy_boto3_events import EventBridgeClient

__all__ = ["EVENT_SOURCE", "EventEmitFailedError", "emit_screening_event"]


EVENT_SOURCE = "retina-screen"


class EventEmitFailedError(RuntimeError):
    """Raised when EventBridge reports any failed entries."""


def emit_screening_event(
    client: EventBridgeClient,
    bus_name: str,
    result: ScreeningResult,
) -> str:
    """Emit the appropriate event for this result. Returns the event_type emitted."""
    event = _build_event(result)
    response = client.put_events(
        Entries=[
            {
                "EventBusName": bus_name,
                "Source": EVENT_SOURCE,
                "DetailType": event.event_type,
                "Detail": event.model_dump_json(),
            }
        ]
    )
    failed = response.get("FailedEntryCount", 0)
    if failed:
        raise EventEmitFailedError(
            f"EventBridge rejected {failed} entries for {event.event_type} "
            f"(encounter_id={result.meta.encounter_id})"
        )
    return event.event_type


def _build_event(result: ScreeningResult) -> ReferralNeeded | ScreeningCompleted:
    occurred_at = datetime.now(UTC)
    if result.final_referable in ("yes", "review"):
        return ReferralNeeded(
            encounter_id=result.meta.encounter_id,
            site_id=result.meta.site_id,
            occurred_at=occurred_at,
            urgency=result.summary.urgency,
            model_name=result.classification.model_name,
        )
    return ScreeningCompleted(
        encounter_id=result.meta.encounter_id,
        site_id=result.meta.site_id,
        occurred_at=occurred_at,
    )
