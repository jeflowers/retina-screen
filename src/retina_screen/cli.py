"""Typer CLI + run_pipeline orchestration.

The pipeline:
  1. Store raw image immutably (replay source) — IF a bucket is configured.
  2. Classify via the injected Classifier.
  3. Summarize via the injected Summarizer.
  4. Apply the confidence-threshold safety rule → final_referable.
  5. Build a ScreeningResult.
  6. Persist idempotently — IF a table is configured.
  7. Emit the appropriate EventBridge event — IF a bus is configured.

AWS-touching steps degrade gracefully when their resource isn't configured: the
step logs `skipped` with a reason and the pipeline continues. This is how the
mock-default mode runs with no AWS credentials and no network — only the ML
steps need clients, and the mock implementations don't need any.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import boto3
import structlog
import typer
import yaml

from retina_screen.config import (
    Config,
    MissingSecretError,
    apply_confidence_floor,
    build_classifier,
    build_summarizer,
    load_config,
)
from retina_screen.dto import EncounterMeta, ScreeningResult
from retina_screen.events import emit_screening_event
from retina_screen.logging_setup import setup_logging
from retina_screen.ml.classifier import Classifier
from retina_screen.ml.summarizer import Summarizer
from retina_screen.storage.results_store import put_screening_result
from retina_screen.storage.s3_store import put_raw_image

if TYPE_CHECKING:
    from mypy_boto3_dynamodb.service_resource import Table
    from mypy_boto3_events import EventBridgeClient
    from mypy_boto3_s3 import S3Client

__all__ = ["PipelineDeps", "app", "main", "run_pipeline"]


app = typer.Typer(
    add_completion=False,
    help="retina-screen: tele-ophthalmology screening pipeline (demo).",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Force subcommand mode so `retina-screen screen ...` is the call form."""


@dataclass(frozen=True)
class PipelineDeps:
    """Injected dependencies for one pipeline run.

    All AWS clients are optional — when omitted, the corresponding step is
    skipped with a log line. This is what makes the mock-default mode runnable
    without AWS credentials.
    """

    config: Config
    classifier: Classifier
    summarizer: Summarizer
    s3_client: S3Client | None = None
    dynamodb_table: Table | None = None
    events_client: EventBridgeClient | None = None


def run_pipeline(deps: PipelineDeps, image_bytes: bytes, meta: EncounterMeta) -> ScreeningResult:
    """Run the screening pipeline end-to-end. Returns the persisted ScreeningResult."""
    log = structlog.get_logger().bind(
        encounter_id=meta.encounter_id,
        site_id=meta.site_id,
        eye=meta.eye,
    )

    # 1. Raw image → S3 (immutable replay source, BEFORE downstream work)
    if deps.s3_client is not None and deps.config.raw_image_bucket:
        s3_key = put_raw_image(
            deps.s3_client,
            deps.config.raw_image_bucket,
            meta.site_id,
            meta.encounter_id,
            image_bytes,
        )
        log.info("raw_image_stored", s3_key=s3_key)
    else:
        log.info("raw_image_skipped", reason="no_bucket_configured")

    # 2. Classify
    classification = deps.classifier.classify(image_bytes, meta)
    log.info(
        "classified",
        model=classification.model_name,
        version=classification.model_version,
        model_referable=classification.referable,
        confidence=classification.confidence,
    )

    # 3. Summarize
    summary = deps.summarizer.summarize(classification, meta)
    log.info("summarized", summarizer=deps.summarizer.name, urgency=summary.urgency)

    # 4. Confidence floor (safety rule — never auto-clear a low-confidence call)
    final_referable = apply_confidence_floor(classification, deps.config.confidence_threshold)
    log.info(
        "final_referable_decided",
        final=final_referable,
        model_said=classification.referable,
        threshold=deps.config.confidence_threshold,
    )

    # 5. Result
    result = ScreeningResult(
        meta=meta,
        classification=classification,
        summary=summary,
        final_referable=final_referable,
        created_at=datetime.now(UTC),
    )

    # 6. Persist (idempotent on encounter_id)
    if deps.dynamodb_table is not None:
        was_new = put_screening_result(deps.dynamodb_table, result)
        log.info("result_persisted", was_new=was_new)
    else:
        log.info("result_persisted_skipped", reason="no_table_configured")

    # 7. Emit event
    if deps.events_client is not None and deps.config.eventbridge_bus:
        event_type = emit_screening_event(deps.events_client, deps.config.eventbridge_bus, result)
        log.info("event_emitted", event_type=event_type)
    else:
        log.info("event_emit_skipped", reason="no_bus_configured")

    return result


@app.command()
def screen(
    image: Annotated[Path, typer.Option("--image", help="Path to fundus image (jpg/png).")],
    encounter: Annotated[
        Path, typer.Option("--encounter", help="Path to encounter metadata YAML.")
    ],
) -> None:
    """Run the screening pipeline on a single image + encounter."""
    setup_logging()
    try:
        config = load_config()
        classifier = build_classifier(config)
        summarizer = build_summarizer(config)
    except MissingSecretError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except NotImplementedError as e:
        typer.echo(f"Backend not implemented: {e}", err=True)
        raise typer.Exit(code=1) from e

    deps = PipelineDeps(
        config=config,
        classifier=classifier,
        summarizer=summarizer,
        s3_client=_maybe_s3(config),
        dynamodb_table=_maybe_table(config),
        events_client=_maybe_events(config),
    )

    image_bytes = image.read_bytes()
    meta = _load_encounter_yaml(encounter)
    result = run_pipeline(deps, image_bytes, meta)

    typer.echo(result.model_dump_json(indent=2))


def main() -> None:
    """Entry point for the `retina-screen` console script."""
    app()


def _load_encounter_yaml(path: Path) -> EncounterMeta:
    data: Any = yaml.safe_load(path.read_text())
    return EncounterMeta.model_validate(data)


def _maybe_s3(config: Config) -> S3Client | None:
    if not config.raw_image_bucket:
        return None
    return boto3.client("s3", region_name=config.aws_region)


def _maybe_table(config: Config) -> Table | None:
    if not config.results_table:
        return None
    return boto3.resource("dynamodb", region_name=config.aws_region).Table(config.results_table)


def _maybe_events(config: Config) -> EventBridgeClient | None:
    if not config.eventbridge_bus:
        return None
    return boto3.client("events", region_name=config.aws_region)
