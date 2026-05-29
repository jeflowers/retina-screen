# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A portfolio/demo build of **`retina-screen`** — an AWS-native Python pipeline for retinal screening (image classification + LLM clinical summary). The canonical, load-bearing spec is `retina-screen.md` at the repo root: @retina-screen.md

Stage: pre-scaffold. Until the package exists, sessions may be (a) iterating on the spec or (b) generating code from it.

## Naming and honesty (load-bearing — do not violate)

- Package name is **`retina_screen`**. The directory is called `guam-project` for historical reasons; do not rename the package, and do not introduce "guam" or "eGuam" into code, READMEs, or commit messages.
- This is **not** the eGuam federal grant project and **not** real PHI. It is an original, synthetic reference implementation. Any README must state this plainly.
- Type-enforced safety: `ClinicalSummary` must reject construction without a non-empty `disclaimer` (model-validator). A summary is a screening AID, never a diagnosis. Don't weaken this to make a test pass.

## Stack (target — not yet installed)

Python 3.11+, **uv**-managed. Typer (CLI), pydantic v2 (DTOs, `extra="forbid"`), structlog (logging), boto3 (S3 / SQS / DynamoDB / EventBridge), anthropic SDK (optional, behind config). Tests: pytest with colocated `*_test.py`. Lint/type: ruff + mypy strict.

## Definition of done

A change is not complete until all four pass:

```
uv run ruff check src
uv run ruff format --check src
uv run mypy
uv run pytest
```

No real network or AWS in unit tests — an autouse fixture blocks sockets. AWS clients and ML backends are injected via Protocols so tests use fakes. Do not weaken strict typing or the ruff selection to make these pass.

## Architectural spine

Pluggable ML via two Protocols, selected by env config:

- `Classifier` — implementations: `MockClassifier` (default, deterministic from image hash), `SageMakerClassifier`, `BedrockImageClassifier`. Selected by `RETINA_CLASSIFIER=mock|sagemaker|bedrock`.
- `Summarizer` — implementations: `MockSummarizer` (default, templated), `BedrockSummarizer`, `AnthropicSummarizer`. Selected by `RETINA_SUMMARIZER=mock|bedrock|anthropic`.

A factory in `config.py` builds the selected pair. Unknown values fail loud at startup with `MissingSecretError` — never mid-pipeline. Confidence below the configured threshold forces `referable=review` (human in the loop), never an auto-clear.

When adding a new backend, add it behind the Protocol — never branch the pipeline on backend identity.

## Conventions

- Named exports only — every module declares `__all__`. No wildcard imports.
- DTOs constructed via `model_validate`; pydantic config `extra="forbid"`.
- Secrets via env in `config.py`; missing required key raises `MissingSecretError`.
- AWS clients and ML backends are **injected**. Don't construct boto3 clients inline in pipeline code.
- No `eval` / `exec`. Parameterized everything. Structured logging (key-value, structlog).
- Defensive parse for real backend responses: degrade to a low-confidence result with a logged warning, never an unhandled crash.

## Layout (target)

Flat package, one module per pipeline stage:

```
retina_screen.py                          # thin entrypoint
src/retina_screen/
  cli.py            config.py            dto.py
  ml/classifier.py  ml/summarizer.py
  storage/s3_store.py  storage/results_store.py
  events.py         logging_setup.py     conftest.py
samples/encounter_example.yaml
```

Tests are colocated as `<module>_test.py`. No top-level `tests/` directory.

## Out of scope

No real model training, no real PHI, no web UI, no auth/IAM authoring, no SNS/SQS consumers (EventBridge emit + README-only documentation of downstream). GCP version is documented as a porting map in the README, not built.
