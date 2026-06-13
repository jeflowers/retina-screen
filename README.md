# retina-screen

A multi-site tele-ophthalmology screening pipeline on AWS: a clinic uploads a
fundus image for a patient encounter, the system classifies it for referable
disease, generates a plain-language clinical summary via an LLM, persists the
structured result, and emits an event when a case needs specialist referral.

## What this is (and isn't)

This is an **original, synthetic reference implementation** I built to
demonstrate the architecture. It is **not** the eGuam federal grant project,
**not** an FDA-path diagnostic, and **never** sees real patient data. The
clinical domain knowledge comes from real tele-ophthalmology work; the code is
mine and new.

The point of the build is the *seam*: ML and AWS clients are injected via
Protocols, so the same pipeline runs on mock implementations (zero AWS spend,
no model access) or on real SageMaker / Bedrock / Anthropic backends with
nothing more than a config change.

If you cloned this from a job application: I can talk you through any part of
the code, the trade-offs I made, and what would change to take it to
production. The honest version of "what's here" is what's in this README.

## Quick start

```bash
uv sync
uv run retina-screen screen \
  --image samples/fundus.jpg \
  --encounter samples/encounter_example.yaml
```

That runs the full pipeline with the default mock backends, no AWS credentials
required. The screening result lands on stdout as JSON; structured logs go to
stderr.

`samples/fundus.jpg` is a tiny synthetic 1×1 placeholder. The mock classifier
hashes the bytes; any valid JPEG works.

## Architecture

```
encounter + image
      │
      ▼
[1] put_raw_image   →  S3 (immutable replay source)
      │
      ▼
[2] Classifier      →  Classification {conditions, referable, confidence}
      │                (MockClassifier | SageMakerClassifier | BedrockImageClassifier)
      ▼
[3] Summarizer      →  ClinicalSummary {findings, recommended_action, urgency, disclaimer}
      │                (MockSummarizer | BedrockSummarizer | AnthropicSummarizer)
      ▼
[4] confidence floor → ScreeningResult.final_referable
      │
      ▼
[5] put_screening_result  →  DynamoDB (idempotent)
      │
      ▼
[6] emit_screening_event  →  EventBridge
                              {ReferralNeeded | ScreeningCompleted}
```

Two `Protocol`s — `Classifier` and `Summarizer` — are the architectural spine.
Every pipeline step depends only on the Protocol, so swapping a backend never
requires touching the orchestration in `cli.py`. AWS clients (S3, DynamoDB,
EventBridge) are injected at the CLI boundary and threaded through
`PipelineDeps`, so every step is unit-testable with a fake.

**The pipeline degrades gracefully** when AWS resources aren't configured: if
no bucket name is set, the raw-image step logs `raw_image_skipped` and
continues; the persistence and event-emission steps behave the same way.
That's how the default `mock`/`mock` setup runs with no AWS credentials.

## Configuration

All configuration comes from environment variables, read once at startup by
`config.load_config()`. Missing or invalid values raise `MissingSecretError`
at startup, never mid-pipeline.

| Variable | Default | Required when |
|---|---|---|
| `RETINA_CLASSIFIER` | `mock` | always one of {mock, sagemaker, bedrock} |
| `RETINA_SUMMARIZER` | `mock` | always one of {mock, bedrock, anthropic} |
| `RETINA_CONFIDENCE_THRESHOLD` | `0.7` | always; must be in [0.0, 1.0] |
| `AWS_REGION` | `us-west-2` | |
| `RETINA_RAW_BUCKET` | unset | persistence enabled when set |
| `RETINA_RESULTS_TABLE` | unset | persistence enabled when set |
| `RETINA_EVENTBRIDGE_BUS` | unset | event emission enabled when set |
| `SAGEMAKER_ENDPOINT` | unset | `RETINA_CLASSIFIER=sagemaker` |
| `BEDROCK_CLASSIFIER_MODEL_ID` | unset | `RETINA_CLASSIFIER=bedrock` |
| `BEDROCK_SUMMARIZER_MODEL_ID` | unset | `RETINA_SUMMARIZER=bedrock` |
| `ANTHROPIC_API_KEY` | unset | `RETINA_SUMMARIZER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-opus-4-7` | |

## Safety invariants (enforced in code, not just docs)

These two properties are load-bearing and cannot be silently weakened.

**1. Type-enforced disclaimer.** The `ClinicalSummary` Pydantic model has a
`model_validator` that rejects empty or whitespace-only disclaimers. A summary
that lacks the screening-aid disclaimer literally cannot be constructed — not
by direct construction, not via `model_validate`, not by a backend that
forgot. See `src/retina_screen/dto.py` and the tests in
`src/retina_screen/dto_test.py::test_clinical_summary_with_empty_disclaimer_raises`.

The Bedrock and Anthropic summarizers go further: the disclaimer is set by
*code* (`BACKEND_DISCLAIMER` in `src/retina_screen/ml/summarizer.py`), not by
the model output. Belt-and-suspenders with the validator.

**2. Confidence floor.** Any classification with confidence below the
configured threshold is forced to `referable="review"` regardless of what the
model said. A low-confidence prediction never auto-clears. See
`config.apply_confidence_floor()` and
`config_test.py::test_floor_never_auto_clears_low_confidence_referable`.

## DynamoDB key design

```
Primary table:
  PK = site_id          ← per-site queries: "all encounters at site_demo_clinic_01"
  SK = encounter_id     ← uniqueness within a site; sorts as the encounter key

GSI:
  PK = patient_pseudo_id ← per-patient queries: "all encounters for pt_abc123"
  SK = encounter_id
```

**Why this shape**

- The composite key `(site_id, encounter_id)` makes per-site listing a single
  `Query` against the table partition, no scan.
- Patient queries cross sites (referrals move patients around), so they go
  through the GSI rather than the primary partition.
- Idempotency is a conditional write keyed on the composite:
  `attribute_not_exists(encounter_id)`. Re-processing the same encounter twice
  is safe — the second `PutItem` raises `ConditionalCheckFailedException`,
  which the adapter swallows and returns `False`.

Top-level item attributes flatten the keys (`site_id`, `encounter_id`,
`patient_pseudo_id`, `created_at`, `final_referable`) so queries don't have to
unpack the nested `result` blob. The full ScreeningResult lives under `result`.

DynamoDB rejects native Python floats — `results_store._floats_to_decimals`
recursively converts before `put_item`.

## Required IAM permissions

| Service | Action(s) | When required |
|---|---|---|
| S3 | `s3:PutObject`, `s3:GetObject` on `arn:aws:s3:::<bucket>/raw/*` | `RETINA_RAW_BUCKET` set |
| DynamoDB | `dynamodb:PutItem` (with `ConditionExpression`), `dynamodb:Query` on the GSI | `RETINA_RESULTS_TABLE` set |
| EventBridge | `events:PutEvents` on the configured bus | `RETINA_EVENTBRIDGE_BUS` set |
| SageMaker | `sagemaker:InvokeEndpoint` on the configured endpoint | `RETINA_CLASSIFIER=sagemaker` |
| Bedrock | `bedrock:InvokeModel` on the configured model id(s) | classifier or summarizer = `bedrock` |

The Anthropic backend doesn't use AWS at all — it goes to `api.anthropic.com`
with the API key.

## Porting to GCP

The Protocol seam means a GCP port changes the adapters under `storage/`,
`events.py`, and the ML backends — the pipeline orchestration in `cli.py`
doesn't move.

| AWS (this build) | Closest GCP equivalent | Adapter that changes |
|---|---|---|
| S3 | Cloud Storage (GCS) | `storage/s3_store.py` |
| DynamoDB | Firestore (Native mode) | `storage/results_store.py` |
| EventBridge | Eventarc | `events.py` |
| SageMaker | Vertex AI Endpoints | `ml/classifier.py` (SageMaker impl) |
| Bedrock (multimodal) | Vertex AI / Model Garden | `ml/classifier.py` + `ml/summarizer.py` (Bedrock impls) |
| SQS (downstream) | Pub/Sub | not implemented here |
| SNS (downstream) | Pub/Sub | not implemented here |

These are *closest* equivalents, not exact. Notable differences:

- **Firestore vs DynamoDB**: Firestore's conditional write semantics are
  expressed via `Precondition.exists=False` rather than a
  `ConditionExpression` string. Same idempotency guarantee, different API.
- **Eventarc vs EventBridge**: Eventarc routes Cloud-native event sources
  (Audit Logs, Pub/Sub) more naturally; the explicit `PutEvents`-style emit
  pattern is less idiomatic — you'd typically publish to Pub/Sub and have
  Eventarc trigger downstream.
- **Vertex AI vs SageMaker**: similar endpoint-invocation shape, but Vertex
  has stricter input formatting for multimodal models.

The pipeline contract — "store image immutably first, classify, summarize,
apply safety floor, persist idempotently, emit event" — does not change.

## Out of scope

Intentional omissions, with rationale:

- **No model training.** Mock backends are deterministic from `sha256(image)`;
  real backends point at endpoints/models that someone else trained.
- **No real PHI.** Sample encounter data is synthetic. Sample image is a 1×1
  placeholder.
- **No web UI.** Two entry points only: the CLI and (in the README's design
  spec) an S3-upload-triggered handler. The CLI is what's implemented.
- **No SNS/SQS consumer code.** EventBridge emit is implemented; downstream
  routing is documented but not built.
- **No IAM policy authoring.** Required permissions are listed above; the
  actual policies are deployment-specific.
- **No GCP version, yet.** Documented as a porting map (above) rather than a
  parallel build — the seam is the proof.

## Development

```bash
# Install dev + runtime deps
uv sync

# Run the four definition-of-done gates
uv run ruff check src
uv run ruff format --check src
uv run mypy
uv run pytest

# Or run them via the bundled Claude Code skill
# (see .claude/skills/done-check/SKILL.md)
```

### Project layout

```
src/retina_screen/
  cli.py                CLI + run_pipeline orchestration
  config.py             env-driven config, MissingSecretError, factory
  dto.py                pydantic v2 DTOs (with disclaimer validator)
  events.py             EventBridge emission
  logging_setup.py      structlog → JSON to stderr
  conftest.py           autouse socket-block fixture (test hermeticity)
  ml/
    classifier.py       Classifier Protocol + Mock / SageMaker / Bedrock impls
    summarizer.py       Summarizer Protocol + Mock / Bedrock / Anthropic impls
  storage/
    s3_store.py         raw-image put/get
    results_store.py    DynamoDB put with conditional-write idempotency
  *_test.py             colocated tests (one per module)
samples/
  fundus.jpg            synthetic 1×1 placeholder
  encounter_example.yaml synthetic encounter metadata
```

### Conventions

- Every module declares `__all__`. No wildcard imports.
- DTOs are constructed via `model_validate`; pydantic config is `extra="forbid"`.
- AWS clients and ML backends are *injected*, never constructed inside
  pipeline code.
- Secrets are read in `config.py`; missing required values raise
  `MissingSecretError` at startup.
- Tests are colocated as `<module>_test.py`. No top-level `tests/` directory.
- The autouse `_block_network` fixture in `conftest.py` blocks `socket.connect`
  — any test that accidentally tries to reach the network fails loudly.
