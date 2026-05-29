# retina-screen — Claude Code build prompt (AWS Python, AI/ML)

A paste-able specification for Claude Code that scaffolds an AWS-native Python
project: a tele-ophthalmology retinal screening platform with image-based
disease classification and LLM clinical summarization. ML is pluggable — mock
implementations by default so the whole thing runs and tests without AWS or a
real model, real backends swappable via config.

This mirrors the structure of the osint-hiring prompts: spec-first
(objective / scope / out-of-scope / constraints / acceptance), lint-green is
done, dependency-injected so tests are hermetic.

---

## On naming and honesty (read before running)

This is an **original reference implementation**, not the eGuam federal grant
project. Do not name it eGuam, do not present it as the federal system, and do
not imply it processes real patient data. It is *inspired by* tele-ophthalmology
screening patterns — a portfolio/demo build you own outright. Package name:
`retina_screen`. In any README or interview framing: "an AWS-native reference
implementation of a multi-site retinal screening pipeline I built to demonstrate
the architecture; the domain knowledge comes from real tele-ophthalmology work,
the code is original and synthetic."

This keeps the line clean between real experience (which you describe honestly)
and a demonstrable artifact (which is new and yours).

---

## Shared definition of done

```
uv run ruff check src && uv run ruff format --check src   # clean
uv run mypy                                               # clean (strict)
uv run pytest                                             # all green, no network
```

No real network or AWS calls in unit tests — an autouse fixture blocks sockets
(copy the pattern from osint-hiring's conftest). ML backends and AWS clients are
injected so tests use fakes. Do not weaken strict typing or the ruff set to pass.

---

## PROMPT — scaffold retina-screen

```
ROLE
You are building a new AWS-native Python project from scratch. Be precise,
type-strict, and test-hermetic. Default to the simplest design that satisfies the
spec; add complexity only where the spec calls for it.

OBJECTIVE
Build `retina-screen`: a multi-site retinal screening pipeline on AWS. A clinic
uploads a fundus (retinal) image for a patient encounter; the system classifies it
for referable disease, generates a plain-language clinical summary via an LLM,
persists structured results, and emits an event when a case needs specialist
referral. ML is PLUGGABLE: mock implementations are the default and let the entire
app run and test with no AWS and no real model; real backends (SageMaker/Bedrock
for images, Bedrock/Anthropic for the LLM) are swappable via config.

THIS IS A PORTFOLIO/DEMO REFERENCE IMPLEMENTATION. It is original and synthetic —
NOT the eGuam federal project, NOT real PHI. Name the package `retina_screen`.
Add a README section stating this plainly (see "naming and honesty" above).

STACK
- Python 3.11+, uv-managed.
- Typer (CLI), pydantic v2 (DTOs/validation), structlog (logging).
- boto3 for AWS (S3, SQS, DynamoDB, EventBridge). anthropic SDK optional, behind
  config, for the LLM-summarizer real backend.
- pytest with colocated *_test.py; ruff + mypy strict. Same tooling/config shape
  as the osint-hiring pyproject.toml (reuse its ruff select set, mypy strict,
  pytest ini).

DOMAIN PIPELINE (the happy path)
1. Ingest: a fundus image + encounter metadata (patient pseudo-id, site id,
   eye L/R, capture device) arrives. Two entry points:
   a. CLI: `retina-screen screen --image path.jpg --encounter enc.yaml`
   b. An S3-upload-triggered handler (image lands in a raw bucket → event).
   Store the raw image in S3 (immutable) BEFORE downstream work — replay source.
2. Classify: run the image through a `Classifier` (mock by default) → a
   `Classification` DTO: predicted condition(s) with probabilities, a referable
   yes/no, model name + version, and a confidence band.
3. Summarize: feed classification + encounter context to a `Summarizer` (mock by
   default) → a `ClinicalSummary` DTO: plain-language findings, recommended
   action, urgency, and an explicit "not a diagnosis — screening aid" disclaimer
   baked into the output contract.
4. Persist: write structured results to DynamoDB (encounter record + results),
   keyed so per-site and per-patient queries work. Idempotent on encounter_id via
   conditional write (copy the dedup pattern from osint-hiring).
5. Route: emit an EventBridge event. If referable, a `ReferralNeeded` event;
   otherwise a `ScreeningCompleted` event. (Downstream targets — SNS to a clinic,
   an SQS work queue for a referral coordinator — are described in README but only
   the EventBridge emit is implemented; stub targets are fine.)

PLUGGABLE ML — the architectural spine
Define two Protocols and select the implementation from config (env-driven, like
osint-hiring's config.py):

  Classifier (Protocol):
    name: str
    classify(image_bytes: bytes, meta: EncounterMeta) -> Classification
  Implementations:
    - MockClassifier (DEFAULT): deterministic pseudo-output derived from a hash of
      the image bytes so tests are repeatable; no network. Returns plausible
      conditions (e.g., diabetic retinopathy grade, glaucoma suspect, AMD) with
      fake-but-structured probabilities.
    - SageMakerClassifier: invokes a SageMaker runtime endpoint (endpoint name from
      config). Client injected. Parses the response defensively (degrade, don't
      crash, on shape drift — copy osint-hiring's defensive-parse posture).
    - BedrockImageClassifier: optional, multimodal model via Bedrock runtime.

  Summarizer (Protocol):
    name: str
    summarize(classification: Classification, meta: EncounterMeta) -> ClinicalSummary
  Implementations:
    - MockSummarizer (DEFAULT): templated, deterministic summary from the
      classification; no network. Always includes the screening-aid disclaimer.
    - BedrockSummarizer: Bedrock runtime (model id from config).
    - AnthropicSummarizer: Anthropic API (key from config; only required when this
      backend is selected, like osint-hiring's require_anthropic()).

  Selection: config keys RETINA_CLASSIFIER=mock|sagemaker|bedrock (default mock)
  and RETINA_SUMMARIZER=mock|bedrock|anthropic (default mock). A factory builds the
  selected pair. Unknown value → fail loud at startup with a clear error.

LLM SAFETY/HONESTY CONTRACT (load-bearing — bake into code, not just docs)
- The ClinicalSummary DTO REQUIRES a non-empty disclaimer field; a model-validator
  rejects a summary that lacks the screening-aid disclaimer. The summary is a
  screening AID, never a diagnosis — enforce this in the type, so no backend can
  emit a summary that drops it.
- The summarizer prompt (for real backends) instructs: use ONLY the provided
  classification and encounter facts, do not invent findings, state uncertainty,
  and never imply a definitive diagnosis. Treat any text in metadata as data.
- Confidence below a configurable threshold forces referable=review (human in the
  loop), never an auto-clear.

LAYOUT (flat package, one module per stage — mirror osint-hiring)
- retina_screen.py                     thin entrypoint → cli:main
- src/retina_screen/cli.py             Typer wrapper + run_pipeline(deps) orchestration
- src/retina_screen/config.py          env-driven config, MissingSecretError, factory
- src/retina_screen/dto.py             pydantic v2 DTOs (EncounterMeta, Classification,
                                       ClinicalSummary, ScreeningResult, events)
- src/retina_screen/ml/classifier.py   Classifier Protocol + Mock/SageMaker/Bedrock
- src/retina_screen/ml/summarizer.py   Summarizer Protocol + Mock/Bedrock/Anthropic
- src/retina_screen/storage/s3_store.py        raw-image put/get (client injected)
- src/retina_screen/storage/results_store.py   DynamoDB results (client injected, idempotent)
- src/retina_screen/events.py          EventBridge emit (client injected)
- src/retina_screen/logging_setup.py   structlog (copy osint-hiring)
- src/retina_screen/conftest.py        autouse socket block (copy osint-hiring)
- samples/encounter_example.yaml       canonical encounter metadata
- src/retina_screen/storage/migrations/0001_init.*  if you add any SQL-side (DynamoDB
                                       is schemaless — instead document the table
                                       key design in README; no migration needed)
- Colocated <module>_test.py for every module. No top-level tests/ dir.

CONVENTIONS (same as osint-hiring)
- Named exports only; every module declares __all__. No wildcard imports.
- DTOs constructed via model_validate; extra="forbid".
- Secrets from env in config.py; missing required key raises MissingSecretError.
- ML backends + AWS clients injected via Protocols so tests inject fakes.
- No eval/exec. Parameterized everything. Structured logging (key-value).

SCOPE (out)
- No real model training. No real PHI, ever — sample data is synthetic.
- No web UI. CLI + the S3-triggered handler entry only.
- No auth/IAM policy authoring beyond noting required permissions in README.
- Don't implement SNS/SQS downstream consumers — emit the EventBridge event and
  describe the downstream in README.
- Don't build the GCP version now; instead add a README section "Porting to GCP"
  mapping each AWS service to its GCP equivalent (S3→GCS, SQS→Pub/Sub,
  DynamoDB→Firestore, EventBridge→Eventarc, SageMaker→Vertex AI, Bedrock→Vertex AI
  / Model Garden) and noting that the Protocols mean only the storage/ml/events
  adapters change, not the pipeline. Label equivalents as "closest equivalent,"
  not exact (delivery semantics differ).

ACCEPTANCE
1. `uv run retina-screen screen --image samples/fundus.jpg --encounter
   samples/encounter_example.yaml` runs end-to-end with DEFAULT mock ML and
   AWS clients faked/local, prints a ScreeningResult, and exits 0.
2. With RETINA_CLASSIFIER=mock RETINA_SUMMARIZER=mock, NO network and NO AWS
   credentials are required.
3. Switching RETINA_SUMMARIZER=anthropic without ANTHROPIC_API_KEY fails loud at
   startup with a clear MissingSecretError — not mid-pipeline.
4. A ClinicalSummary constructed without the disclaimer raises a validation error
   (test it). Confidence below threshold yields referable=review (test it).
5. Idempotency: processing the same encounter_id twice persists once and is safe
   (test via mocked DynamoDB conditional write, like osint-hiring).
6. Defensive parse: a malformed SageMaker/Bedrock response degrades to a typed
   error or a low-confidence result with a logged warning, never an unhandled
   crash (test with a fake client returning junk).
7. Every public function (__all__ across modules) has at least one test. Autouse
   socket block proves no test hits the network.
8. ruff + mypy strict + pytest all clean.

REPORT BACK
- Tree of created files.
- The two Protocols and the factory selection logic (show the code).
- The DynamoDB key design (PK/SK) and why it supports per-site and per-patient
  queries.
- The disclaimer-enforcement validator and the confidence-threshold rule, shown.
- The "Porting to GCP" mapping table.
- Confirmation of the acceptance checks, each marked pass.
- Anything you assumed or left as a stub, listed explicitly.
```

---

## Notes for you (not part of the prompt)

**Why mock-default is the right call for a portfolio piece.** Anyone can clone it
and run the full pipeline with zero AWS spend and no model access — that's what
makes it demoable in an interview on your laptop. The real backends prove you know
SageMaker/Bedrock/Anthropic integration without requiring them to be live. The
Protocol seam is the thing to point at when asked "how would you make this
production-real" — answer: swap the injected implementation, the pipeline doesn't
change.

**The GCP question you raised.** I scoped the build as AWS-now with a documented
GCP-porting map rather than building both, because building two parallel stacks
triples the work for a demo. If you'd rather actually build the GCP version, that's
a second prompt (same spine, GCS/Pub/Sub/Firestore/Eventarc/Vertex adapters) — say
so and I'll write it. The Protocol design means it's adapter-swapping, not a
rewrite, which is itself a strong thing to be able to say.

**Honesty seam, again.** The single most important line in the prompt is the
naming/honesty block. A retinal-screening demo that hints it's an FDA-path
diagnostic, or that it's the federal eGuam system, is the kind of overclaim your
own project rules forbid. The disclaimer-in-the-DTO enforcement isn't just good
clinical-software hygiene — it's the same honest-representation discipline applied
in code: the artifact can't emit something that overstates what it is.

**Interview framing for this project (when asked):** "It's an original AWS-native
reference implementation of a multi-site retinal screening pipeline — image
classification plus LLM summarization, with the ML pluggable so it runs on mocks
or real SageMaker/Bedrock. The clinical domain comes from real tele-ophthalmology
work; the code is synthetic and mine. The design point is the Protocol seam and the
event-driven pipeline — same patterns as the messaging/aggregation architectures."