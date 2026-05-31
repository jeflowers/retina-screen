"""Tests for the Summarizer Protocol and MockSummarizer.

The single most important property: ClinicalSummary's disclaimer validator
must NEVER trip — every code path through MockSummarizer produces a valid
summary. This is the implementation-side complement to the type-level
enforcement in dto.py.
"""

import io
import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from retina_screen.dto import (
    Classification,
    ClinicalSummary,
    ConditionProbability,
    EncounterMeta,
    Referable,
)
from retina_screen.ml.summarizer import (
    BACKEND_DISCLAIMER,
    MOCK_DISCLAIMER,
    AnthropicSummarizer,
    BedrockSummarizer,
    MockSummarizer,
    Summarizer,
)


def _meta(eye: str = "L") -> EncounterMeta:
    return EncounterMeta(
        encounter_id="enc_001",
        patient_pseudo_id="pt_001",
        site_id="site_A",
        eye=eye,  # type: ignore[arg-type]
        capture_device="topcon_nw400",
        captured_at=datetime(2026, 5, 28, 14, 30, tzinfo=UTC),
    )


def _classification(
    referable: Referable, primary_name: str = "diabetic_retinopathy_grade_2"
) -> Classification:
    return Classification(
        conditions=(
            ConditionProbability(name=primary_name, probability=0.78),
            ConditionProbability(name="normal", probability=0.22),
        ),
        referable=referable,
        model_name="mock-classifier",
        model_version="0.1.0",
        confidence=0.85,
    )


def test_mock_summarizer_implements_protocol() -> None:
    summarizer = MockSummarizer()
    assert isinstance(summarizer, Summarizer)
    assert summarizer.name == "mock-summarizer"


def test_mock_summarizer_always_includes_disclaimer() -> None:
    """Every branch (yes/no/review) must produce a non-empty disclaimer."""
    summarizer = MockSummarizer()
    meta = _meta()

    for referable in ("yes", "no", "review"):
        summary = summarizer.summarize(_classification(referable), meta)
        assert summary.disclaimer == MOCK_DISCLAIMER
        assert "screening aid" in summary.disclaimer


def test_mock_summarizer_is_deterministic() -> None:
    summarizer = MockSummarizer()
    meta = _meta()
    classification = _classification("yes")

    assert summarizer.summarize(classification, meta) == summarizer.summarize(classification, meta)


def test_referable_yes_yields_soon_urgency_and_refer_action() -> None:
    summary = MockSummarizer().summarize(_classification("yes"), _meta())
    assert summary.urgency == "soon"
    assert "refer" in summary.recommended_action.lower()


def test_referable_review_yields_grader_queue_action() -> None:
    summary = MockSummarizer().summarize(_classification("review"), _meta())
    assert summary.urgency == "soon"
    assert (
        "grader" in summary.recommended_action.lower()
        or "review" in summary.recommended_action.lower()
    )


def test_referable_no_yields_routine_urgency() -> None:
    summary = MockSummarizer().summarize(_classification("no", primary_name="normal"), _meta())
    assert summary.urgency == "routine"
    assert "routine" in summary.recommended_action.lower()


def test_summary_mentions_eye_side() -> None:
    """Findings text should reflect which eye was examined (left/right)."""
    s_left = MockSummarizer().summarize(_classification("yes"), _meta(eye="L"))
    s_right = MockSummarizer().summarize(_classification("yes"), _meta(eye="R"))
    assert "left" in s_left.findings.lower()
    assert "right" in s_right.findings.lower()


def test_summary_returns_clinical_summary_type() -> None:
    summary = MockSummarizer().summarize(_classification("yes"), _meta())
    assert isinstance(summary, ClinicalSummary)


# ─── BedrockSummarizer ───────────────────────────────────────────────────────


def _bedrock_text_response(model_text: str) -> dict[str, object]:
    """Wrap model text as a Bedrock invoke_model response."""
    envelope = {"content": [{"type": "text", "text": model_text}]}
    return {"body": io.BytesIO(json.dumps(envelope).encode())}


def _good_summary_payload() -> str:
    return json.dumps(
        {
            "findings": "Probable referable disease in the left eye.",
            "recommended_action": "Refer to ophthalmology.",
            "urgency": "soon",
        }
    )


def test_bedrock_summarizer_happy_path() -> None:
    client = MagicMock()
    client.invoke_model.return_value = _bedrock_text_response(_good_summary_payload())

    summarizer = BedrockSummarizer(client=client, model_id="anthropic.claude-3")
    summary = summarizer.summarize(_classification("yes"), _meta())

    assert summary.urgency == "soon"
    # Disclaimer is set by code (BACKEND_DISCLAIMER), not by the model output.
    assert summary.disclaimer == BACKEND_DISCLAIMER
    call = client.invoke_model.call_args.kwargs
    assert call["modelId"] == "anthropic.claude-3"
    body = json.loads(call["body"])
    assert body["system"]  # safety prompt present
    assert "never imply a definitive diagnosis" in body["system"].lower()


def test_bedrock_summarizer_implements_protocol() -> None:
    assert isinstance(BedrockSummarizer(client=MagicMock(), model_id="m"), Summarizer)


def test_bedrock_summarizer_disclaimer_is_not_taken_from_model() -> None:
    """Belt-and-suspenders: the model could not omit the disclaimer if it tried."""
    client = MagicMock()
    # Model returns a summary that omits a 'disclaimer' field entirely.
    payload = json.dumps(
        {
            "findings": "Findings text.",
            "recommended_action": "Action text.",
            "urgency": "routine",
        }
    )
    client.invoke_model.return_value = _bedrock_text_response(payload)

    summary = BedrockSummarizer(client=client, model_id="m").summarize(
        _classification("no"), _meta()
    )

    assert summary.disclaimer == BACKEND_DISCLAIMER
    assert (
        "screening aid" in summary.disclaimer.lower() or "specialist" in summary.disclaimer.lower()
    )


def test_bedrock_summarizer_malformed_json_returns_fallback() -> None:
    """Acceptance #6 (summarizer side): junk text → fallback summary, no crash."""
    client = MagicMock()
    client.invoke_model.return_value = _bedrock_text_response("Sorry, I cannot summarize.")

    summary = BedrockSummarizer(client=client, model_id="m").summarize(
        _classification("yes"), _meta()
    )
    assert summary.disclaimer == BACKEND_DISCLAIMER  # still valid
    assert "failed" in summary.findings.lower() or "review" in summary.findings.lower()


def test_bedrock_summarizer_invalid_urgency_returns_fallback() -> None:
    payload = json.dumps({"findings": "f", "recommended_action": "a", "urgency": "whenever"})
    client = MagicMock()
    client.invoke_model.return_value = _bedrock_text_response(payload)

    summary = BedrockSummarizer(client=client, model_id="m").summarize(
        _classification("yes"), _meta()
    )
    assert "failed" in summary.findings.lower() or "review" in summary.findings.lower()


def test_bedrock_summarizer_client_error_returns_fallback() -> None:
    client = MagicMock()
    client.invoke_model.side_effect = ClientError(
        error_response={"Error": {"Code": "ThrottlingException", "Message": "slow"}},
        operation_name="InvokeModel",
    )

    summary = BedrockSummarizer(client=client, model_id="m").summarize(
        _classification("yes"), _meta()
    )
    assert summary.disclaimer == BACKEND_DISCLAIMER


# ─── AnthropicSummarizer ─────────────────────────────────────────────────────


def _anthropic_response(text: str) -> object:
    """Build a fake anthropic response object with content[0].text == text."""
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def test_anthropic_summarizer_happy_path() -> None:
    client = MagicMock()
    client.messages.create.return_value = _anthropic_response(_good_summary_payload())

    summarizer = AnthropicSummarizer(client=client, model="claude-opus-4-7")
    summary = summarizer.summarize(_classification("yes"), _meta())

    assert summary.urgency == "soon"
    assert summary.disclaimer == BACKEND_DISCLAIMER
    call = client.messages.create.call_args.kwargs
    assert call["model"] == "claude-opus-4-7"
    assert "never imply a definitive diagnosis" in call["system"].lower()


def test_anthropic_summarizer_implements_protocol() -> None:
    assert isinstance(AnthropicSummarizer(client=MagicMock(), model="claude-opus-4-7"), Summarizer)


def test_anthropic_summarizer_malformed_text_returns_fallback() -> None:
    """Acceptance #6 (anthropic side)."""
    client = MagicMock()
    client.messages.create.return_value = _anthropic_response("Sorry, no JSON for you.")

    summary = AnthropicSummarizer(client=client, model="claude-opus-4-7").summarize(
        _classification("yes"), _meta()
    )
    assert summary.disclaimer == BACKEND_DISCLAIMER
    assert "failed" in summary.findings.lower() or "review" in summary.findings.lower()


def test_anthropic_summarizer_sdk_exception_returns_fallback() -> None:
    """Any anthropic SDK exception (auth, rate limit, etc.) must fall back."""

    class _FakeAuthError(Exception):
        pass

    client = MagicMock()
    client.messages.create.side_effect = _FakeAuthError("invalid api key")

    summary = AnthropicSummarizer(client=client, model="claude-opus-4-7").summarize(
        _classification("yes"), _meta()
    )
    assert summary.disclaimer == BACKEND_DISCLAIMER


def test_anthropic_summarizer_disclaimer_set_by_code_not_model() -> None:
    """Even if the model output is valid JSON without a disclaimer, code sets it."""
    payload = json.dumps({"findings": "f", "recommended_action": "a", "urgency": "routine"})
    client = MagicMock()
    client.messages.create.return_value = _anthropic_response(payload)

    summary = AnthropicSummarizer(client=client, model="claude-opus-4-7").summarize(
        _classification("no"), _meta()
    )
    assert summary.disclaimer == BACKEND_DISCLAIMER
