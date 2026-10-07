"""
Tests for the LLM Controller and JSON parsing logic.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from mrna_design.controller.llm import LLMController
from mrna_design.controller.llm_client import (
    ChatResponse,
    LLMError,
    LLMUnavailable,
    extract_json_from_response,
)
from mrna_design.models.candidate import Candidate
from mrna_design.models.diagnostics import IssueType, Region, RegionDiagnostic, Severity


def _reply(payload) -> ChatResponse:
    """Wrap a mock payload in the ChatResponse the client now returns."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return ChatResponse(text=text, model="mock", prompt_tokens=120, completion_tokens=40)


def test_extract_json_valid():
    text = json.dumps(
        {
            "expected_improvement": "Fixes stem",
            "edits": [
                {
                    "codon_index": 5,
                    "original_codon": "GCA",
                    "new_codon": "GCC",
                    "reason": "more GC",
                    "targeting_issue": "low_mfe",
                }
            ],
        }
    )
    parsed = extract_json_from_response(text)
    assert isinstance(parsed, dict)
    assert len(parsed["edits"]) == 1


def test_extract_json_with_markdown():
    text = """Here are your edits:
```json
{
    "edits": []
}
```
Good luck!"""
    parsed = extract_json_from_response(text)
    assert isinstance(parsed, dict)
    assert parsed["edits"] == []


def test_extract_json_invalid():
    with pytest.raises(LLMError):
        extract_json_from_response("I am an AI model. I cannot output JSON.")


@pytest.fixture
def mock_candidate():
    # ATGGCA encodes M-A; repeated 10x gives 60 nt CDS
    cds = "AUGGCA" * 10  # RNA alphabet
    protein = "MA" * 10  # protein for the CDS
    return Candidate.from_cds(cds=cds, protein=protein)


@pytest.fixture
def mock_diagnostics():
    return [
        RegionDiagnostic(
            region=Region.CDS,
            issue=IssueType.LOCAL_STABLE_STEM,
            severity=Severity.HIGH,
            metric=0.8,
            threshold=0.5,
            codon_start=0,
            codon_end=5,
            detail="Too much structure in window",
        )
    ]


@patch("mrna_design.controller.llm.chat_with_usage")
def test_llm_controller_success(mock_chat, mock_candidate, mock_diagnostics):
    mock_response = {
        "expected_improvement": "Reduced structure",
        "edits": [
            {
                "codon_index": 1,
                "original_codon": "GCA",
                "new_codon": "GCG",
                "reason": "Synonymous change",
                "targeting_issue": "low_mfe_window",
            }
        ],
    }
    mock_chat.return_value = _reply(mock_response)

    ctrl = LLMController()
    proposal = ctrl.propose_edits(
        mock_candidate, mock_diagnostics, history=[], max_edits=5, iteration=1
    )

    assert proposal.controller_type == "llm"
    assert len(proposal.edits) == 1
    edit = proposal.edits[0]
    assert edit.codon_index == 1
    assert edit.original_codon == "GCA"
    assert edit.new_codon == "GCG"
    assert edit.targeting_issue == "low_mfe_window"
    assert proposal.metadata["attempt"] == 1


@patch("mrna_design.controller.llm.chat_with_usage")
def test_llm_controller_all_resolved(mock_chat, mock_candidate):
    # LOW severity diagnostics (value=1) are below the >= 2 filter → no LLM call
    diagnostics = [
        RegionDiagnostic(
            region=Region.CDS,
            issue=IssueType.CPG_HOTSPOT,
            severity=Severity.LOW,
            metric=0.1,
            threshold=0.3,
        )
    ]

    ctrl = LLMController()
    proposal = ctrl.propose_edits(mock_candidate, diagnostics, history=[], max_edits=5, iteration=2)

    # Should not call chat because no severity >= 2 diagnostics
    mock_chat.assert_not_called()
    assert len(proposal.edits) == 0
    assert proposal.metadata["empty_reason"] == "no active diagnostics"


@patch("mrna_design.controller.llm.chat_with_usage")
def test_llm_controller_hallucination_recovery(mock_chat, mock_candidate, mock_diagnostics):
    # First attempt: returns invalid JSON format
    # Second attempt: returns valid JSON but invalid schema (non-RNA char)
    # Third attempt: valid

    mock_chat.side_effect = [
        _reply("not json"),
        _reply(
            {
                "edits": [
                    {"codon_index": 1, "original_codon": "GCA", "new_codon": "GCX", "reason": "bad"}
                ]
            }
        ),
        _reply(
            {
                "edits": [
                    {
                        "codon_index": 1,
                        "original_codon": "GCA",
                        "new_codon": "GCG",
                        "reason": "good",
                    }
                ]
            }
        ),
    ]

    ctrl = LLMController(max_retries=3)
    proposal = ctrl.propose_edits(
        mock_candidate, mock_diagnostics, history=[], max_edits=5, iteration=3
    )

    assert mock_chat.call_count == 3
    assert len(proposal.edits) == 1
    assert proposal.edits[0].new_codon == "GCG"
    assert proposal.metadata["attempt"] == 3


@patch("mrna_design.controller.llm.chat_with_usage")
def test_llm_controller_max_retries_exceeded(mock_chat, mock_candidate, mock_diagnostics):
    mock_chat.side_effect = [_reply("bad")] * 3

    ctrl = LLMController(max_retries=3)
    proposal = ctrl.propose_edits(
        mock_candidate, mock_diagnostics, history=[], max_edits=5, iteration=4
    )

    assert mock_chat.call_count == 3
    assert len(proposal.edits) == 0
    assert proposal.metadata["empty_reason"] == "max retries exceeded"


# ── Regression tests for the hardened LLM controller ──────────────────────────


@patch("mrna_design.controller.llm.chat_with_usage")
def test_llm_controller_records_token_usage(mock_chat, mock_candidate, mock_diagnostics):
    """
    The research question compares controllers under a shared budget, so the LLM
    controller must report what it actually consumed.
    """
    mock_chat.return_value = _reply(
        {"edits": [{"codon_index": 1, "original_codon": "GCA", "new_codon": "GCG", "reason": "ok"}]}
    )
    ctrl = LLMController()
    ctrl.propose_edits(mock_candidate, mock_diagnostics, history=[], max_edits=5, iteration=1)
    ctrl.propose_edits(mock_candidate, mock_diagnostics, history=[], max_edits=5, iteration=2)

    usage = ctrl.usage_summary()
    assert usage["api_calls"] == 2
    assert usage["prompt_tokens"] == 240
    assert usage["completion_tokens"] == 80
    assert usage["total_tokens"] == 320


@patch("mrna_design.controller.llm.chat_with_usage")
def test_llm_unavailable_degrades_instead_of_crashing(mock_chat, mock_candidate, mock_diagnostics):
    """
    A missing SDK or bad API key used to propagate out of propose_edits and abort
    the whole benchmark sweep. It must now degrade to an empty proposal.
    """
    mock_chat.side_effect = LLMUnavailable("No module named 'openai'")
    ctrl = LLMController(max_retries=3)

    proposal = ctrl.propose_edits(
        mock_candidate, mock_diagnostics, history=[], max_edits=5, iteration=1
    )
    assert proposal.edits == []
    assert ctrl.unavailable is True
    assert "openai" in ctrl.unavailable_reason

    # And it must not keep hammering a provider that cannot work.
    ctrl.propose_edits(mock_candidate, mock_diagnostics, history=[], max_edits=5, iteration=2)
    assert mock_chat.call_count == 1


@patch("mrna_design.controller.llm.chat_with_usage")
def test_llm_skips_edits_already_tried(mock_chat, mock_candidate, mock_diagnostics):
    """Loop detection: an edit already present in history is not re-proposed."""
    from mrna_design.models.edits import CodonEdit, EditProposal

    mock_chat.return_value = _reply(
        {
            "edits": [
                {"codon_index": 1, "original_codon": "GCA", "new_codon": "GCG", "reason": "repeat"}
            ]
        }
    )
    history = [
        EditProposal(
            edits=[
                CodonEdit(
                    codon_index=1, original_codon="GCA", new_codon="GCG", reason="already tried"
                )
            ],
            controller_type="llm",
            iteration=1,
        )
    ]
    ctrl = LLMController(max_retries=1, avoid_repeat_edits=True)
    proposal = ctrl.propose_edits(
        mock_candidate, mock_diagnostics, history=history, max_edits=5, iteration=2
    )
    assert proposal.edits == []


@patch("mrna_design.controller.llm.chat_with_usage")
def test_llm_counts_invalid_edits(mock_chat, mock_candidate, mock_diagnostics):
    """Schema-invalid edits are counted as a model-quality signal, not silently dropped."""
    mock_chat.return_value = _reply(
        {
            "edits": [
                {
                    "codon_index": 1,
                    "original_codon": "GCA",
                    "new_codon": "GCX",
                    "reason": "bad base",
                },
                {"codon_index": 2, "original_codon": "GCA", "new_codon": "GCG", "reason": "fine"},
            ]
        }
    )
    ctrl = LLMController(max_retries=1)
    proposal = ctrl.propose_edits(
        mock_candidate, mock_diagnostics, history=[], max_edits=5, iteration=1
    )
    assert len(proposal.edits) == 1
    assert ctrl.usage_summary()["invalid_edits_rejected"] == 1
