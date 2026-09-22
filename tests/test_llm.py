"""
Tests for the LLM Controller and JSON parsing logic.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from mrna_design.controller.llm import LLMController
from mrna_design.controller.llm_client import extract_json_from_response, LLMError
from mrna_design.models.candidate import Candidate
from mrna_design.models.diagnostics import RegionDiagnostic, IssueType, Region, Severity


def test_extract_json_valid():
    text = '{"expected_improvement": "Fixes stem", "edits": [{"codon_index": 5, "original_codon": "GCA", "new_codon": "GCC", "reason": "more GC", "targeting_issue": "low_mfe"}]}'
    parsed = extract_json_from_response(text)
    assert isinstance(parsed, dict)
    assert len(parsed["edits"]) == 1


def test_extract_json_with_markdown():
    text = '''Here are your edits:
```json
{
    "edits": []
}
```
Good luck!'''
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


@patch("mrna_design.controller.llm.chat")
def test_llm_controller_success(mock_chat, mock_candidate, mock_diagnostics):
    mock_response = {
        "expected_improvement": "Reduced structure",
        "edits": [
            {
                "codon_index": 1,
                "original_codon": "GCA",
                "new_codon": "GCG",
                "reason": "Synonymous change",
                "targeting_issue": "low_mfe_window"
            }
        ]
    }
    mock_chat.return_value = json.dumps(mock_response)

    ctrl = LLMController()
    proposal = ctrl.propose_edits(mock_candidate, mock_diagnostics, history=[], max_edits=5, iteration=1)

    assert proposal.controller_type == "llm"
    assert len(proposal.edits) == 1
    edit = proposal.edits[0]
    assert edit.codon_index == 1
    assert edit.original_codon == "GCA"
    assert edit.new_codon == "GCG"
    assert edit.targeting_issue == "low_mfe_window"
    assert proposal.metadata["attempt"] == 1


@patch("mrna_design.controller.llm.chat")
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
    assert proposal.metadata["reason"] == "No active diagnostics"



@patch("mrna_design.controller.llm.chat")
def test_llm_controller_hallucination_recovery(mock_chat, mock_candidate, mock_diagnostics):
    # First attempt: returns invalid JSON format
    # Second attempt: returns valid JSON but invalid schema (non-RNA char)
    # Third attempt: valid
    
    mock_chat.side_effect = [
        "not json",
        json.dumps({"edits": [{"codon_index": 1, "original_codon": "GCA", "new_codon": "GCX", "reason": "bad"}]}),
        json.dumps({"edits": [{"codon_index": 1, "original_codon": "GCA", "new_codon": "GCG", "reason": "good"}]}),
    ]
    
    ctrl = LLMController(max_retries=3)
    proposal = ctrl.propose_edits(mock_candidate, mock_diagnostics, history=[], max_edits=5, iteration=3)
    
    assert mock_chat.call_count == 3
    assert len(proposal.edits) == 1
    assert proposal.edits[0].new_codon == "GCG"
    assert proposal.metadata["attempt"] == 3


@patch("mrna_design.controller.llm.chat")
def test_llm_controller_max_retries_exceeded(mock_chat, mock_candidate, mock_diagnostics):
    mock_chat.side_effect = ["bad"] * 3
    
    ctrl = LLMController(max_retries=3)
    proposal = ctrl.propose_edits(mock_candidate, mock_diagnostics, history=[], max_edits=5, iteration=4)
    
    assert mock_chat.call_count == 3
    assert len(proposal.edits) == 0
    assert proposal.metadata["error"] == "Max retries exceeded"
