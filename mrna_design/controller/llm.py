"""
LLM Controller — uses large language models to propose targeted synonymous edits.
"""

from __future__ import annotations

import json
from typing import Any

from mrna_design.controller.base import BaseController
from mrna_design.controller.llm_client import chat, extract_json_from_response, LLMError
from mrna_design.logging_utils import get_logger
from mrna_design.models.candidate import Candidate
from mrna_design.models.diagnostics import RegionDiagnostic
from mrna_design.models.edits import CodonEdit, EditProposal
from mrna_design.validators.codon_table import aa_synonyms, STANDARD_CODE, HUMAN_FREQUENCIES

log = get_logger("controller.llm")


PROMPT_TEMPLATE = """You are an expert mRNA sequence designer optimizing a coding sequence (CDS).

Your goal is to propose synonymous codon mutations to resolve structural, safety, or immunogenicity issues.

# Input Information
CDS Length: {cds_length} nucleotides ({cds_codons} codons)

## Critical Diagnostics (Issues to Fix)
{diagnostics}

## Synonymous Codon Menu (Allowed Replacements)
For the problematic codons identified above, here are the allowed synonymous replacements and their human usage frequencies (higher is better):
{synonymous_menus}

# Instructions
1. Analyze the critical diagnostics.
2. Select up to {max_edits} specific codons to mutate.
3. For each selected codon, choose a NEW codon from the allowed Synonymous Codon Menu that might resolve the issue (e.g., choosing a different GC content, removing a motif, breaking a stem).
4. Do NOT mutate a codon to the exact same original codon.
5. The new codon MUST be synonymous (encode the same amino acid).

# Output Format
Output ONLY a JSON object with the following schema:
{{
  "expected_improvement": "A brief explanation of how these edits will help.",
  "edits": [
    {{
      "codon_index": 0, // The 0-based index of the codon to change
      "original_codon": "GCA",
      "new_codon": "GCC",
      "reason": "Increases GC content in a loose region.",
      "targeting_issue": "low_mfe_window"
    }}
  ]
}}
"""


class LLMController(BaseController):
    """
    Agentic controller that prompts an LLM to propose edits.
    """

    name: str = "llm"

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1, max_retries: int = 3):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries

    def _build_diagnostics_text(self, diagnostics: list[RegionDiagnostic]) -> str:
        if not diagnostics:
            return "No critical issues detected. The sequence is fully optimized."
        
        lines = []
        # Sort by severity descending
        sorted_diags = sorted(diagnostics, key=lambda d: d.severity.value, reverse=True)
        for d in sorted_diags[:10]:  # Cap at top 10 to fit context window
            lines.append(f"- Issue: {d.issue.value} in {d.region.value} (Severity: {d.severity.name})")
            lines.append(f"  Metric: {d.metric:.4f} (threshold: {d.threshold:.4f})")
            if d.detail:
                lines.append(f"  Detail: {d.detail}")
            if d.codon_start is not None:
                lines.append(f"  Affected Codons: {d.codon_start} to {d.codon_end}")
            if d.suggestion:
                lines.append(f"  Suggestion: {d.suggestion}")
            lines.append("")
        return "\n".join(lines)

    def _build_synonymous_menus(self, candidate: Candidate, diagnostics: list[RegionDiagnostic]) -> str:
        # Collect codon indices that are problematic
        target_indices = set()
        for d in diagnostics:
            if d.codon_start is not None and d.codon_end is not None:
                for i in range(d.codon_start, d.codon_end + 1):
                    target_indices.add(i)
        
        lines = []
        cds = candidate.cds
        
        for idx in sorted(target_indices):
            if idx * 3 + 3 > len(cds):
                continue
                
            codon = cds[idx * 3 : idx * 3 + 3]
            aa = STANDARD_CODE.get(codon, "*")
            syns = aa_synonyms(aa)
            
            syn_info = []
            for s in syns:
                freq = HUMAN_FREQUENCIES.get(s, 0.0)
                mark = "(CURRENT)" if s == codon else ""
                syn_info.append(f"{s}: {freq:.2f}{mark}")
                
            lines.append(f"Codon {idx} (AA: {aa}): " + ", ".join(syn_info))
            
        return "\n".join(lines)

    def propose_edits(
        self,
        candidate: Candidate,
        diagnostics: list[RegionDiagnostic],
        history: list[EditProposal],
        max_edits: int = 5,
        iteration: int = 0,
    ) -> EditProposal:
        
        # Filter to high/medium severity diagnostics
        active_diags = [d for d in diagnostics if d.severity.value >= 2]
        if not active_diags:
            log.event("no_active_diagnostics", iteration=iteration)
            return EditProposal(
                edits=[], 
                controller_type="llm", 
                iteration=iteration,
                metadata={"reason": "No active diagnostics"}
            )
            
        cds_len = len(candidate.cds)
        prompt = PROMPT_TEMPLATE.format(
            cds_length=cds_len,
            cds_codons=cds_len // 3,
            diagnostics=self._build_diagnostics_text(active_diags),
            synonymous_menus=self._build_synonymous_menus(candidate, active_diags),
            max_edits=max_edits,
        )
        
        # We loop internally for parsing retries (llm_client handles network retries)
        for attempt in range(self.max_retries):
            try:
                response_text = chat(
                    prompt=prompt,
                    model=self.model_name,
                    temperature=self.temperature,
                    json_mode=True
                )
                
                parsed = extract_json_from_response(response_text)
                
                # Validate the parsed JSON against EditProposal fields
                edits = []
                for e_dict in parsed.get("edits", []):
                    try:
                        edits.append(CodonEdit(**e_dict))
                    except Exception as e:
                        log.warn("invalid_codon_edit_from_llm", error=str(e), data=str(e_dict))
                        continue
                        
                if not edits:
                    raise ValueError("LLM returned no valid edits.")
                    
                return EditProposal(
                    edits=edits[:max_edits],
                    expected_improvement=parsed.get("expected_improvement", ""),
                    controller_type="llm",
                    targeting_diagnostics=list({e.targeting_issue for e in edits if e.targeting_issue}),
                    iteration=iteration,
                    metadata={"model": self.model_name, "attempt": attempt + 1}
                )
                
            except (LLMError, ValueError) as e:
                log.warn("llm_parse_failed", error=str(e), attempt=attempt+1)
                
        # If all retries fail, return an empty proposal
        log.error("llm_max_retries_exceeded", iteration=iteration)
        return EditProposal(
            edits=[], 
            controller_type="llm", 
            iteration=iteration,
            metadata={"error": "Max retries exceeded"}
        )
