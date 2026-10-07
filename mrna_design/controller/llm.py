"""
LLM Controller — uses large language models to propose targeted synonymous edits.
"""

from __future__ import annotations

from mrna_design.controller.base import BaseController
from mrna_design.controller.llm_client import (
    LLMError,
    LLMUnavailable,
    chat_with_usage,
    extract_json_from_response,
)
from mrna_design.logging_utils import get_logger
from mrna_design.models.candidate import Candidate
from mrna_design.models.diagnostics import RegionDiagnostic
from mrna_design.models.edits import CodonEdit, EditProposal
from mrna_design.validators.codon_table import HUMAN_FREQUENCIES, STANDARD_CODE, aa_synonyms

log = get_logger("controller.llm")


PROMPT_TEMPLATE = """You are an expert mRNA sequence designer optimizing a coding sequence (CDS).

Your goal is to propose synonymous codon mutations that resolve structural,
safety, or immunogenicity issues.

# Input Information
CDS Length: {cds_length} nucleotides ({cds_codons} codons)

## Critical Diagnostics (Issues to Fix)
{diagnostics}

## Synonymous Codon Menu (Allowed Replacements)
For the problematic codons identified above, here are the allowed synonymous
replacements and their human usage frequencies (higher is better):
{synonymous_menus}

# Instructions
1. Analyze the critical diagnostics.
2. Select up to {max_edits} specific codons to mutate.
3. For each selected codon, choose a NEW codon from the allowed Synonymous Codon
   Menu that might resolve the issue (e.g. different GC content, removing a
   motif, breaking a stem).
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

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        max_retries: int = 3,
        avoid_repeat_edits: bool = True,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.avoid_repeat_edits = avoid_repeat_edits

        # Run-level accounting. The research question is about budget, so an LLM
        # controller has to report what it actually consumed.
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_calls = 0
        self.parse_failures = 0
        self.invalid_edits = 0
        self.unavailable = False
        self.unavailable_reason: str | None = None

    def reset(self) -> None:
        """Clear per-run accounting between benchmark runs."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_calls = 0
        self.parse_failures = 0
        self.invalid_edits = 0
        self.unavailable = False
        self.unavailable_reason = None

    def usage_summary(self) -> dict:
        """Token and failure accounting, written into the run manifest."""
        return {
            "model": self.model_name,
            "api_calls": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "parse_failures": self.parse_failures,
            "invalid_edits_rejected": self.invalid_edits,
            "unavailable": self.unavailable,
            "unavailable_reason": self.unavailable_reason,
        }

    @staticmethod
    def _recent_edit_keys(history: list[EditProposal]) -> set[tuple[int, str]]:
        """(codon_index, new_codon) pairs already tried, for loop detection."""
        return {(e.codon_index, e.new_codon) for proposal in history for e in proposal.edits}

    def _build_diagnostics_text(self, diagnostics: list[RegionDiagnostic]) -> str:
        if not diagnostics:
            return "No critical issues detected. The sequence is fully optimized."

        lines = []
        # Sort by severity descending
        sorted_diags = sorted(diagnostics, key=lambda d: d.severity.value, reverse=True)
        for d in sorted_diags[:10]:  # Cap at top 10 to fit context window
            lines.append(
                f"- Issue: {d.issue.value} in {d.region.value} (Severity: {d.severity.name})"
            )
            lines.append(f"  Metric: {d.metric:.4f} (threshold: {d.threshold:.4f})")
            if d.detail:
                lines.append(f"  Detail: {d.detail}")
            if d.codon_start is not None:
                lines.append(f"  Affected Codons: {d.codon_start} to {d.codon_end}")
            if d.suggestion:
                lines.append(f"  Suggestion: {d.suggestion}")
            lines.append("")
        return "\n".join(lines)

    def _build_synonymous_menus(
        self, candidate: Candidate, diagnostics: list[RegionDiagnostic]
    ) -> str:
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
            return self._empty(iteration, reason="no active diagnostics")

        cds_len = len(candidate.cds)
        prompt = PROMPT_TEMPLATE.format(
            cds_length=cds_len,
            cds_codons=cds_len // 3,
            diagnostics=self._build_diagnostics_text(active_diags),
            synonymous_menus=self._build_synonymous_menus(candidate, active_diags),
            max_edits=max_edits,
        )

        # Once the provider is known to be unusable, stop calling it. Retrying a
        # missing SDK or a bad API key for every candidate wastes wall-clock time
        # and floods the log without any chance of succeeding.
        if self.unavailable:
            return self._empty(iteration, reason=self.unavailable_reason or "unavailable")

        already_tried = self._recent_edit_keys(history) if self.avoid_repeat_edits else set()

        # Internal loop handles *parsing* retries; llm_client handles network retries.
        for attempt in range(self.max_retries):
            try:
                response = chat_with_usage(
                    prompt=prompt,
                    model=self.model_name,
                    temperature=self.temperature,
                    json_mode=True,
                )
                self.api_calls += 1
                self.prompt_tokens += response.prompt_tokens
                self.completion_tokens += response.completion_tokens

                parsed = extract_json_from_response(response.text)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")

                edits = []
                for e_dict in parsed.get("edits", []):
                    try:
                        edit = CodonEdit(**e_dict)
                    except Exception as e:
                        # Pydantic rejected it (non-RNA characters, identity edit,
                        # wrong length). The applicator would reject it too; we
                        # drop it here and count it as a model-quality signal.
                        self.invalid_edits += 1
                        log.warn("invalid_codon_edit_from_llm", error=str(e), data=str(e_dict))
                        continue
                    if (edit.codon_index, edit.new_codon) in already_tried:
                        # Loop detection: the controller is re-proposing an edit
                        # that a previous iteration already made or had rejected.
                        log.debug("llm_repeat_edit_skipped", codon_index=edit.codon_index)
                        continue
                    edits.append(edit)

                if not edits:
                    raise ValueError("LLM returned no usable edits.")

                return EditProposal(
                    edits=edits[:max_edits],
                    expected_improvement=parsed.get("expected_improvement", ""),
                    controller_type="llm",
                    targeting_diagnostics=list(
                        {e.targeting_issue for e in edits if e.targeting_issue}
                    ),
                    iteration=iteration,
                    metadata={
                        "model": self.model_name,
                        "attempt": attempt + 1,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                    },
                )

            except LLMUnavailable as e:
                # Permanent for this run: record it once and degrade to no-op.
                self.unavailable = True
                self.unavailable_reason = str(e)
                log.error("llm_unavailable", error=str(e), iteration=iteration)
                return self._empty(iteration, reason=str(e))

            except (LLMError, ValueError) as e:
                self.parse_failures += 1
                log.warn("llm_parse_failed", error=str(e), attempt=attempt + 1)

            except Exception as e:  # noqa: BLE001
                # Anything else (transport, provider quirk, malformed SDK object)
                # must not abort a long benchmark sweep.
                self.parse_failures += 1
                log.error(
                    "llm_call_failed",
                    error=f"{type(e).__name__}: {e}",
                    attempt=attempt + 1,
                    iteration=iteration,
                )

        log.error("llm_max_retries_exceeded", iteration=iteration)
        return self._empty(iteration, reason="max retries exceeded")

    def _empty(self, iteration: int, reason: str) -> EditProposal:
        """An explicit no-op proposal. The loop applies the run's uniform policy."""
        return EditProposal(
            edits=[],
            controller_type="llm",
            iteration=iteration,
            metadata={"model": self.model_name, "empty_reason": reason},
        )
