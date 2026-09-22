"""
CodonEdit, EditProposal, EditRecord — the edit grammar used by all controllers.

Design invariant
----------------
The controller (rule-based or LLM) never writes nucleotide sequences.
It outputs CodonEdit objects that reference a codon position and a replacement
codon chosen from the synonymous-codon menu.  The CodonApplicator validates
and applies them; the Candidate's lineage records the accepted EditRecords.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class CodonEdit(BaseModel):
    """A single proposed synonymous codon substitution."""

    codon_index: int = Field(
        ge=0,
        description="0-based index of the codon in the CDS codon list.",
    )
    original_codon: str = Field(
        min_length=3, max_length=3,
        description="The codon currently at codon_index (RNA alphabet, uppercase).",
    )
    new_codon: str = Field(
        min_length=3, max_length=3,
        description="The replacement codon (RNA alphabet, uppercase, must be synonymous).",
    )
    reason: str = Field(
        description="Plain-English reason for this specific edit.",
    )
    targeting_issue: str | None = Field(
        default=None,
        description="IssueType string of the RegionDiagnostic this edit targets.",
    )

    @field_validator("original_codon", "new_codon", mode="before")
    @classmethod
    def _normalise_codon(cls, v: Any) -> str:
        v = str(v).strip().upper().replace("T", "U")
        if len(v) != 3:
            raise ValueError(f"Codon must be exactly 3 nt, got '{v}'")
        if not all(c in "ACGU" for c in v):
            raise ValueError(f"Codon contains non-RNA characters: '{v}'")
        return v

    @model_validator(mode="after")
    def _not_identity(self) -> "CodonEdit":
        if self.original_codon == self.new_codon:
            raise ValueError(
                f"original_codon and new_codon are identical ({self.original_codon}). "
                "Identity edits are not allowed."
            )
        return self


class ValidationStatus(str):
    """Status codes returned by the codon applicator."""
    ACCEPTED = "accepted"
    REJECTED_NOT_SYNONYMOUS = "rejected_not_synonymous"
    REJECTED_PREMATURE_STOP = "rejected_premature_stop"
    REJECTED_GC_BOUNDS = "rejected_gc_bounds"
    REJECTED_RESTRICTION_SITE = "rejected_restriction_site"
    REJECTED_WRONG_ORIGINAL = "rejected_wrong_original"
    REJECTED_INDEX_OOB = "rejected_index_out_of_bounds"


class EditRecord(BaseModel):
    """A CodonEdit that has been validated and applied (or rejected)."""

    edit: CodonEdit
    status: str = Field(description="ValidationStatus string.")
    iteration: int = Field(ge=0, description="Optimisation iteration when applied.")
    controller_type: Literal["rule_based", "llm", "ga", "random"] = "rule_based"
    delta_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Score deltas (objective → new - old) for accepted edits.",
    )
    applied_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp.",
    )

    @property
    def accepted(self) -> bool:
        return self.status == ValidationStatus.ACCEPTED


class EditProposal(BaseModel):
    """
    A set of edits proposed by a controller for one candidate in one iteration.

    The controller populates this; the CodonApplicator validates each edit
    and builds a new Candidate with the accepted subset.
    """

    edits: list[CodonEdit] = Field(
        default_factory=list,
        description=(
            "Ordered list of proposed codon edits (applied left-to-right). "
            "May be empty when no edits are warranted."
        ),
    )
    expected_improvement: str = Field(
        default="",
        description="Controller's natural-language prediction of what should improve.",
    )
    controller_type: Literal["rule_based", "llm", "ga", "random"] = "rule_based"
    targeting_diagnostics: list[str] = Field(
        default_factory=list,
        description="IssueType strings of diagnostics this proposal is targeting.",
    )
    iteration: int = Field(ge=0, default=0)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Controller-specific metadata (e.g. LLM model, token usage).",
    )

    def summary(self) -> str:
        lines = [
            f"EditProposal ({self.controller_type}, iter={self.iteration}, "
            f"{len(self.edits)} edits)"
        ]
        for e in self.edits:
            lines.append(
                f"  codon[{e.codon_index}]: {e.original_codon} → {e.new_codon}"
                f"  [{e.targeting_issue or '—'}]  {e.reason}"
            )
        return "\n".join(lines)
