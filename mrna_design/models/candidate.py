"""
Candidate — the central data object flowing through the pipeline.

A Candidate represents one mRNA sequence with:
  - The full sequence (5'UTR + CDS + 3'UTR, RNA alphabet)
  - The protected protein sequence (ground truth; never mutated)
  - Coordinates of each region
  - Computed ObjectiveScores
  - Per-region RegionDiagnostics from agents
  - Full edit lineage (EditRecord list)

Immutability convention
-----------------------
Candidates are treated as immutable values.  The CodonApplicator produces a
*new* Candidate (model_copy + updates) rather than mutating in place.  This
makes lineage tracking trivial and supports population-level Pareto comparison.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from pydantic import BaseModel, Field, model_validator

from mrna_design.models.diagnostics import RegionDiagnostic
from mrna_design.models.edits import EditRecord
from mrna_design.models.objectives import ObjectiveScores


class Candidate(BaseModel):
    """A single mRNA sequence candidate with all associated metadata."""

    # ── Sequence & coordinates ────────────────────────────────────────────────
    sequence: str = Field(
        description="Full mRNA sequence in RNA alphabet (A/U/G/C, uppercase). "
        "Includes 5'UTR + CDS + 3'UTR if present.",
    )
    protein: str = Field(
        description="Protected amino acid sequence (single-letter code, uppercase). "
        "Used as ground truth for protein-identity validation.",
    )
    cds_start: int = Field(
        ge=0,
        description="0-based index of the first nucleotide of the CDS (the A of AUG).",
    )
    cds_end: int = Field(
        description="0-based index one past the last nucleotide of the CDS stop codon.",
    )
    utr5: str = Field(
        default="",
        description="5'UTR sequence (may be empty).",
    )
    utr3: str = Field(
        default="",
        description="3'UTR sequence (may be empty).",
    )

    # ── Scores & diagnostics ──────────────────────────────────────────────────
    scores: ObjectiveScores = Field(default_factory=ObjectiveScores)
    diagnostics: list[RegionDiagnostic] = Field(default_factory=list)

    # ── Lineage ───────────────────────────────────────────────────────────────
    lineage: list[EditRecord] = Field(
        default_factory=list,
        description="All accepted and rejected edits applied to this candidate, oldest first.",
    )
    parent_id: str | None = Field(
        default=None,
        description="sequence_id of the parent candidate, or None for seeds.",
    )
    seed_strategy: str | None = Field(
        default=None,
        description="Name of the seeding strategy that created this candidate (e.g. 'cai_max').",
    )
    iteration: int = Field(
        default=0,
        description="Optimisation iteration at which this candidate was created.",
    )
    created_at: float = Field(default_factory=time.time)

    # ── Derived / cached ──────────────────────────────────────────────────────
    _sequence_id: str | None = None  # lazy-computed SHA-256 prefix

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _validate_structure(self) -> Candidate:
        # Normalise sequence to uppercase RNA
        seq = self.sequence.upper().replace("T", "U")
        object.__setattr__(self, "sequence", seq)

        if self.cds_end <= self.cds_start:
            raise ValueError(f"cds_end ({self.cds_end}) must be > cds_start ({self.cds_start})")
        cds_len = self.cds_end - self.cds_start
        if cds_len % 3 != 0:
            raise ValueError(f"CDS length ({cds_len}) is not a multiple of 3.")
        if self.cds_end > len(seq):
            raise ValueError(f"cds_end ({self.cds_end}) exceeds sequence length ({len(seq)}).")
        return self

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def sequence_id(self) -> str:
        """Short SHA-256 hex digest of the sequence for deduplication."""
        if self._sequence_id is None:
            digest = hashlib.sha256(self.sequence.encode()).hexdigest()[:16]
            object.__setattr__(self, "_sequence_id", digest)
        return self._sequence_id  # type: ignore[return-value]

    @property
    def cds(self) -> str:
        """CDS subsequence."""
        return self.sequence[self.cds_start : self.cds_end]

    @property
    def codons(self) -> list[str]:
        """List of codons in the CDS (length = len(protein) + 1 for stop)."""
        c = self.cds
        return [c[i : i + 3] for i in range(0, len(c), 3)]

    @property
    def n_codons(self) -> int:
        """Number of codons including stop."""
        return (self.cds_end - self.cds_start) // 3

    @property
    def accepted_edits(self) -> list[EditRecord]:
        return [r for r in self.lineage if r.accepted]

    @property
    def rejected_edits(self) -> list[EditRecord]:
        return [r for r in self.lineage if not r.accepted]

    def high_severity_diagnostics(self, min_severity: int = 2) -> list[RegionDiagnostic]:
        return [d for d in self.diagnostics if d.severity.value >= min_severity]

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_cds(
        cls,
        cds: str,
        protein: str,
        utr5: str = "",
        utr3: str = "",
        seed_strategy: str | None = None,
        **kwargs: Any,
    ) -> Candidate:
        """Construct a Candidate from a bare CDS string + optional UTRs."""
        cds = cds.upper().replace("T", "U")
        utr5 = utr5.upper().replace("T", "U")
        utr3 = utr3.upper().replace("T", "U")
        sequence = utr5 + cds + utr3
        return cls(
            sequence=sequence,
            protein=protein,
            cds_start=len(utr5),
            cds_end=len(utr5) + len(cds),
            utr5=utr5,
            utr3=utr3,
            seed_strategy=seed_strategy,
            **kwargs,
        )

    def with_new_cds(
        self,
        new_cds: str,
        iteration: int = 0,
        lineage_append: list[EditRecord] | None = None,
    ) -> Candidate:
        """Return a new Candidate with the CDS replaced, preserving UTRs and lineage."""
        new_cds = new_cds.upper().replace("T", "U")
        new_seq = self.utr5 + new_cds + self.utr3
        new_lineage = list(self.lineage) + (lineage_append or [])
        return Candidate(
            sequence=new_seq,
            protein=self.protein,
            cds_start=self.cds_start,
            cds_end=self.cds_start + len(new_cds),
            utr5=self.utr5,
            utr3=self.utr3,
            seed_strategy=self.seed_strategy,
            parent_id=self.sequence_id,
            iteration=iteration,
            lineage=new_lineage,
            # Scores and diagnostics are cleared — must be recomputed
            scores=ObjectiveScores(),
            diagnostics=[],
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_log_dict(self) -> dict[str, Any]:
        """Compact dict for JSON event logging (no full sequence)."""
        return {
            "sequence_id": self.sequence_id,
            "parent_id": self.parent_id,
            "seed_strategy": self.seed_strategy,
            "iteration": self.iteration,
            "n_codons": self.n_codons,
            "scores": self.scores.summary_dict(),
            "n_diagnostics": len(self.diagnostics),
            "n_accepted_edits": len(self.accepted_edits),
        }

    def __repr__(self) -> str:
        return (
            f"Candidate(id={self.sequence_id!r}, iter={self.iteration}, "
            f"cai={self.scores.cai}, mfe={self.scores.mfe})"
        )
