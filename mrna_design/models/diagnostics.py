"""
RegionDiagnostic — structured diagnostic emitted by an agent.

Agents (StructureAgent, SafetyAgent, ImmunogenicityAgent) never return raw tool
output.  They return a list[RegionDiagnostic] that the Refinement Controller
reads to decide which codon edits to propose.

Issue taxonomy
--------------
Structure:
  stable_hairpin          — hairpin in 5'UTR / near start codon
  local_stable_stem       — CDS window with unusually low local MFE
  low_ensemble_diversity  — over-compacted fold, few reachable structures
  start_codon_paired      — AUG region has high pairing probability

Safety:
  mirna_seed_match        — 7mer-m8 / 8mer seed of a human miRNA
  blast_hit               — significant homology to human transcript
  rnahybrid_hit           — RNAhybrid significant interaction

Immunogenicity:
  cpg_hotspot             — CpG density > threshold in a local window
  upa_hotspot             — UpA density > threshold
  gu_tlr_motif            — known TLR7/8 GU-rich stimulatory motif
  long_dsrna_stem         — ≥40 bp perfect complement (RIG-I / MDA5 / PKR)
  high_uridine            — U-fraction > threshold in a window

Translation:
  uorf_detected           — upstream ORF that could sequester ribosomes
  codon_desert            — local sliding-window CAI below threshold
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Region(str, Enum):
    UTR5 = "5UTR"
    CDS = "CDS"
    UTR3 = "3UTR"
    JUNCTION_5_CDS = "5UTR-CDS_junction"
    JUNCTION_CDS_3 = "CDS-3UTR_junction"
    FULL = "full_sequence"
    UNKNOWN = "unknown"


class IssueType(str, Enum):
    # Structure
    STABLE_HAIRPIN = "stable_hairpin"
    LOCAL_STABLE_STEM = "local_stable_stem"
    LOW_ENSEMBLE_DIVERSITY = "low_ensemble_diversity"
    START_CODON_PAIRED = "start_codon_paired"
    # Safety
    MIRNA_SEED_MATCH = "mirna_seed_match"
    BLAST_HIT = "blast_hit"
    RNAHYBRID_HIT = "rnahybrid_hit"
    # Immunogenicity
    CPG_HOTSPOT = "cpg_hotspot"
    UPA_HOTSPOT = "upa_hotspot"
    GU_TLR_MOTIF = "gu_tlr_motif"
    LONG_DSRNA_STEM = "long_dsrna_stem"
    HIGH_URIDINE = "high_uridine"
    # Translation
    UORF_DETECTED = "uorf_detected"
    CODON_DESERT = "codon_desert"


class Severity(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class RegionDiagnostic(BaseModel):
    """One flagged issue in one region of the candidate sequence."""

    region: Region = Field(description="Broad region of the sequence.")
    issue: IssueType = Field(description="Issue type (taxonomic enum).")
    severity: Severity = Field(description="1=low, 2=medium, 3=high.")
    metric: float = Field(description="Numerical value that triggered the flag.")
    threshold: float = Field(description="Threshold that was exceeded.")
    window_start: int = Field(
        description="0-based nt index of window start (-1 if not applicable).",
        default=-1,
    )
    window_end: int = Field(
        description="0-based nt index of window end, exclusive (-1 if not applicable).",
        default=-1,
    )
    # For codon-level issues (e.g., codon_desert, mirna_seed_match in CDS)
    codon_start: int | None = Field(
        default=None,
        description="0-based codon index of first affected codon.",
    )
    codon_end: int | None = Field(
        default=None,
        description="0-based codon index of last affected codon (inclusive).",
    )
    detail: str = Field(
        default="",
        description="Human-readable description (e.g. miRNA name, motif sequence).",
    )
    suggestion: str | None = Field(
        default=None,
        description="Optional pre-computed edit hint (used by rule-based controller).",
    )
    source_agent: str = Field(
        default="unknown",
        description="Name of the agent that raised this diagnostic.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific payload (e.g. RNAhybrid ΔG, BLAST e-value).",
    )

    @field_validator("metric", "threshold", mode="before")
    @classmethod
    def _finite(cls, v: Any) -> float:
        import math

        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"metric/threshold must be finite, got {v}")
        return v

    def affects_codon(self, codon_index: int) -> bool:
        """Return True if this diagnostic covers the given codon index."""
        if self.codon_start is None or self.codon_end is None:
            return False
        return self.codon_start <= codon_index <= self.codon_end

    def to_controller_dict(self) -> dict[str, Any]:
        """
        Compact dict passed to the controller prompt.
        Excludes large or redundant fields to keep token count down.
        """
        d: dict[str, Any] = {
            "region": self.region.value,
            "issue": self.issue.value,
            "severity": self.severity.value,
            "metric": round(self.metric, 4),
            "threshold": round(self.threshold, 4),
        }
        if self.codon_start is not None:
            d["codon_range"] = [self.codon_start, self.codon_end]
        if self.window_start >= 0:
            d["window"] = [self.window_start, self.window_end]
        if self.detail:
            d["detail"] = self.detail
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d
