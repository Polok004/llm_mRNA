"""
ObjectiveScores — all quantitative objectives for one mRNA candidate.

Objectives are split into five categories matching the research plan:
  1. Stability   (MFE, ensemble diversity, local-window MFE)
  2. Translation (CAI, tAI, start-codon unpairing, uORF count)
  3. Immunogenicity (CpG, UpA, GU motifs, dsRNA, uridine fraction)
  4. Off-target safety (miRNA seed hits, BLAST hits)
  5. Composite / surrogate (lightweight feature-based surrogate TE score, added Week 3)

All floats use sentinel None to represent "not yet computed" so that partially
scored candidates can exist during incremental evaluation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class ObjectiveScores(BaseModel):
    """Flat container for all per-candidate objective metrics."""

    # ── Stability ─────────────────────────────────────────────────────────────
    mfe: float | None = Field(
        default=None,
        description="Global minimum free energy (kcal/mol). More negative = more stable.",
    )
    ensemble_diversity: float | None = Field(
        default=None,
        description="Ensemble diversity from RNAfold -p. Higher = more flexible ensemble.",
    )
    gc_content: float | None = Field(
        default=None,
        description="Overall GC fraction [0, 1].",
    )
    gc3_content: float | None = Field(
        default=None,
        description="GC fraction at codon third positions [0, 1].",
    )
    mean_local_mfe: float | None = Field(
        default=None,
        description="Mean MFE across 240-nt sliding windows (kcal/mol).",
    )
    start_unpairing_prob: float | None = Field(
        default=None,
        description="Probability that the AUG start codon is unpaired [0, 1].",
    )

    # ── Translation ───────────────────────────────────────────────────────────
    cai: float | None = Field(
        default=None,
        description="Codon Adaptation Index relative to human [0, 1].",
    )
    tai: float | None = Field(
        default=None,
        description="tRNA Adaptation Index (not yet implemented — placeholder).",
    )
    uorf_count: int | None = Field(
        default=None,
        description="Number of upstream open reading frames in the 5'UTR.",
    )

    # ── Immunogenicity ─────────────────────────────────────────────────────────
    cpg_density: float | None = Field(
        default=None,
        description="CpG dinucleotide count per 100 nt.",
    )
    upa_density: float | None = Field(
        default=None,
        description="UpA dinucleotide count per 100 nt.",
    )
    gu_motif_count: int | None = Field(
        default=None,
        description="Count of known TLR7/8-stimulatory GU-rich 7-mer motifs.",
    )
    uridine_fraction: float | None = Field(
        default=None,
        description="Fraction of uridine residues in the sequence [0, 1].",
    )
    long_dsrna_count: int | None = Field(
        default=None,
        description="Number of complementary stems ≥ 40 bp (RIG-I/MDA5 triggers).",
    )

    # ── Off-target safety ─────────────────────────────────────────────────────
    mirna_seed_hits: int | None = Field(
        default=None,
        description="Number of miRNA 7mer/8mer seed matches in CDS + 3'UTR.",
    )
    blast_hits: int | None = Field(
        default=None,
        description="Number of significant BLAST hits to human transcriptome.",
    )

    # ── Surrogate / composite ─────────────────────────────────────────────────
    surrogate_te: float | None = Field(
        default=None,
        description="Predicted translation efficiency from the lightweight surrogate (Week 3+).",
    )
    surrogate_reactivity: float | None = Field(
        default=None,
        description="Surrogate-predicted mean SHAPE reactivity (structural openness proxy).",
    )
    surrogate_degradation: float | None = Field(
        default=None,
        description="Surrogate-predicted mean degradation rate (lower is better).",
    )
    surrogate_confidence: float | None = Field(
        default=None,
        description="Surrogate prediction confidence [0, 1] (from RF tree variance).",
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    computed_at: float | None = Field(
        default=None,
        description="Unix timestamp when scores were computed.",
    )

    @model_validator(mode="after")
    def _check_fractions(self) -> "ObjectiveScores":
        for name in ("gc_content", "gc3_content", "cai", "tai", "uridine_fraction",
                     "start_unpairing_prob", "surrogate_te", "surrogate_confidence"):
            val = getattr(self, name)
            if val is not None and not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        return self

    def is_complete(self, require_surrogate: bool = False) -> bool:
        """Return True if basic primary objectives are computed."""
        primary = [self.cai, self.gc_content, self.cpg_density]
        if require_surrogate:
            primary.append(self.surrogate_te)
        return all(v is not None for v in primary)

    def to_objective_vector(self) -> list[float]:
        """
        Return a fixed-length objective vector for Pareto dominance checks.

        Convention: **all values are to be minimised** by the archive.
        Signs are flipped where higher is better (CAI, start_unpairing_prob).

        Order:
          0  mfe                       (min; already negative)
          1  -cai                      (min proxy for max CAI)
          2  cpg_density               (min)
          3  upa_density               (min)
          4  gu_motif_count            (min)
          5  mirna_seed_hits           (min)
          6  uorf_count                (min)
          7  -start_unpairing_prob     (min proxy for max unpairing)
          8  long_dsrna_count          (min)
          9  surrogate_degradation     (min; only if available)
        """
        def _safe(v: float | None, default: float = 0.0) -> float:
            return float(v) if v is not None else default

        vec = [
            _safe(self.mfe, default=0.0),
            -_safe(self.cai, default=0.0),
            _safe(self.cpg_density, default=0.0),
            _safe(self.upa_density, default=0.0),
            float(_safe(self.gu_motif_count, default=0.0)),
            float(_safe(self.mirna_seed_hits, default=0.0)),
            float(_safe(self.uorf_count, default=0.0)),
            -_safe(self.start_unpairing_prob, default=0.0),
            float(_safe(self.long_dsrna_count, default=0.0)),
        ]
        if self.surrogate_degradation is not None:
            vec.append(_safe(self.surrogate_degradation))
        return vec

    def summary_dict(self) -> dict[str, Any]:
        """Compact dict for logging / JSON output."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
