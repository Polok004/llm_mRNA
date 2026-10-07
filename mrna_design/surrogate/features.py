"""
Feature engineering for the surrogate model.

All features are computed from the raw mRNA sequence (+ CDS coordinates)
without calling ViennaRNA or any external tool — intentionally lightweight
so the surrogate can be evaluated thousands of times per second.

Feature vector layout (fixed-length, documented below):
  [0]   cai                  — Codon Adaptation Index
  [1]   gc_content           — GC fraction of full sequence
  [2]   gc3_content          — GC fraction at 3rd codon positions only
  [3]   uridine_fraction     — U fraction of CDS
  [4]   cpg_density          — CpG / 100 nt (CDS)
  [5]   upa_density          — UpA / 100 nt (CDS)
  [6]   gu_motif_count       — count of TLR7 GU motifs in CDS
  [7]   uorf_count           — number of upstream ORFs in 5'UTR
  [8]   cai_mean             — mean per-codon CAI weight
  [9]   cai_min              — min per-codon CAI weight (worst codon)
  [10]  cai_std              — std of per-codon CAI weights
  [11]  gc_cv                — coefficient of variation of windowed GC
  [12]  gc_min_window        — min windowed GC (30 codons)
  [13]  gc_max_window        — max windowed GC (30 codons)
  [14]  codon_rare_frac      — fraction of codons with freq < 0.20
  [15]  dinuc_entropy        — Shannon entropy of dinucleotide frequencies
  [16]  codon_entropy        — Shannon entropy of codon usage
  [17]  stop_codon_type      — encoded stop: UAA=0, UAG=1, UGA=2
  [18]  cds_length_log       — log(CDS length in nt)
  [19]  poly_a_count         — count of A-runs ≥4 in CDS
  [20]  poly_u_count         — count of U-runs ≥4 in CDS
  [21]  start_context_gc     — GC fraction of ±15 nt around AUG

DIM = 22
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from mrna_design.metrics.cai import cai, cai_vector, gc3_content, gc_content
from mrna_design.metrics.immunogenicity import compute_immunogenicity
from mrna_design.validators.codon_table import HUMAN_FREQUENCIES
from mrna_design.validators.sequence_validator import check_no_uorfs

DIM = 22
FEATURE_NAMES = [
    "cai",
    "gc_content",
    "gc3_content",
    "uridine_fraction",
    "cpg_density",
    "upa_density",
    "gu_motif_count",
    "uorf_count",
    "cai_mean",
    "cai_min",
    "cai_std",
    "gc_cv",
    "gc_min_window",
    "gc_max_window",
    "codon_rare_frac",
    "dinuc_entropy",
    "codon_entropy",
    "stop_codon_type",
    "cds_length_log",
    "poly_a_count",
    "poly_u_count",
    "start_context_gc",
]

_STOP_MAP = {"UAA": 0.0, "UAG": 1.0, "UGA": 2.0}
_GC_WIN_CODONS = 30


def extract(
    sequence: str,
    cds_start: int,
    cds_end: int,
    utr5: str = "",
) -> np.ndarray:
    """
    Compute the 22-dimensional feature vector for a single candidate.

    Parameters
    ----------
    sequence : str
        Full mRNA sequence (5'UTR + CDS + 3'UTR), RNA alphabet.
    cds_start : int
        Index in `sequence` where the CDS starts (inclusive).
    cds_end : int
        Index in `sequence` where the CDS ends (exclusive, includes stop).
    utr5 : str
        The raw 5'UTR (for uORF count). Can be empty.

    Returns
    -------
    np.ndarray of shape (22,), dtype float32.
    """
    seq = sequence.upper().replace("T", "U")
    cds = seq[cds_start:cds_end]
    # Truncate to nearest multiple of 3 for codon-based features
    # (OpenVaccine sequences are 107 nt; raw FASTA may not be in-frame)
    cds_trimmed = cds[: (len(cds) // 3) * 3]
    if len(cds_trimmed) < 3:
        # Degenerate: return zeros
        return np.zeros(DIM, dtype=np.float32)

    # ── Basic global metrics ──────────────────────────────────────────────────
    cai_score = cai(cds_trimmed)
    gc = gc_content(cds_trimmed)
    gc3 = gc3_content(cds_trimmed)

    imm = compute_immunogenicity(cds_trimmed)
    uridine_frac = imm.uridine_fraction
    cpg = imm.cpg_density
    upa = imm.upa_density
    gu = float(imm.gu_motif_count)

    # uORF count
    uorf_result = check_no_uorfs(utr5)
    uorf_count = 0.0 if uorf_result.passed else float(uorf_result.reason.count(",") + 1)

    # ── Per-codon CAI stats ───────────────────────────────────────────────────
    vec = cai_vector(cds_trimmed)
    # Filter NaN (Met / Trp / stop codons have no synonyms — excluded from CAI)
    vec_valid = np.array([v for v in vec if not math.isnan(v)], dtype=np.float32)
    if len(vec_valid) == 0:
        cai_mean, cai_min, cai_std, rare_frac = cai_score, cai_score, 0.0, 0.0
    else:
        cai_mean = float(np.mean(vec_valid))
        cai_min = float(np.min(vec_valid))
        cai_std = float(np.std(vec_valid))
        rare_frac = float(np.mean(vec_valid < 0.20))

    # ── Windowed GC variation (30-codon windows) ──────────────────────────────
    codons = [
        cds_trimmed[i : i + 3]
        for i in range(0, len(cds_trimmed) - 2, 3)
        if i + 3 <= len(cds_trimmed)
    ]
    n_codons = len(codons)
    win_gc_vals = []
    step = max(1, _GC_WIN_CODONS // 3)
    for start in range(0, n_codons, step):
        window_seq = "".join(codons[start : start + _GC_WIN_CODONS])
        if len(window_seq) >= 6:
            win_gc_vals.append(gc_content(window_seq))
    if win_gc_vals:
        gc_arr = np.array(win_gc_vals)
        gc_cv = float(gc_arr.std() / (gc_arr.mean() + 1e-9))
        gc_min_w = float(gc_arr.min())
        gc_max_w = float(gc_arr.max())
    else:
        gc_cv = 0.0
        gc_min_w = gc
        gc_max_w = gc

    # ── Dinucleotide entropy ──────────────────────────────────────────────────
    dinuc_entropy = _dinuc_shannon(cds_trimmed)

    # ── Codon usage entropy ───────────────────────────────────────────────────
    codon_entropy = _codon_usage_entropy(cds_trimmed)

    # ── Stop codon type ───────────────────────────────────────────────────────
    stop = cds_trimmed[-3:] if len(cds_trimmed) >= 3 else "UAA"
    stop_type = _STOP_MAP.get(stop, 0.0)

    # ── Length ───────────────────────────────────────────────────────────────
    cds_len_log = math.log(max(len(cds_trimmed), 1))

    # ── Poly-runs in CDS ─────────────────────────────────────────────────────
    poly_a = _count_poly_runs(cds_trimmed, "A", min_run=4)
    poly_u = _count_poly_runs(cds_trimmed, "U", min_run=4)

    # ── Start-codon Kozak context GC ─────────────────────────────────────────
    ctx_start = max(0, cds_start - 15)
    ctx_end = min(len(seq), cds_start + 18)  # +3 for AUG itself
    ctx = seq[ctx_start:ctx_end]
    start_ctx_gc = gc_content(ctx) if ctx else gc

    feat = np.array(
        [
            cai_score,
            gc,
            gc3,
            uridine_frac,
            cpg,
            upa,
            gu,
            uorf_count,
            cai_mean,
            cai_min,
            cai_std,
            gc_cv,
            gc_min_w,
            gc_max_w,
            rare_frac,
            dinuc_entropy,
            codon_entropy,
            stop_type,
            cds_len_log,
            float(poly_a),
            float(poly_u),
            start_ctx_gc,
        ],
        dtype=np.float32,
    )

    assert len(feat) == DIM, f"Feature vector length mismatch: {len(feat)} != {DIM}"
    return feat


def batch_extract(
    sequences: Sequence[str],
    cds_starts: Sequence[int],
    cds_ends: Sequence[int],
    utr5s: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Compute feature matrix for a batch of sequences.

    Returns
    -------
    np.ndarray of shape (N, 22), dtype float32.
    """
    n = len(sequences)
    out = np.empty((n, DIM), dtype=np.float32)
    utr5s = utr5s or [""] * n
    for i, (seq, s, e, u5) in enumerate(zip(sequences, cds_starts, cds_ends, utr5s, strict=True)):
        out[i] = extract(seq, s, e, utr5=u5)
    return out


# ── Internal helpers ──────────────────────────────────────────────────────────


def _dinuc_shannon(seq: str) -> float:
    """Shannon entropy of dinucleotide frequencies."""
    if len(seq) < 2:
        return 0.0
    counts: dict[str, int] = {}
    for i in range(len(seq) - 1):
        dn = seq[i : i + 2]
        counts[dn] = counts.get(dn, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy / 4.0  # normalise by max possible (log2(16)=4)


def _codon_usage_entropy(cds: str) -> float:
    """Shannon entropy of the codon-frequency distribution used in this CDS."""
    codons = [cds[i : i + 3] for i in range(0, len(cds) - 2, 3) if i + 3 <= len(cds)]
    if not codons:
        return 0.0
    weights = [HUMAN_FREQUENCIES.get(c, 1e-6) for c in codons]
    total = sum(weights)
    entropy = 0.0
    for w in weights:
        p = w / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy / math.log2(max(len(codons), 2))  # normalised


def _count_poly_runs(seq: str, base: str, min_run: int = 4) -> int:
    """Count non-overlapping runs of `base` of length >= `min_run`."""
    count = 0
    i = 0
    while i < len(seq):
        if seq[i] == base:
            run = 0
            while i < len(seq) and seq[i] == base:
                run += 1
                i += 1
            if run >= min_run:
                count += 1
        else:
            i += 1
    return count
