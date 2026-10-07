"""
CAI (Codon Adaptation Index) and sliding-window CAI.

Implements Sharp & Li (1987) CAI.
Reference set: human codon usage fractions from data/codon_tables/human_codon_usage.json.

The CAI of a sequence is the geometric mean of the RSCU-relative usage
weights for each codon in the CDS (excluding Met, Trp, and stop codons,
which have no alternatives).
"""

from __future__ import annotations

import math

from mrna_design.validators.codon_table import (
    HUMAN_FREQUENCIES,
    SYNONYMOUS_CODONS,
    codon_to_aa,
)


# Pre-compute w_i = f_i / max(f_i for synonyms of AA_i) for all codons
def _build_cai_weights() -> dict[str, float]:
    weights: dict[str, float] = {}
    for aa, codons in SYNONYMOUS_CODONS.items():
        if aa == "*":
            continue
        max_f = max(HUMAN_FREQUENCIES.get(c, 0.0) for c in codons)
        for c in codons:
            f = HUMAN_FREQUENCIES.get(c, 0.0)
            weights[c] = f / max_f if max_f > 0 else 0.0
    return weights


_CAI_WEIGHTS: dict[str, float] = _build_cai_weights()

# Codons that are excluded from CAI (no synonyms)
_EXCLUDED_AAS: frozenset[str] = frozenset({"M", "W", "*"})


def cai(cds: str) -> float:
    """
    Compute CAI for a CDS string (RNA alphabet, uppercase).

    Returns a value in (0, 1]. Returns 0.0 if no scoreable codons.
    Codons for Met (AUG), Trp (UGG), and stop codons are excluded.
    """
    cds = cds.upper().replace("T", "U")
    if len(cds) % 3 != 0:
        raise ValueError(f"CDS length {len(cds)} not divisible by 3.")

    log_sum = 0.0
    n = 0
    for i in range(0, len(cds), 3):
        codon = cds[i : i + 3]
        try:
            aa = codon_to_aa(codon)
        except KeyError:
            continue
        if aa in _EXCLUDED_AAS:
            continue
        w = _CAI_WEIGHTS.get(codon, 1e-10)
        log_sum += math.log(max(w, 1e-10))
        n += 1

    if n == 0:
        return 0.0
    return math.exp(log_sum / n)


def cai_vector(cds: str) -> list[float]:
    """
    Return per-codon CAI weights (w_i) for all codons in the CDS.

    Excluded codons (M, W, stop) get weight NaN for easy masking.
    """
    import math as _math

    cds = cds.upper().replace("T", "U")
    result = []
    for i in range(0, len(cds), 3):
        codon = cds[i : i + 3]
        if len(codon) < 3:
            break
        try:
            aa = codon_to_aa(codon)
        except KeyError:
            result.append(_math.nan)
            continue
        if aa in _EXCLUDED_AAS:
            result.append(_math.nan)
        else:
            result.append(_CAI_WEIGHTS.get(codon, 0.0))
    return result


def sliding_window_cai(
    cds: str,
    window_codons: int = 30,
    step_codons: int = 10,
) -> list[dict]:
    """
    Compute CAI for overlapping codon windows.

    Parameters
    ----------
    window_codons : int
        Number of codons per window.
    step_codons : int
        Stride in codons.

    Returns
    -------
    list of dicts with keys: codon_start, codon_end, cai
    """
    cds = cds.upper().replace("T", "U")
    n_codons = len(cds) // 3
    results = []
    for start in range(0, n_codons - window_codons + 1, step_codons):
        end = start + window_codons
        window_cds = cds[start * 3 : end * 3]
        results.append(
            {
                "codon_start": start,
                "codon_end": end,
                "cai": cai(window_cds),
            }
        )
    return results


def gc_content(sequence: str) -> float:
    """GC fraction of any RNA/DNA string."""
    seq = sequence.upper()
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def gc3_content(cds: str) -> float:
    """GC fraction at codon third positions only."""
    cds = cds.upper().replace("T", "U")
    third_positions = [cds[i + 2] for i in range(0, len(cds) - 2, 3)]
    if not third_positions:
        return 0.0
    return sum(1 for c in third_positions if c in "GC") / len(third_positions)
