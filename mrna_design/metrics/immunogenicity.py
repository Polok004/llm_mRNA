"""
Immunogenicity metrics.

Computes sequence-level signals that correlate with innate immune activation
for mRNA therapeutics. These are in-silico proxies, not measured immunogenicity.

Metrics
-------
cpg_density         CpG dinucleotide count per 100 nt
upa_density         UpA dinucleotide count per 100 nt
gu_motif_count      Count of TLR7/8-stimulatory GU-rich 7-mer motifs
uridine_fraction    Fraction of U residues
long_dsrna_count    Number of complementary stems ≥ 40 bp (RIG-I/MDA5 triggers)

References
----------
- Nallagatla & Bevilacqua (2008) — TLR7/8 GU-rich motifs
- Kariko et al. (2005, 2008) — CpG, UpA suppression in mRNA
- Hornung et al. (2006) — RIG-I dsRNA sensing threshold
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mrna_design.logging_utils import get_logger

log = get_logger("metrics.immunogenicity")

# ── TLR7/8 GU-rich motifs (RNA, 7-mer) ───────────────────────────────────────
# Curated from Nallagatla & Bevilacqua (2008) and Heil et al. (2004).
# These are exact matches; extend as needed.
_TLR_GU_MOTIFS: list[str] = [
    "GUUGUGU",  # canonical TLR7 stimulatory motif
    "GUCCUUC",
    "UUGUCUU",
    "UGUCUUG",
    "GUGUGUU",
    "UUGUUGU",
    "GUUGUUG",
    "UCUUGUG",
    "GUGUUGU",
    "UUGCUUC",
]

# Also flag any window with ≥ 4 consecutive GU dinucleotides
_GU_REPEAT_PATTERN = re.compile(r"(GU){4,}")


# ── Long dsRNA detection ──────────────────────────────────────────────────────
# Heuristic: find perfect-complement pairs of length ≥ 40 nt using naive scan.
# Full structural detection requires RNAfold stem annotations (done in agents).
_COMP: dict[str, str] = {"A": "U", "U": "A", "G": "C", "C": "G"}


def reverse_complement(seq: str) -> str:
    """Reverse complement of an RNA sequence. Unknown characters map to 'N'."""
    return "".join(_COMP.get(c, "N") for c in reversed(seq.upper().replace("T", "U")))


# Backwards-compatible private alias (several modules imported the underscored name).
_reverse_complement = reverse_complement


def _count_long_dsrna_stems(seq: str, min_len: int = 40) -> int:
    """
    Count regions where seq contains a substring that is complementary to
    another region in the same sequence (intra-molecular).

    This is a coarse heuristic — full stem detection uses RNAfold.
    We scan for perfect RC pairs of ≥ min_len nt.
    """
    seq = seq.upper().replace("T", "U")
    n = len(seq)
    count = 0
    step = min_len // 2
    checked: set[tuple[int, int]] = set()

    for i in range(0, n - min_len, step):
        window = seq[i : i + min_len]
        rc = _reverse_complement(window)
        # Search for rc in the rest of the sequence
        pos = seq.find(rc, i + min_len)
        if pos != -1 and (i, pos) not in checked:
            checked.add((i, pos))
            count += 1

    return count


# ── Public API ────────────────────────────────────────────────────────────────


@dataclass
class ImmunogenicityResult:
    cpg_density: float  # per 100 nt
    upa_density: float  # per 100 nt
    gu_motif_count: int
    uridine_fraction: float
    long_dsrna_count: int


def compute_immunogenicity(sequence: str) -> ImmunogenicityResult:
    """
    Compute all immunogenicity metrics for a full mRNA sequence.

    Parameters
    ----------
    sequence : str
        Full mRNA sequence (RNA alphabet, uppercase). May include UTRs.

    Returns
    -------
    ImmunogenicityResult
    """
    seq = sequence.upper().replace("T", "U")
    n = len(seq)
    if n == 0:
        return ImmunogenicityResult(0.0, 0.0, 0, 0.0, 0)

    per100 = 100.0 / n

    # CpG density (DNA convention — but applies to RNA as CG dinucleotide)
    cpg = seq.count("CG") * per100

    # UpA density
    upa = seq.count("UA") * per100

    # GU motifs
    gu_count = sum(seq.count(m) for m in _TLR_GU_MOTIFS)
    gu_count += len(_GU_REPEAT_PATTERN.findall(seq))

    # Uridine fraction
    u_frac = seq.count("U") / n

    # Long dsRNA stems
    dsrna = _count_long_dsrna_stems(seq)

    result = ImmunogenicityResult(
        cpg_density=round(cpg, 4),
        upa_density=round(upa, 4),
        gu_motif_count=gu_count,
        uridine_fraction=round(u_frac, 4),
        long_dsrna_count=dsrna,
    )
    log.event(
        "immunogenicity_computed",
        n=n,
        cpg_density=result.cpg_density,
        upa_density=result.upa_density,
        gu_motif_count=result.gu_motif_count,
        uridine_fraction=result.uridine_fraction,
        long_dsrna_count=result.long_dsrna_count,
    )
    return result


def cpg_density_windows(
    sequence: str,
    window: int = 100,
    step: int = 25,
) -> list[dict]:
    """Return per-window CpG density for hotspot detection."""
    seq = sequence.upper().replace("T", "U")
    n = len(seq)
    results = []
    for start in range(0, n - window + 1, step):
        end = start + window
        w = seq[start:end]
        results.append(
            {
                "start": start,
                "end": end,
                "cpg_density": w.count("CG") * (100.0 / len(w)),
            }
        )
    return results
