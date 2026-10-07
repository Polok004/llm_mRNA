"""
Sequence validators — deterministic hard constraints on mRNA candidates.

All validators are pure functions that return True/False or a list of hits.
They are called by CodonApplicator before accepting any edit.

Hard constraints
----------------
1. Protein identity: the CDS must translate to exactly the original protein.
2. No premature stop codons before the annotated stop.
3. GC content in [GC_LO, GC_HI].
4. No forbidden restriction enzyme recognition sites in the CDS.
5. Valid start codon (AUG) at cds_start.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from mrna_design.validators.codon_table import translate

# ── Constants ─────────────────────────────────────────────────────────────────

GC_LO: float = 0.30  # absolute minimum; typical optimisation target is 0.45–0.65
GC_HI: float = 0.80  # absolute maximum

# Default forbidden restriction sites (DNA alphabet for regex; converted to RNA internally)
# Format: name → recognition site (5'→3', IUPAC DNA)
DEFAULT_FORBIDDEN_SITES: dict[str, str] = {
    "EcoRI": "GAATTC",
    "BamHI": "GGATCC",
    "HindIII": "AAGCTT",
    "NotI": "GCGGCCGC",
    "XhoI": "CTCGAG",
    "SalI": "GTCGAC",
    "NheI": "GCTAGC",
    "SpeI": "ACTAGT",
    "AscI": "GGCGCGCC",
    "PacI": "TTAATTAA",
}

# IUPAC ambiguity → regex character class
_IUPAC_REGEX: dict[str, str] = {
    "R": "[AG]",
    "Y": "[CU]",
    "S": "[GC]",
    "W": "[AU]",
    "K": "[GU]",
    "M": "[AC]",
    "B": "[CGU]",
    "D": "[AGU]",
    "H": "[ACU]",
    "V": "[ACG]",
    "N": "[ACGU]",
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
}


def _site_to_rna_regex(site_dna: str) -> str:
    """Convert a DNA IUPAC restriction site to an RNA regex pattern."""
    rna = site_dna.upper().replace("T", "U")
    return "".join(_IUPAC_REGEX.get(c, c) for c in rna)


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RestrictionHit:
    enzyme: str
    site: str
    position: int  # 0-based nt start in the full sequence


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.passed


# ── Validators ────────────────────────────────────────────────────────────────


def check_protein_identity(
    sequence: str,
    cds_start: int,
    cds_end: int,
    expected_protein: str,
    *,
    include_stop: bool = True,
) -> ValidationResult:
    """
    Verify that sequence[cds_start:cds_end] translates to expected_protein.

    Parameters
    ----------
    include_stop : bool
        If True, the translated string will end with '*'; the comparison
        strips it and also strips a trailing '*' from expected_protein.
    """
    cds = sequence[cds_start:cds_end].upper().replace("T", "U")
    try:
        translated = translate(cds)
    except (KeyError, ValueError) as exc:
        return ValidationResult(False, f"Translation failed: {exc}")

    # Strip stop codon symbol for comparison
    translated_aa = translated.rstrip("*")
    expected_aa = expected_protein.strip().rstrip("*").upper()

    if translated_aa == expected_aa:
        return ValidationResult(True)
    # Provide a useful diff hint
    mismatches = [
        i for i, (a, b) in enumerate(zip(translated_aa, expected_aa, strict=False)) if a != b
    ]
    length_diff = len(translated_aa) - len(expected_aa)
    return ValidationResult(
        False,
        f"Protein mismatch: length_diff={length_diff}, "
        f"first {min(5, len(mismatches))} mismatch positions={mismatches[:5]}",
    )


def check_no_premature_stop(
    sequence: str,
    cds_start: int,
    cds_end: int,
) -> ValidationResult:
    """
    Verify there is no stop codon before cds_end - 3 (i.e. before the final stop).

    The CDS is expected to end with a stop codon, so the final codon is allowed to
    be a stop.
    """
    cds = sequence[cds_start:cds_end].upper().replace("T", "U")
    stop_codons = {"UAA", "UAG", "UGA"}
    for i in range(0, len(cds) - 3, 3):  # all codons except the last (stop)
        codon = cds[i : i + 3]
        if codon in stop_codons:
            return ValidationResult(
                False,
                f"Premature stop codon '{codon}' at CDS codon index {i // 3} (nt {cds_start + i})",
            )
    return ValidationResult(True)


def check_start_codon(sequence: str, cds_start: int) -> ValidationResult:
    """Verify the CDS begins with AUG."""
    start = sequence[cds_start : cds_start + 3].upper().replace("T", "U")
    if start == "AUG":
        return ValidationResult(True)
    return ValidationResult(False, f"Start codon is '{start}', expected 'AUG'.")


def check_gc_bounds(
    sequence: str,
    cds_start: int,
    cds_end: int,
    lo: float = GC_LO,
    hi: float = GC_HI,
) -> ValidationResult:
    """Verify GC content of the CDS is within [lo, hi]."""
    cds = sequence[cds_start:cds_end].upper()
    if not cds:
        return ValidationResult(False, "Empty CDS.")
    gc = (cds.count("G") + cds.count("C")) / len(cds)
    if lo <= gc <= hi:
        return ValidationResult(True)
    return ValidationResult(False, f"GC content {gc:.3f} outside [{lo}, {hi}].")


def check_restriction_sites(
    sequence: str,
    cds_start: int,
    cds_end: int,
    forbidden: dict[str, str] | None = None,
) -> list[RestrictionHit]:
    """
    Find forbidden restriction enzyme sites in the CDS region.

    Parameters
    ----------
    forbidden : dict[str, str] | None
        enzyme_name → recognition_site (DNA IUPAC). Defaults to DEFAULT_FORBIDDEN_SITES.

    Returns
    -------
    list[RestrictionHit]
        Empty if no hits.
    """
    if forbidden is None:
        forbidden = DEFAULT_FORBIDDEN_SITES
    cds = sequence[cds_start:cds_end].upper().replace("T", "U")
    hits: list[RestrictionHit] = []
    for name, site_dna in forbidden.items():
        pattern = _site_to_rna_regex(site_dna)
        for m in re.finditer(pattern, cds):
            hits.append(
                RestrictionHit(
                    enzyme=name,
                    site=site_dna,
                    position=cds_start + m.start(),
                )
            )
    return hits


def check_no_uorfs(utr5: str) -> ValidationResult:
    """
    Check for upstream open reading frames in the 5'UTR.

    A uORF is defined as an AUG ... stop codon within the 5'UTR.
    We do not require the stop to be in-frame with the main ORF.
    """
    utr = utr5.upper().replace("T", "U")
    stop_codons = {"UAA", "UAG", "UGA"}
    uorf_starts = []
    for i in range(len(utr) - 2):
        if utr[i : i + 3] == "AUG":
            # Search for any stop codon in-frame from this AUG
            for j in range(i + 3, len(utr) - 2, 3):
                if utr[j : j + 3] in stop_codons:
                    uorf_starts.append(i)
                    break
    if uorf_starts:
        return ValidationResult(
            False,
            f"uORF(s) detected at 5'UTR positions: {uorf_starts[:5]}",
        )
    return ValidationResult(True)


def validate_all(
    sequence: str,
    cds_start: int,
    cds_end: int,
    protein: str,
    utr5: str = "",
    forbidden_sites: dict[str, str] | None = None,
    gc_lo: float = GC_LO,
    gc_hi: float = GC_HI,
) -> list[ValidationResult]:
    """
    Run all hard constraints and return a list of results.
    An empty list or all-True results means the sequence passes.
    """
    results = [
        check_start_codon(sequence, cds_start),
        check_no_premature_stop(sequence, cds_start, cds_end),
        check_protein_identity(sequence, cds_start, cds_end, protein),
        check_gc_bounds(sequence, cds_start, cds_end, lo=gc_lo, hi=gc_hi),
    ]
    hits = check_restriction_sites(sequence, cds_start, cds_end, forbidden_sites)
    if hits:
        enzymes = ", ".join(h.enzyme for h in hits)
        results.append(ValidationResult(False, f"Restriction sites found: {enzymes}"))
    return results


def all_pass(results: Sequence[ValidationResult]) -> bool:
    return all(r.passed for r in results)
