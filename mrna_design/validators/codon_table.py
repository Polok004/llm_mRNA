"""
Codon table, synonymous menu, and human codon usage frequencies.

All codons use RNA alphabet (U not T).

Public API
----------
STANDARD_CODE          : dict[str, str]   codon → single-letter AA
SYNONYMOUS_CODONS      : dict[str, list[str]]  AA → list of synonymous codons
HUMAN_FREQUENCIES      : dict[str, float]  codon → usage fraction (0–1)
HUMAN_FREQ_PER1K       : dict[str, float]  codon → frequency per 1000 codons
codon_to_aa(codon)     : str
aa_synonyms(aa)        : list[str]  — codons that encode this AA (excluding stop)
max_freq_codon(aa)     : str        — highest-frequency human codon for this AA
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

# ── Raw data ────────────────────────────────────────────────────────────────

_DATA_FILE = (
    Path(__file__).parent.parent.parent / "data" / "codon_tables" / "human_codon_usage.json"
)


def _load_usage() -> dict:
    with open(_DATA_FILE) as f:
        return json.load(f)


_RAW = _load_usage()

# ── Standard genetic code (RNA, all 64 codons) ───────────────────────────────

STANDARD_CODE: Final[dict[str, str]] = {
    k: v["aa"] for k, v in _RAW.items() if not k.startswith("_")
}

# ── Human codon usage ─────────────────────────────────────────────────────────

HUMAN_FREQUENCIES: Final[dict[str, float]] = {
    k: v["fraction"] for k, v in _RAW.items() if not k.startswith("_")
}

HUMAN_FREQ_PER1K: Final[dict[str, float]] = {
    k: v["freq_per1k"] for k, v in _RAW.items() if not k.startswith("_")
}

# ── Synonymous codon menu ─────────────────────────────────────────────────────


def _build_synonymous() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for codon, aa in STANDARD_CODE.items():
        groups.setdefault(aa, []).append(codon)
    # Sort each group by human frequency descending (most frequent first)
    for aa in groups:
        groups[aa].sort(key=lambda c: HUMAN_FREQUENCIES.get(c, 0.0), reverse=True)
    return groups


SYNONYMOUS_CODONS: Final[dict[str, list[str]]] = _build_synonymous()

# ── Helpers ───────────────────────────────────────────────────────────────────


def codon_to_aa(codon: str) -> str:
    """
    Translate a single codon (RNA alphabet, uppercase) to single-letter amino acid.
    Returns '*' for stop codons.
    Raises KeyError for unknown codons.
    """
    codon = codon.upper().replace("T", "U")
    return STANDARD_CODE[codon]


def aa_synonyms(aa: str, exclude_stops: bool = True) -> list[str]:
    """
    Return the list of codons that encode `aa`, sorted by human frequency (desc).
    If `exclude_stops` is True and aa == '*', returns [].
    """
    aa = aa.upper()
    if exclude_stops and aa == "*":
        return []
    return SYNONYMOUS_CODONS.get(aa, [])


def max_freq_codon(aa: str) -> str:
    """Return the single highest-frequency human codon for amino acid `aa`."""
    synonyms = aa_synonyms(aa)
    if not synonyms:
        raise ValueError(f"No synonymous codons for '{aa}'")
    return synonyms[0]  # Already sorted by frequency descending


def is_synonymous(codon_a: str, codon_b: str) -> bool:
    """Return True if codon_a and codon_b encode the same amino acid."""
    return codon_to_aa(codon_a) == codon_to_aa(codon_b)


def synonymous_menu_for_region(codons: list[str]) -> dict[int, list[str]]:
    """
    Build the synonymous-codon menu for a list of codons.

    Returns a dict: codon_index → [alternative synonymous codons] (excl. self).
    Used in controller prompts to show the LLM what substitutions are possible.
    """
    menu: dict[int, list[str]] = {}
    for i, codon in enumerate(codons):
        aa = codon_to_aa(codon)
        alternatives = [c for c in aa_synonyms(aa) if c != codon]
        if alternatives:
            menu[i] = alternatives
    return menu


def translate(cds: str, stop_symbol: str = "*") -> str:
    """
    Translate a CDS string (RNA alphabet) to amino acid sequence.

    Parameters
    ----------
    cds : str
        CDS including stop codon. Length must be divisible by 3.
    stop_symbol : str
        Character to use for stop codons in the output string.

    Returns
    -------
    str
        Amino acid sequence (stop codon represented by stop_symbol).
    """
    cds = cds.upper().replace("T", "U")
    if len(cds) % 3 != 0:
        raise ValueError(f"CDS length {len(cds)} is not divisible by 3.")
    return "".join((codon_to_aa(cds[i : i + 3]) or stop_symbol) for i in range(0, len(cds), 3))
