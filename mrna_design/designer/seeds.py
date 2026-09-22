"""
Seeding strategies — produce initial mRNA Candidate objects from a protein sequence.

Four strategies are implemented, each returning a Candidate:

1. cai_max         — greedy highest-frequency human codon at every position
2. gc_balanced     — random synonymous sampling weighted toward target GC
3. harmonised      — match human codon-usage ratios (no source organism data needed;
                     uses relative synonymous codon frequencies)
4. lineardesign    — call the LinearDesign binary (if installed); falls back to cai_max
"""

from __future__ import annotations

import random
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from mrna_design.models.candidate import Candidate
from mrna_design.validators.codon_table import (
    HUMAN_FREQUENCIES,
    SYNONYMOUS_CODONS,
    aa_synonyms,
    max_freq_codon,
    translate,
)
from mrna_design.logging_utils import get_logger

log = get_logger("designer.seeds")

SeedStrategy = Literal["cai_max", "gc_balanced", "harmonised", "lineardesign"]

_LINEARDESIGN_BINARY = "LinearDesign"   # expected on PATH


# ── Internal helpers ──────────────────────────────────────────────────────────

def _protein_to_codons_cai_max(protein: str) -> list[str]:
    """For each amino acid, pick the highest-frequency human codon."""
    result = []
    for aa in protein.upper():
        if aa == "*":
            result.append("UAA")   # preferred human stop
        else:
            result.append(max_freq_codon(aa))
    return result


def _protein_to_codons_gc_balanced(
    protein: str,
    target_gc: float = 0.55,
    rng: random.Random | None = None,
) -> list[str]:
    """
    For each amino acid, sample a synonymous codon with weights that push
    GC content toward `target_gc`.

    Weight each codon proportional to:
        human_frequency * exp(- lambda * |gc(codon) - target_gc|)
    where lambda controls strength of GC pressure (empirically 5).
    """
    import math
    rng = rng or random.Random()
    _LAMBDA = 5.0

    result = []
    for aa in protein.upper():
        synonyms = aa_synonyms(aa) if aa != "*" else ["UAA", "UAG", "UGA"]
        if not synonyms:
            result.append("AUG")
            continue
        weights = []
        for codon in synonyms:
            codon_gc = (codon.count("G") + codon.count("C")) / 3.0
            hf = HUMAN_FREQUENCIES.get(codon, 0.01)
            w = hf * math.exp(-_LAMBDA * abs(codon_gc - target_gc))
            weights.append(max(w, 1e-10))
        result.append(rng.choices(synonyms, weights=weights, k=1)[0])
    return result


def _protein_to_codons_harmonised(
    protein: str,
    rng: random.Random | None = None,
) -> list[str]:
    """
    Codon harmonisation: randomly sample codons proportional to human RSCU.
    This matches the relative codon-usage profile of the human host without
    always choosing the max-frequency codon (which can create structural issues).
    """
    rng = rng or random.Random()
    result = []
    for aa in protein.upper():
        synonyms = aa_synonyms(aa) if aa != "*" else ["UAA", "UAG", "UGA"]
        if not synonyms:
            result.append("AUG")
            continue
        weights = [max(HUMAN_FREQUENCIES.get(c, 1e-6), 1e-6) for c in synonyms]
        result.append(rng.choices(synonyms, weights=weights, k=1)[0])
    return result


def _try_lineardesign(protein: str) -> list[str] | None:
    """
    Run LinearDesign on the protein and parse its optimised CDS.
    Returns None if LinearDesign is not available or fails.
    """
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".aa", delete=False) as tf:
            tf.write(protein + "\n")
            tf_path = tf.name

        proc = subprocess.run(
            [_LINEARDESIGN_BINARY, "-i", tf_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            log.warn("lineardesign_nonzero_exit", code=proc.returncode, stderr=proc.stderr[:200])
            return None

        # Parse: LinearDesign outputs the optimised mRNA sequence on stdout
        # Expected format: one line with the mRNA sequence (may have spaces)
        for line in proc.stdout.splitlines():
            line = line.strip().replace(" ", "").upper().replace("T", "U")
            if len(line) >= len(protein) * 3 and all(c in "ACGU" for c in line):
                codons = [line[i : i + 3] for i in range(0, len(line), 3)]
                # Validate: must encode the same protein
                translated = translate("".join(codons))
                if translated.rstrip("*") == protein.rstrip("*").upper():
                    log.event("lineardesign_success", n_codons=len(codons))
                    return codons
        log.warn("lineardesign_parse_failed", stdout_preview=proc.stdout[:200])
        return None

    except FileNotFoundError:
        log.warn("lineardesign_not_installed", hint="Build LinearDesign from source and add to PATH")
        return None
    except subprocess.TimeoutExpired:
        log.warn("lineardesign_timeout")
        return None
    finally:
        import os as _os
        try:
            _os.unlink(tf_path)
        except Exception:
            pass


# ── Public API ────────────────────────────────────────────────────────────────

def seed_candidate(
    protein: str,
    strategy: SeedStrategy,
    utr5: str = "",
    utr3: str = "",
    rng: random.Random | None = None,
    gc_target: float = 0.55,
) -> Candidate:
    """
    Create a single Candidate from `protein` using the specified seeding strategy.

    Parameters
    ----------
    protein : str
        Amino acid sequence (single-letter, uppercase, with or without stop '*').
    strategy : SeedStrategy
    utr5 / utr3 : str
        UTR sequences (RNA or DNA; will be normalised to RNA).
    rng : random.Random | None
        Optional seeded RNG for reproducibility.
    gc_target : float
        Target GC content for the "gc_balanced" strategy.

    Returns
    -------
    Candidate
        A Candidate with cds_start, cds_end, and protein set; scores are empty.
    """
    # Ensure protein ends with stop codon for full-CDS generation
    prot_clean = protein.upper().strip()
    if not prot_clean.endswith("*"):
        prot_clean = prot_clean + "*"

    if strategy == "cai_max":
        codons = _protein_to_codons_cai_max(prot_clean)
    elif strategy == "gc_balanced":
        codons = _protein_to_codons_gc_balanced(prot_clean, target_gc=gc_target, rng=rng)
    elif strategy == "harmonised":
        codons = _protein_to_codons_harmonised(prot_clean, rng=rng)
    elif strategy == "lineardesign":
        codons = _try_lineardesign(prot_clean)
        if codons is None:
            log.warn("lineardesign_fallback", fallback="cai_max")
            codons = _protein_to_codons_cai_max(prot_clean)
    else:
        raise ValueError(f"Unknown seeding strategy: '{strategy}'")

    cds = "".join(codons)
    # Validate CDS translates back to the same protein
    translated = translate(cds)
    if translated.rstrip("*") != prot_clean.rstrip("*"):
        raise RuntimeError(
            f"Seed validation failed for strategy '{strategy}': "
            f"translated protein does not match input.\n"
            f"Expected: {prot_clean[:20]}...\n"
            f"Got:      {translated[:20]}..."
        )

    candidate = Candidate.from_cds(
        cds=cds,
        protein=prot_clean.rstrip("*"),
        utr5=utr5,
        utr3=utr3,
        seed_strategy=strategy,
    )
    log.event(
        "candidate_seeded",
        strategy=strategy,
        sequence_id=candidate.sequence_id,
        cds_length=len(cds),
    )
    return candidate


def seed_all_strategies(
    protein: str,
    utr5: str = "",
    utr3: str = "",
    rng_seed: int = 42,
    include_lineardesign: bool = True,
) -> list[Candidate]:
    """
    Seed one Candidate per strategy and return all of them.
    Useful for initialising a population.
    """
    strategies: list[SeedStrategy] = ["cai_max", "gc_balanced", "harmonised"]
    if include_lineardesign:
        strategies.append("lineardesign")

    candidates = []
    for strategy in strategies:
        rng = random.Random(rng_seed)
        try:
            c = seed_candidate(protein, strategy=strategy, utr5=utr5, utr3=utr3, rng=rng)
            candidates.append(c)
        except Exception as exc:
            log.error("seed_failed", strategy=strategy, error=str(exc))

    return candidates
