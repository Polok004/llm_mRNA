"""
ViennaRNA structure metrics.

Wraps the ViennaRNA Python bindings (import RNA).
Falls back gracefully if ViennaRNA is not installed — all functions return
None/empty and log a warning so the rest of the pipeline can continue.

Primary functions
-----------------
fold(seq)                   → (dot_bracket: str, mfe: float)
ensemble_fold(seq)          → EnsembleResult
fold_windows(seq, ...)      → list[WindowFoldResult]
start_codon_unpairing(...)  → float   (probability AUG is unpaired)
positional_entropy(seq)     → np.ndarray  (per-nt pairing entropy)

Long-sequence handling
----------------------
RNAfold scales as O(n³). For sequences > 2500 nt use LinearFold (if installed)
or the windowed approach.  LinearFold is tried first for long sequences.

Caching
-------
A simple in-process LRU cache keyed on (sequence, algorithm) avoids
re-folding identical sequences in population-based runs.
"""

from __future__ import annotations

import functools
import subprocess
import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

try:
    import RNA  # ViennaRNA Python bindings
    _RNA_AVAILABLE = True
except ImportError:
    _RNA_AVAILABLE = False
    warnings.warn(
        "ViennaRNA Python bindings not found. Structure metrics will be unavailable. "
        "Install with: pip install ViennaRNA",
        stacklevel=2,
    )

from mrna_design.logging_utils import get_logger

log = get_logger("metrics.structure")

_LINEARFOLD_LONG_THRESHOLD = 2500   # nt — use LinearFold above this
_WINDOW_SIZE_NT = 240               # nt per local-fold window
_WINDOW_STEP_NT = 60                # nt step


# ── Result types ──────────────────────────────────────────────────────────────

class FoldResult(NamedTuple):
    dot_bracket: str
    mfe: float          # kcal/mol


@dataclass
class EnsembleResult:
    centroid_structure: str = ""
    centroid_energy: float = 0.0
    ensemble_diversity: float = 0.0         # Ensemble diversity (ED)
    mfe: float = 0.0
    dot_bracket: str = ""
    positional_entropy: list[float] = field(default_factory=list)


@dataclass
class WindowFoldResult:
    window_start: int       # 0-based nt
    window_end: int         # 0-based nt, exclusive
    mfe: float
    dot_bracket: str
    gc_content: float
    algorithm: str = "RNAfold"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gc(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def _linearfold(seq: str) -> FoldResult | None:
    """Try to run LinearFold as a subprocess. Returns None if not found."""
    try:
        proc = subprocess.run(
            ["LinearFold"],
            input=seq,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0:
            lines = proc.stdout.strip().splitlines()
            # LinearFold output: seq on line 0, "struct (mfe)" on line 1
            for line in reversed(lines):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mfe = float(parts[-1].strip("()"))
                        return FoldResult(dot_bracket=parts[0], mfe=mfe)
                    except ValueError:
                        continue
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ── Public API ────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=512)
def fold(seq: str, algorithm: str = "auto") -> FoldResult:
    """
    Fold a sequence and return (dot_bracket, mfe).

    Parameters
    ----------
    seq : str
        RNA sequence (A/U/G/C).
    algorithm : "auto" | "rnafold" | "linearfold"
        "auto" uses LinearFold for sequences > _LINEARFOLD_LONG_THRESHOLD nt.
    """
    seq = seq.upper().replace("T", "U")
    n = len(seq)

    if algorithm in ("auto", "linearfold") and n > _LINEARFOLD_LONG_THRESHOLD:
        result = _linearfold(seq)
        if result is not None:
            log.event("fold_called", algorithm="linearfold", n=n, mfe=result.mfe)
            return result
        log.warn("linearfold_unavailable", fallback="rnafold", n=n)

    if not _RNA_AVAILABLE:
        log.warn("rnafold_unavailable", n=n)
        return FoldResult(dot_bracket="." * n, mfe=0.0)

    ss, mfe = RNA.fold(seq)
    log.event("fold_called", algorithm="rnafold", n=n, mfe=round(mfe, 3))
    return FoldResult(dot_bracket=ss, mfe=mfe)


def ensemble_fold(seq: str) -> EnsembleResult:
    """
    Compute MFE structure, centroid, ensemble diversity, and positional entropy.

    Requires ViennaRNA. Returns a zero-filled EnsembleResult if unavailable.
    """
    seq = seq.upper().replace("T", "U")
    n = len(seq)

    if not _RNA_AVAILABLE:
        log.warn("ensemble_fold_unavailable", n=n)
        return EnsembleResult(
            centroid_structure="." * n,
            positional_entropy=[0.0] * n,
        )

    md = RNA.md()
    fc = RNA.fold_compound(seq, md)
    ss, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()

    centroid, centroid_e = fc.centroid()
    bp_probs = fc.bpp()  # (n+1) × (n+1) matrix (1-indexed)

    # Per-nucleotide entropy: H_i = -sum_j P_ij log P_ij
    entropy = np.zeros(n)
    for i in range(1, n + 1):
        prob_paired = sum(bp_probs[min(i, j)][max(i, j)] for j in range(1, n + 1) if j != i)
        prob_paired = min(prob_paired, 1.0)
        prob_unpaired = 1.0 - prob_paired
        h = 0.0
        if prob_paired > 1e-10:
            h -= prob_paired * np.log2(prob_paired)
        if prob_unpaired > 1e-10:
            h -= prob_unpaired * np.log2(prob_unpaired)
        entropy[i - 1] = h

    # Ensemble diversity: mean base-pair distance from centroid
    diversity = fc.mean_bp_distance()

    log.event(
        "ensemble_fold_called",
        n=n,
        mfe=round(mfe, 3),
        ensemble_diversity=round(diversity, 3),
    )
    return EnsembleResult(
        centroid_structure=centroid,
        centroid_energy=centroid_e,
        ensemble_diversity=diversity,
        mfe=mfe,
        dot_bracket=ss,
        positional_entropy=entropy.tolist(),
    )


def fold_windows(
    seq: str,
    window_size: int = _WINDOW_SIZE_NT,
    step: int = _WINDOW_STEP_NT,
) -> list[WindowFoldResult]:
    """
    Fold the sequence in overlapping windows.

    Returns one WindowFoldResult per window.
    """
    seq = seq.upper().replace("T", "U")
    n = len(seq)
    results: list[WindowFoldResult] = []

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window = seq[start:end]
        fr = fold(window, algorithm="rnafold")
        results.append(WindowFoldResult(
            window_start=start,
            window_end=end,
            mfe=fr.mfe,
            dot_bracket=fr.dot_bracket,
            gc_content=_gc(window),
        ))

    if not results and n > 0:
        # Sequence shorter than window — fold the whole thing
        fr = fold(seq)
        results.append(WindowFoldResult(
            window_start=0,
            window_end=n,
            mfe=fr.mfe,
            dot_bracket=fr.dot_bracket,
            gc_content=_gc(seq),
        ))

    return results


def start_codon_unpairing(seq: str, cds_start: int) -> float:
    """
    Return the probability that the AUG start codon (3 nt at cds_start) is
    completely unpaired, using the base-pair probability matrix.

    Returns 0.0 if ViennaRNA is unavailable or cds_start is out of range.
    """
    seq = seq.upper().replace("T", "U")
    n = len(seq)
    if cds_start + 3 > n:
        return 0.0
    if not _RNA_AVAILABLE:
        return 0.0

    # Use a 40 nt window centred on the start codon for speed
    window_start = max(0, cds_start - 20)
    window_end = min(n, cds_start + 20)
    window = seq[window_start:window_end]
    aug_offset = cds_start - window_start   # AUG position in the window

    md = RNA.md()
    fc = RNA.fold_compound(window, md)
    fc.mfe()
    fc.exp_params_rescale()
    fc.pf()
    bp = fc.bpp()

    # Probability AUG position i is unpaired = 1 - sum of p(i,j) for all j
    prob_unpaired_all = []
    for k in range(aug_offset + 1, aug_offset + 4):   # 1-indexed for bpp
        paired_prob = sum(
            bp[min(k, j)][max(k, j)]
            for j in range(1, len(window) + 1)
            if j != k
        )
        prob_unpaired_all.append(max(0.0, 1.0 - min(paired_prob, 1.0)))

    return float(np.mean(prob_unpaired_all))
