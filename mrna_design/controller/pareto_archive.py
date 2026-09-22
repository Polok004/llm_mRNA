"""
ParetoArchive — maintains the non-dominated front across all iterations.

All candidates are stored with their objective vectors.  The archive:
  - Rejects a new candidate if it is dominated by any archive member.
  - Removes existing members dominated by the new candidate.
  - Computes hypervolume relative to a reference point.

Objective convention
---------------------
All objectives are **minimised** (see ObjectiveScores.to_objective_vector()).
Reference point is set slightly worse than the worst observed values.

Hypervolume
-----------
Computed via pymoo's HV indicator.  Falls back to a simple dominated-area
approximation if pymoo is not available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from pymoo.indicators.hv import HV
    _PYMOO = True
except ImportError:
    _PYMOO = False

from mrna_design.models.candidate import Candidate
from mrna_design.logging_utils import get_logger

log = get_logger("controller.pareto_archive")


@dataclass
class ArchiveEntry:
    candidate: Candidate
    objective_vector: list[float]
    added_at: float = field(default_factory=time.time)
    iteration: int = 0


class ParetoArchive:
    """Non-dominated archive with hypervolume tracking."""

    def __init__(self, ref_point_offset: float = 0.1) -> None:
        """
        Parameters
        ----------
        ref_point_offset : float
            How much worse than the worst observed value to set the HV reference point.
        """
        self._entries: list[ArchiveEntry] = []
        self._ref_offset = ref_point_offset
        self._hv_history: list[tuple[int, float]] = []   # (iteration, hv)
        self._global_worst: np.ndarray | None = None

    # ── Core operations ────────────────────────────────────────────────────────

    def update(self, candidate: Candidate, iteration: int = 0) -> bool:
        """
        Add candidate to the archive if it is non-dominated.

        Returns True if the candidate was added (i.e. it is non-dominated).
        """
        if not candidate.scores.is_complete():
            log.warn("archive_update_incomplete_scores", candidate_id=candidate.sequence_id)
            return False

        vec = candidate.scores.to_objective_vector()

        # Check if dominated by any existing entry
        for entry in self._entries:
            if _dominates(entry.objective_vector, vec):
                log.debug("candidate_dominated", candidate_id=candidate.sequence_id)
                return False

        # Update global worst for monotonic hypervolume reference point
        vec_np = np.array(vec, dtype=float)
        if self._global_worst is None:
            self._global_worst = vec_np.copy()
        else:
            self._global_worst = np.maximum(self._global_worst, vec_np)

        # Remove entries dominated by the new candidate
        dominated = [e for e in self._entries if _dominates(vec, e.objective_vector)]
        for e in dominated:
            self._entries.remove(e)
            log.debug("entry_removed_dominated", removed_id=e.candidate.sequence_id)

        self._entries.append(ArchiveEntry(
            candidate=candidate,
            objective_vector=vec,
            iteration=iteration,
        ))
        hv = self.hypervolume()
        self._hv_history.append((iteration, hv))
        log.event(
            "archive_updated",
            iteration=iteration,
            archive_size=len(self._entries),
            hypervolume=round(hv, 6),
            candidate_id=candidate.sequence_id,
        )
        return True

    def hypervolume(self) -> float:
        """Compute hypervolume of the current Pareto front."""
        if not self._entries:
            return 0.0

        vecs = np.array([e.objective_vector for e in self._entries], dtype=float)
        if self._global_worst is None:
            ref = vecs.max(axis=0) + self._ref_offset
        else:
            ref = self._global_worst + self._ref_offset

        if _PYMOO:
            ind = HV(ref_point=ref)
            return float(ind(vecs))
        else:
            # Fallback: sum of dominated hypervolume for 2-objective approximation
            return float(_simple_hv_2d(vecs, ref))

    def dominated_hypervolume_contribution(self) -> list[float]:
        """Per-entry hypervolume contribution (for crowding-distance selection)."""
        if len(self._entries) <= 1:
            return [self.hypervolume()]
        total = self.hypervolume()
        contributions = []
        for i in range(len(self._entries)):
            subset = [e for j, e in enumerate(self._entries) if j != i]
            vecs = np.array([e.objective_vector for e in subset], dtype=float)
            if self._global_worst is None:
                ref = vecs.max(axis=0) + self._ref_offset
            else:
                ref = self._global_worst + self._ref_offset
            if _PYMOO:
                ind = HV(ref_point=ref)
                hv_without = float(ind(vecs))
            else:
                hv_without = float(_simple_hv_2d(vecs, ref))
            contributions.append(total - hv_without)
        return contributions

    # ── Queries ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(e.candidate for e in self._entries)

    def best_by(self, objective: str) -> Candidate | None:
        """Return the archive member with the best single-objective value."""
        if not self._entries:
            return None
        maximize = objective in ("cai", "tai", "start_unpairing_prob", "ensemble_diversity")
        def _key(e: ArchiveEntry) -> float:
            v = getattr(e.candidate.scores, objective, None)
            return float(v or 0.0)
        if maximize:
            return max(self._entries, key=_key).candidate
        return min(self._entries, key=_key).candidate

    def hv_history(self) -> list[tuple[int, float]]:
        """Return list of (iteration, hypervolume) tuples."""
        return list(self._hv_history)

    def has_converged(self, window: int = 10, tol: float = 1e-4) -> bool:
        """Return True if hypervolume has not improved by more than tol in the last `window` iterations."""
        if len(self._hv_history) < window:
            return False
        recent = [hv for _, hv in self._hv_history[-window:]]
        return (max(recent) - min(recent)) < tol

    def to_dataframe(self):
        """Return a pandas DataFrame of all archive entries and their scores."""
        import pandas as pd
        rows = []
        for e in self._entries:
            row = {
                "sequence_id": e.candidate.sequence_id,
                "iteration": e.iteration,
                "seed_strategy": e.candidate.seed_strategy,
                **e.candidate.scores.summary_dict(),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        """Save archive to a JSON Lines file."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for e in self._entries:
                f.write(json.dumps({
                    "sequence_id": e.candidate.sequence_id,
                    "iteration": e.iteration,
                    "sequence": e.candidate.sequence,
                    "protein": e.candidate.protein,
                    "cds_start": e.candidate.cds_start,
                    "cds_end": e.candidate.cds_end,
                    "scores": e.candidate.scores.model_dump(),
                    "objective_vector": e.objective_vector,
                    "added_at": e.added_at,
                }) + "\n")
        log.event("archive_saved", path=str(path), size=len(self._entries))


# ── Pareto dominance ──────────────────────────────────────────────────────────

def _dominates(a: list[float], b: list[float]) -> bool:
    """Return True if solution `a` dominates `b` (all-minimise convention)."""
    at_least_one_better = False
    for ai, bi in zip(a, b):
        if ai > bi:
            return False
        if ai < bi:
            at_least_one_better = True
    return at_least_one_better


def _simple_hv_2d(vecs: np.ndarray, ref: np.ndarray) -> float:
    """Approximate 2D hypervolume (uses first two objectives)."""
    if vecs.shape[0] == 0:
        return 0.0
    v = vecs[:, :2]
    r = ref[:2]
    # Sort by first objective ascending
    order = np.argsort(v[:, 0])
    v = v[order]
    hv = 0.0
    prev_x = r[0]
    for i in range(len(v) - 1, -1, -1):
        width = prev_x - v[i, 0]
        height = r[1] - v[i, 1]
        if width > 0 and height > 0:
            hv += width * height
        prev_x = v[i, 0]
    return max(hv, 0.0)
