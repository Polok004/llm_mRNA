"""
ParetoArchive — the non-dominated front, with a hypervolume you can actually compare.

What changed and why
--------------------
The archive is where the project's headline number comes from, so its
definition has to be airtight. Three things were fixed:

1. **Objectives are normalised.** Entries are stored as vectors in [0, 1]^m
   produced by an explicit :class:`~mrna_design.metrics.objective_spec.ObjectiveSet`,
   rather than as raw mixed-unit values. Previously MFE (hundreds of kcal/mol)
   shared a volume computation with CAI (a fraction), and the volume was
   therefore ~95% a function of MFE alone.

2. **The reference point is fixed.** It used to be derived from the worst value
   seen *within the run* (``self._global_worst + 0.1``), which made HV a
   function of the run's own history: a run that happened to evaluate one very
   bad candidate got a larger reference box and a larger HV, for free. Two runs
   produced two incomparable numbers, yet the benchmark compared them with a
   Mann-Whitney test. The reference point is now ``1 + REF_EPS`` in every
   dimension, fixed by the objective set, so HV is an absolute quantity on a
   known scale and is comparable across iterations, seeds, controllers and
   targets.

3. **Vectors are fixed-length.** Dominance is checked with an explicit length
   assertion instead of ``zip``, which used to silently truncate to the shorter
   of the two vectors.

The archive also tracks hypervolume against *evaluations spent*, not just
iteration index, because iterations are not a fair unit of budget when different
controllers propose different numbers of edits per call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pymoo.indicators.hv import HV

    _PYMOO = True
except ImportError:  # pragma: no cover - exercised only in minimal installs
    _PYMOO = False

from mrna_design.logging_utils import get_logger
from mrna_design.metrics.objective_spec import (
    DEFAULT_OBJECTIVE_SET,
    ObjectiveSet,
    degenerate_objectives,
    get_objective_set,
)
from mrna_design.models.candidate import Candidate

log = get_logger("controller.pareto_archive")


@dataclass
class ArchiveEntry:
    candidate: Candidate
    objective_vector: list[float]  # normalised, all-minimise, in [0, 1]^m
    added_at: float = field(default_factory=time.time)
    iteration: int = 0
    evaluations: int = 0  # budget spent when this entry was admitted


@dataclass
class HVPoint:
    """One point on the convergence trace."""

    iteration: int
    evaluations: int
    hypervolume: float
    archive_size: int

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "evaluations": self.evaluations,
            "hypervolume": self.hypervolume,
            "archive_size": self.archive_size,
            # Legacy key: older analysis scripts and saved summaries read "hv".
            "hv": self.hypervolume,
        }


class ParetoArchive:
    """Non-dominated archive over a normalised objective space."""

    def __init__(
        self,
        objective_set: ObjectiveSet | str | None = None,
        max_size: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        objective_set : ObjectiveSet | str | None
            The objective space to work in. Defaults to the PRIMARY set.
        max_size : int | None
            Optional cap on archive size. When exceeded, the entry with the
            smallest hypervolume contribution is dropped — the standard
            crowding-based truncation. ``None`` means unbounded.
        """
        self.objective_set = (
            DEFAULT_OBJECTIVE_SET if objective_set is None else get_objective_set(objective_set)
        )
        self.max_size = max_size
        self._entries: list[ArchiveEntry] = []
        self._hv_history: list[HVPoint] = []
        self._evaluations = 0

        # Populated by optimize() when the run finishes, so callers, the CLI and
        # the run manifest can read back what the run actually cost. Declared
        # here rather than attached dynamically so the attributes are typed.
        self.run_budget: Any = None
        self.run_stats: Any = None
        self.run_cache: Any = None

    # ── Budget bookkeeping ────────────────────────────────────────────────────

    @property
    def evaluations(self) -> int:
        """Objective evaluations charged to this archive so far."""
        return self._evaluations

    def note_evaluations(self, n: int) -> None:
        """Record that ``n`` objective evaluations have been spent."""
        self._evaluations += n

    # ── Core operations ───────────────────────────────────────────────────────

    def vector_for(self, candidate: Candidate) -> list[float]:
        """Normalised objective vector for a candidate, in this archive's space."""
        return self.objective_set.vector(candidate.scores, len(candidate.sequence))

    def update(self, candidate: Candidate, iteration: int = 0) -> bool:
        """
        Add ``candidate`` to the archive if it is non-dominated.

        Returns True if the candidate was admitted.
        """
        if not candidate.scores.is_complete():
            log.warn("archive_update_incomplete_scores", candidate_id=candidate.sequence_id)
            return False

        vec = self.vector_for(candidate)

        # Dominated by an incumbent? Then reject.
        for entry in self._entries:
            if _dominates(entry.objective_vector, vec):
                log.debug("candidate_dominated", candidate_id=candidate.sequence_id)
                return False

        # Identical objective vector adds nothing to the front.
        if any(np.allclose(e.objective_vector, vec) for e in self._entries):
            log.debug("candidate_duplicate_objectives", candidate_id=candidate.sequence_id)
            return False

        # Evict incumbents this candidate dominates.
        survivors = [e for e in self._entries if not _dominates(vec, e.objective_vector)]
        n_evicted = len(self._entries) - len(survivors)
        self._entries = survivors

        self._entries.append(
            ArchiveEntry(
                candidate=candidate,
                objective_vector=vec,
                iteration=iteration,
                evaluations=self._evaluations,
            )
        )

        if self.max_size is not None and len(self._entries) > self.max_size:
            self._truncate_to_max_size()

        hv = self.hypervolume()
        self._hv_history.append(
            HVPoint(
                iteration=iteration,
                evaluations=self._evaluations,
                hypervolume=hv,
                archive_size=len(self._entries),
            )
        )
        log.event(
            "archive_updated",
            iteration=iteration,
            evaluations=self._evaluations,
            archive_size=len(self._entries),
            evicted=n_evicted,
            hypervolume=round(hv, 6),
            candidate_id=candidate.sequence_id,
        )
        return True

    def _truncate_to_max_size(self) -> None:
        """Drop the least-contributing entries until the size cap is met."""
        while len(self._entries) > (self.max_size or len(self._entries)):
            contributions = self.hypervolume_contributions()
            worst = int(np.argmin(contributions))
            dropped = self._entries.pop(worst)
            log.debug("archive_truncated", dropped_id=dropped.candidate.sequence_id)

    # ── Hypervolume ───────────────────────────────────────────────────────────

    def hypervolume(self) -> float:
        """
        Hypervolume of the current front against the fixed reference point.

        Because objectives are normalised to [0, 1] and the reference point is
        ``1 + REF_EPS`` in every dimension, the result lies in
        ``[0, objective_set.max_hypervolume()]`` and is directly comparable
        between any two runs using the same objective set.
        """
        if not self._entries:
            return 0.0
        vecs = np.array([e.objective_vector for e in self._entries], dtype=float)
        ref = np.array(self.objective_set.reference_point(), dtype=float)
        return _hypervolume(vecs, ref)

    def normalised_hypervolume(self) -> float:
        """Hypervolume as a fraction of the theoretical maximum, in [0, 1]."""
        ceiling = self.objective_set.max_hypervolume()
        return self.hypervolume() / ceiling if ceiling > 0 else 0.0

    def hypervolume_contributions(self) -> list[float]:
        """Per-entry hypervolume contribution (used for crowding-based truncation)."""
        if len(self._entries) <= 1:
            return [self.hypervolume()]
        total = self.hypervolume()
        ref = np.array(self.objective_set.reference_point(), dtype=float)
        out: list[float] = []
        for i in range(len(self._entries)):
            subset = np.array(
                [e.objective_vector for j, e in enumerate(self._entries) if j != i],
                dtype=float,
            )
            out.append(total - _hypervolume(subset, ref))
        return out

    # Backwards-compatible alias.
    dominated_hypervolume_contribution = hypervolume_contributions

    # ── Queries ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(e.candidate for e in self._entries)

    @property
    def entries(self) -> list[ArchiveEntry]:
        """Read-only view of the archive entries."""
        return list(self._entries)

    def best_by(self, objective: str) -> Candidate | None:
        """Return the archive member with the best value on a single raw objective."""
        if not self._entries:
            return None
        maximize = objective in ("cai", "tai", "start_unpairing_prob", "ensemble_diversity")

        def _key(e: ArchiveEntry) -> float:
            v = getattr(e.candidate.scores, objective, None)
            return float(v) if v is not None else (-np.inf if maximize else np.inf)

        return (max if maximize else min)(self._entries, key=_key).candidate

    def hv_history(self) -> list[tuple[int, float]]:
        """Legacy ``(iteration, hypervolume)`` pairs."""
        return [(_hv_iteration(p), _hv_value(p)) for p in self._hv_history]

    def hv_trace(self) -> list[dict]:
        """Full convergence trace, including evaluations spent at each point."""
        return [
            p.to_dict()
            if isinstance(p, HVPoint)
            else HVPoint(_hv_iteration(p), 0, _hv_value(p), 0).to_dict()
            for p in self._hv_history
        ]

    def degenerate_objective_indices(self) -> list[int]:
        """
        Indices of objectives that are constant across the whole front.

        Returns an empty list for fronts of fewer than two points, where every
        axis is trivially constant and the notion carries no information.
        """
        if len(self._entries) < 2:
            return []
        return degenerate_objectives([e.objective_vector for e in self._entries])

    def degenerate_objective_labels(self) -> list[str]:
        """Labels of objectives that carry no information in this run."""
        labels = self.objective_set.labels
        return [labels[i] for i in self.degenerate_objective_indices()]

    def has_converged(self, window: int = 10, tol: float = 1e-4) -> bool:
        """
        True if hypervolume has not moved by more than ``tol`` over the last
        ``window`` archive updates.
        """
        if len(self._hv_history) < window:
            return False
        recent = [_hv_value(p) for p in self._hv_history[-window:]]
        return (max(recent) - min(recent)) < tol

    def to_dataframe(self):
        """Return a pandas DataFrame of all archive entries and their scores."""
        import pandas as pd

        rows = []
        for e in self._entries:
            rows.append(
                {
                    "sequence_id": e.candidate.sequence_id,
                    "iteration": e.iteration,
                    "evaluations": e.evaluations,
                    "seed_strategy": e.candidate.seed_strategy,
                    **e.candidate.scores.summary_dict(),
                }
            )
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        """Save the archive to a JSON Lines file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for e in self._entries:
                f.write(
                    json.dumps(
                        {
                            "sequence_id": e.candidate.sequence_id,
                            "iteration": e.iteration,
                            "evaluations": e.evaluations,
                            "sequence": e.candidate.sequence,
                            "protein": e.candidate.protein,
                            "cds_start": e.candidate.cds_start,
                            "cds_end": e.candidate.cds_end,
                            "scores": e.candidate.scores.model_dump(),
                            "objective_set": self.objective_set.name,
                            "objective_labels": self.objective_set.labels,
                            "objective_vector": e.objective_vector,
                            "legacy_objective_vector": e.candidate.scores.to_objective_vector(),
                            "added_at": e.added_at,
                        }
                    )
                    + "\n"
                )
        log.event("archive_saved", path=str(path), size=len(self._entries))


# ── History helpers ───────────────────────────────────────────────────────────
#
# _hv_history normally holds HVPoint records, but older code (and tests that
# inject a synthetic convergence trace) append plain (iteration, hv) tuples.
# These accessors tolerate both rather than crashing on attribute access.


def _hv_value(point) -> float:
    return float(point.hypervolume if isinstance(point, HVPoint) else point[1])


def _hv_iteration(point) -> int:
    return int(point.iteration if isinstance(point, HVPoint) else point[0])


# ── Pareto dominance ──────────────────────────────────────────────────────────


def _dominates(a, b) -> bool:
    """
    True if ``a`` dominates ``b`` under the all-minimise convention:
    no worse on every objective, and strictly better on at least one.

    Raises if the vectors differ in length. The previous implementation used
    ``zip``, which truncates to the shorter input — so a 9-element vector
    compared against a 10-element one silently ignored the tenth objective.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Objective vectors must have equal length for dominance comparison, "
            f"got {len(a)} and {len(b)}. This usually means two candidates were "
            f"scored against different objective sets."
        )
    at_least_one_better = False
    for ai, bi in zip(a, b, strict=True):
        if ai > bi:
            return False
        if ai < bi:
            at_least_one_better = True
    return at_least_one_better


def _hypervolume(vecs: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume of a point set against ``ref``, via pymoo where available."""
    if vecs.size == 0:
        return 0.0
    if _PYMOO:
        return float(HV(ref_point=ref)(vecs))
    return float(_monte_carlo_hv(vecs, ref))


def _monte_carlo_hv(vecs: np.ndarray, ref: np.ndarray, n_samples: int = 200_000) -> float:
    """
    Monte-Carlo hypervolume fallback for installs without pymoo.

    Samples uniformly in the box [0, ref] and estimates the dominated fraction.
    The previous fallback silently collapsed to the first two objectives, which
    reported a *different quantity* depending on whether pymoo was installed.
    """
    rng = np.random.default_rng(0)
    box = np.prod(ref)
    if box <= 0:
        return 0.0
    pts = rng.random((n_samples, len(ref))) * ref
    dominated = np.zeros(n_samples, dtype=bool)
    for v in vecs:
        dominated |= np.all(pts >= v, axis=1)
    return float(box * dominated.mean())
