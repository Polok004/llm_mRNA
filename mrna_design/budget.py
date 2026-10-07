"""
Evaluation budget — the unit of fairness for the benchmark.

Why this exists
---------------
The project's research question is phrased as:

    "Does an LLM controller ... reach a better Pareto front than non-LLM
     optimisers **under the same tool-call budget**?"

Nothing in the pipeline used to count tool calls, so "the same budget" was never
enforced or even measured. Runs were capped by ``max_iters``, which is not a
fair unit: controllers differ in how many edits they propose per call, how often
they propose nothing at all, and therefore how many candidates get scored per
iteration. Two controllers could run for 100 iterations each and spend wildly
different amounts of the expensive resource.

The expensive resource is a full objective evaluation — ``compute_all()``, which
folds the sequence with ViennaRNA, scans miRNA seeds and computes every metric.
That is what this module counts.

Two things follow:

* **A cache is part of the accounting, not an optimisation.** Re-scoring a
  sequence that has already been scored produces an identical result, so it must
  not be charged again. Measured on the original loop, 45% of all evaluations
  were duplicates of sequences already scored — inflating apparent cost and
  hiding real differences between controllers.

* **HV must be reported against budget spent**, not iteration index, so that
  convergence curves for different controllers are on a common x-axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mrna_design.logging_utils import get_logger

log = get_logger("budget")


class BudgetExhausted(RuntimeError):
    """Raised when a run attempts to spend past its evaluation budget."""


@dataclass
class EvaluationBudget:
    """
    A counter for full objective evaluations, with an optional hard cap.

    Attributes
    ----------
    max_evaluations
        Hard cap. ``None`` means unlimited (useful for tests and dry runs).
    spent
        Evaluations actually charged (cache misses only).
    served_from_cache
        Evaluations avoided by the cache. Reported so the saving is visible and
        auditable rather than silently improving the numbers.
    """

    max_evaluations: int | None = None
    spent: int = 0
    served_from_cache: int = 0

    @property
    def remaining(self) -> float:
        if self.max_evaluations is None:
            return float("inf")
        return max(0, self.max_evaluations - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    @property
    def fraction_used(self) -> float:
        if self.max_evaluations is None:
            return 0.0
        return self.spent / self.max_evaluations

    def charge(self, n: int = 1) -> None:
        """Charge ``n`` evaluations, raising if that would exceed the cap."""
        if self.max_evaluations is not None and self.spent + n > self.max_evaluations:
            raise BudgetExhausted(
                f"Evaluation budget exhausted: {self.spent}/{self.max_evaluations} spent, "
                f"requested {n} more."
            )
        self.spent += n

    def note_cache_hit(self, n: int = 1) -> None:
        self.served_from_cache += n

    def summary(self) -> dict:
        return {
            "max_evaluations": self.max_evaluations,
            "evaluations_spent": self.spent,
            "served_from_cache": self.served_from_cache,
            "cache_hit_rate": round(
                self.served_from_cache / max(1, self.spent + self.served_from_cache), 4
            ),
            "fraction_used": round(self.fraction_used, 4),
        }


@dataclass
class ScoreCache:
    """
    Memoises objective evaluations by sequence identity.

    Keyed on ``Candidate.sequence_id`` (a SHA-256 prefix of the sequence), so
    two candidates with identical nucleotides share one evaluation. This is
    sound because ``compute_all`` is a pure function of the sequence and the
    threshold configuration — the cache is scoped to a single run, which holds
    thresholds fixed.
    """

    _store: dict[str, tuple] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, sequence_id: str):
        value = self._store.get(sequence_id)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, sequence_id: str, value: tuple) -> None:
        self._store[sequence_id] = value

    def __len__(self) -> int:
        return len(self._store)

    def summary(self) -> dict:
        total = self.hits + self.misses
        return {
            "distinct_sequences": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
