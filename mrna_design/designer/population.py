"""
Population — a fixed-size collection of Candidates maintained during optimisation.

The population stores up to K candidates and prunes by hypervolume contribution
when over-full. It is separate from the ParetoArchive: the population is the
*working set* for the current iteration; the archive stores all non-dominated
candidates seen so far.
"""

from __future__ import annotations

import random
from typing import Iterator

from mrna_design.models.candidate import Candidate
from mrna_design.designer.seeds import seed_all_strategies
from mrna_design.logging_utils import get_logger

log = get_logger("designer.population")


class Population:
    """Fixed-capacity collection of Candidates."""

    def __init__(self, max_size: int = 20) -> None:
        self.max_size = max_size
        self._candidates: list[Candidate] = []

    def add(self, candidate: Candidate) -> None:
        """Add a candidate; drop the lowest-CAI candidate if over capacity."""
        if candidate.sequence_id in {c.sequence_id for c in self._candidates}:
            return  # Deduplicate
        self._candidates.append(candidate)
        if len(self._candidates) > self.max_size:
            # Simple culling: remove candidate with lowest CAI (proxy for quality)
            self._candidates.sort(
                key=lambda c: (c.scores.cai or 0.0),
                reverse=True,
            )
            dropped = self._candidates.pop()
            log.debug("population_culled", dropped_id=dropped.sequence_id)

    def update(self, candidate: Candidate) -> None:
        """Replace the candidate with the same sequence_id, or add if new."""
        for i, c in enumerate(self._candidates):
            if c.sequence_id == candidate.sequence_id:
                self._candidates[i] = candidate
                return
        self.add(candidate)

    def sample(self, k: int = 1, rng: random.Random | None = None) -> list[Candidate]:
        """Return up to k candidates sampled without replacement."""
        rng = rng or random.Random()
        n = min(k, len(self._candidates))
        return rng.sample(self._candidates, n)

    def best(self, objective: str = "cai") -> Candidate | None:
        """Return the candidate with the best value for a single objective."""
        scored = [c for c in self._candidates if getattr(c.scores, objective) is not None]
        if not scored:
            return None
        maximize = objective in ("cai", "tai", "start_unpairing_prob", "ensemble_diversity")
        return max(scored, key=lambda c: getattr(c.scores, objective)) if maximize else \
               min(scored, key=lambda c: getattr(c.scores, objective))

    def __len__(self) -> int:
        return len(self._candidates)

    def __iter__(self) -> Iterator[Candidate]:
        return iter(self._candidates)

    def __repr__(self) -> str:
        return f"Population(size={len(self._candidates)}/{self.max_size})"


def seed_population(
    protein: str,
    utr5: str = "",
    utr3: str = "",
    k: int = 20,
    rng_seed: int = 42,
    include_lineardesign: bool = True,
) -> Population:
    """
    Seed a Population of size K.

    Seeds from all four strategies first (up to 4 candidates).
    Then fills remaining slots with GC-balanced random samples using different
    random seeds.
    """
    pop = Population(max_size=k)

    # Seed from all strategies (up to 4 candidates)
    base_candidates = seed_all_strategies(
        protein,
        utr5=utr5,
        utr3=utr3,
        rng_seed=rng_seed,
        include_lineardesign=include_lineardesign,
    )
    for c in base_candidates:
        pop.add(c)

    # Fill remaining slots with harmonised samples (different seeds)
    from mrna_design.designer.seeds import seed_candidate
    slot = len(base_candidates)
    for i in range(k - len(base_candidates)):
        try:
            rng = random.Random(rng_seed + slot + i)
            strategy = "gc_balanced" if i % 2 == 0 else "harmonised"
            c = seed_candidate(protein, strategy=strategy, utr5=utr5, utr3=utr3, rng=rng)
            pop.add(c)
        except Exception as exc:
            log.error("population_fill_failed", slot=slot + i, error=str(exc))

    log.event("population_seeded", size=len(pop), k=k)
    return pop
