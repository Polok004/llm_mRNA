"""
Population — a fixed-size collection of Candidates maintained during optimisation.

The population stores up to K candidates and prunes by hypervolume contribution
when over-full. It is separate from the ParetoArchive: the population is the
*working set* for the current iteration; the archive stores all non-dominated
candidates seen so far.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from mrna_design.designer.seeds import seed_all_strategies
from mrna_design.logging_utils import get_logger
from mrna_design.models.candidate import Candidate

log = get_logger("designer.population")


class Population:
    """
    Fixed-capacity collection of Candidates.

    Culling policy
    --------------
    When the population is over capacity, one member is dropped. This used to be
    "drop the lowest CAI", which quietly turned a multi-objective search into a
    CAI-biased one: a candidate could be excellent on stability, CpG content and
    start-codon accessibility and still be evicted for having a mediocre CAI,
    and no other objective ever influenced survival.

    The population now culls by **non-domination rank first, then crowding**:

      * A member dominated by many others in the current population is dropped
        first (it is Pareto-inferior, on every axis at once).
      * Ties are broken by dropping the most crowded member — the one whose
        nearest neighbours in normalised objective space are closest — which
        preserves spread along the front. This is the standard NSGA-II
        survival rule.

    Candidates that have not been scored yet are never culled, since their
    objective vectors are unknown and evicting them would bias the search toward
    whatever was evaluated first.
    """

    def __init__(self, max_size: int = 20, objective_set=None) -> None:
        from mrna_design.metrics.objective_spec import (
            DEFAULT_OBJECTIVE_SET,
            get_objective_set,
        )

        self.max_size = max_size
        self.objective_set = (
            DEFAULT_OBJECTIVE_SET if objective_set is None else get_objective_set(objective_set)
        )
        self._candidates: list[Candidate] = []

    def add(self, candidate: Candidate) -> None:
        """Add a candidate, culling by non-domination rank if over capacity."""
        if candidate.sequence_id in {c.sequence_id for c in self._candidates}:
            return  # Deduplicate
        self._candidates.append(candidate)
        while len(self._candidates) > self.max_size:
            victim = self._select_cull_victim()
            if victim is None:
                break
            dropped = self._candidates.pop(victim)
            log.debug("population_culled", dropped_id=dropped.sequence_id)

    def _select_cull_victim(self) -> int | None:
        """Index of the member to evict: most-dominated, then most-crowded."""
        import numpy as np

        scored = [(i, c) for i, c in enumerate(self._candidates) if c.scores.is_complete()]
        if len(scored) < 2:
            # Nothing comparable; fall back to evicting the oldest scored member,
            # or the oldest member overall if none are scored.
            return scored[0][0] if scored else 0

        idx = [i for i, _ in scored]
        vecs = np.array(
            [self.objective_set.vector(c.scores, len(c.sequence)) for _, c in scored],
            dtype=float,
        )

        # Domination count: how many population members beat this one outright.
        n = len(vecs)
        dominated_by = np.zeros(n, dtype=int)
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                if np.all(vecs[b] <= vecs[a]) and np.any(vecs[b] < vecs[a]):
                    dominated_by[a] += 1

        worst = dominated_by.max()
        contenders = np.flatnonzero(dominated_by == worst)
        if len(contenders) == 1:
            return idx[int(contenders[0])]

        # Tie-break on crowding: evict whoever sits in the densest neighbourhood.
        crowding = []
        for c in contenders:
            d = np.linalg.norm(vecs - vecs[c], axis=1)
            d[c] = np.inf
            crowding.append(d.min())
        return idx[int(contenders[int(np.argmin(crowding))])]

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
        return (
            max(scored, key=lambda c: getattr(c.scores, objective))
            if maximize
            else min(scored, key=lambda c: getattr(c.scores, objective))
        )

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
    from mrna_design.designer.seeds import SeedStrategy, seed_candidate

    slot = len(base_candidates)
    for i in range(k - len(base_candidates)):
        try:
            rng = random.Random(rng_seed + slot + i)
            strategy: SeedStrategy = "gc_balanced" if i % 2 == 0 else "harmonised"
            c = seed_candidate(protein, strategy=strategy, utr5=utr5, utr3=utr3, rng=rng)
            pop.add(c)
        except Exception as exc:
            log.error("population_fill_failed", slot=slot + i, error=str(exc))

    log.event("population_seeded", size=len(pop), k=k)
    return pop
