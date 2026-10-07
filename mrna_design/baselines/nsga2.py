"""
NSGA-II controller — a genuine multi-objective baseline.

Why this was needed
-------------------
The existing GA baseline scalarises fitness to **CAI alone**
(``GeneticAlgorithmController._fitness``). Comparing a multi-objective agentic
controller against a single-objective GA and reporting the win in *hypervolume*
is not a fair test: the GA was never optimising the quantity it is being scored
on. Any advantage the agent shows could be entirely explained by the baseline
having been pointed at the wrong target.

NSGA-II (Deb et al., 2002) is the standard answer. It optimises the whole
objective vector at once using two ideas:

**Non-dominated sorting.** The population is partitioned into fronts. Front 0 is
the set of individuals nobody dominates; front 1 is what you get after removing
front 0; and so on. Lower front index is strictly better.

**Crowding distance.** Within a front no individual dominates another, so ties
are broken by preferring the one in the *emptier* neighbourhood — measured as
the perimeter of the cuboid spanned by its nearest neighbours on each axis.
This is what keeps the front spread out instead of collapsing into a cluster.

Selection is a binary tournament on ``(front rank, -crowding distance)``.

Fitness bookkeeping
-------------------
NSGA-II needs objective vectors, but a controller is not allowed to call
``compute_all`` itself — that would spend evaluation budget outside the loop's
accounting and break the very comparison this class exists to make fair.

Instead the controller learns lazily: every time the optimisation loop hands it
a scored candidate, it records that genome's objective vector. Selection runs
over the individuals whose fitness is known; genomes that have never been
evaluated are treated as unranked and are explored first. This makes it a
*steady-state* NSGA-II driven by the loop's evaluation stream, and every
evaluation it benefits from is charged to its budget like everyone else's.
"""

from __future__ import annotations

import random

import numpy as np

from mrna_design.controller.base import BaseController
from mrna_design.logging_utils import get_logger
from mrna_design.metrics.objective_spec import (
    DEFAULT_OBJECTIVE_SET,
    ObjectiveSet,
    get_objective_set,
)
from mrna_design.models.candidate import Candidate
from mrna_design.models.diagnostics import RegionDiagnostic
from mrna_design.models.edits import CodonEdit, EditProposal
from mrna_design.validators.codon_table import aa_synonyms, codon_to_aa

log = get_logger("baselines.nsga2")


def fast_non_dominated_sort(vectors: np.ndarray) -> list[int]:
    """
    Assign each row of ``vectors`` its Pareto front index (0 = best front).

    All objectives are minimised. This is the standard O(M N²) formulation from
    Deb et al. (2002), which is ample at the population sizes used here.
    """
    n = len(vectors)
    if n == 0:
        return []

    dominated_by: list[list[int]] = [[] for _ in range(n)]  # who i dominates
    domination_count = np.zeros(n, dtype=int)  # how many dominate i

    for i in range(n):
        for j in range(i + 1, n):
            i_dom_j = bool(np.all(vectors[i] <= vectors[j]) and np.any(vectors[i] < vectors[j]))
            j_dom_i = bool(np.all(vectors[j] <= vectors[i]) and np.any(vectors[j] < vectors[i]))
            if i_dom_j:
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif j_dom_i:
                dominated_by[j].append(i)
                domination_count[i] += 1

    ranks = np.full(n, -1, dtype=int)
    current = [i for i in range(n) if domination_count[i] == 0]
    front = 0
    while current:
        nxt: list[int] = []
        for i in current:
            ranks[i] = front
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        current = nxt
        front += 1
    return ranks.tolist()


def crowding_distance(vectors: np.ndarray) -> np.ndarray:
    """
    Crowding distance within one front (Deb et al., 2002).

    Boundary solutions get infinite distance so the extremes of the front are
    never discarded, which is what preserves the spread of trade-offs.
    """
    n, m = vectors.shape
    if n <= 2:
        return np.full(n, np.inf)

    distance = np.zeros(n)
    for obj in range(m):
        order = np.argsort(vectors[:, obj])
        lo, hi = vectors[order[0], obj], vectors[order[-1], obj]
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = hi - lo
        if span <= 0:
            continue  # degenerate axis contributes nothing
        for k in range(1, n - 1):
            prev_v = vectors[order[k - 1], obj]
            next_v = vectors[order[k + 1], obj]
            distance[order[k]] += (next_v - prev_v) / span
    return distance


class NSGA2Controller(BaseController):
    """
    Steady-state NSGA-II over synonymous codon choices.

    Parameters
    ----------
    pop_size
        Internal GA population size.
    mutation_rate
        Per-codon probability of a random synonymous substitution.
    crossover_rate
        Probability of performing uniform crossover rather than cloning.
    init_mutation_rate
        Per-codon mutation probability used to diversify the initial population.
    objective_set
        Objective space used for non-dominated sorting. Must match the archive's
        set, or the baseline would be optimising a different space from the one
        it is scored in.
    """

    name = "nsga2"

    def __init__(
        self,
        pop_size: int = 24,
        mutation_rate: float = 0.02,
        crossover_rate: float = 0.9,
        init_mutation_rate: float = 0.10,
        rng_seed: int = 42,
        objective_set: ObjectiveSet | str | None = None,
    ) -> None:
        self._pop_size = pop_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._init_mutation_rate = init_mutation_rate
        self._rng = random.Random(rng_seed)
        self.objective_set = (
            DEFAULT_OBJECTIVE_SET if objective_set is None else get_objective_set(objective_set)
        )
        self._population: list[tuple[str, ...]] = []
        self._fitness: dict[tuple[str, ...], list[float]] = {}

    def reset(self) -> None:
        self._population = []
        self._fitness = {}

    # ── Fitness bookkeeping ───────────────────────────────────────────────────

    def observe(self, candidate: Candidate) -> None:
        """
        Record the objective vector of a candidate the loop has already scored.

        Called at the top of every ``propose_edits``. The controller never
        triggers an evaluation itself, so it cannot spend budget off the books.
        """
        if not candidate.scores.is_complete():
            return
        genome = tuple(candidate.codons)
        self._fitness[genome] = self.objective_set.vector(candidate.scores, len(candidate.sequence))

    # ── GA operators ──────────────────────────────────────────────────────────

    def _synonyms(self, codon: str) -> list[str]:
        return [c for c in aa_synonyms(codon_to_aa(codon)) if c != codon]

    def _init_population(self, seed: tuple[str, ...]) -> None:
        self._population = [seed]
        while len(self._population) < self._pop_size:
            individual = list(seed)
            for i, codon in enumerate(individual):
                if self._rng.random() < self._init_mutation_rate:
                    alts = self._synonyms(codon)
                    if alts:
                        individual[i] = self._rng.choice(alts)
            self._population.append(tuple(individual))

    def _ranked(self) -> dict[tuple[str, ...], tuple[int, float]]:
        """Map each evaluated genome to ``(front rank, crowding distance)``."""
        known = [g for g in self._population if g in self._fitness]
        if not known:
            return {}
        vectors = np.array([self._fitness[g] for g in known], dtype=float)
        ranks = fast_non_dominated_sort(vectors)

        out: dict[tuple[str, ...], tuple[int, float]] = {}
        for front in set(ranks):
            members = [k for k, r in enumerate(ranks) if r == front]
            dist = crowding_distance(vectors[members])
            for local, global_idx in enumerate(members):
                out[known[global_idx]] = (front, float(dist[local]))
        return out

    def _tournament(self, ranked: dict[tuple[str, ...], tuple[int, float]]) -> tuple[str, ...]:
        """Binary tournament on (front rank asc, crowding distance desc)."""
        a, b = self._rng.choice(self._population), self._rng.choice(self._population)
        # An unevaluated genome is worth exploring, so it wins by default.
        ra, rb = ranked.get(a), ranked.get(b)
        if ra is None:
            return a
        if rb is None:
            return b
        if ra[0] != rb[0]:
            return a if ra[0] < rb[0] else b
        return a if ra[1] > rb[1] else b

    def _crossover(self, a: tuple[str, ...], b: tuple[str, ...]) -> list[str]:
        """Uniform crossover. Always synonymous-safe: both parents encode the same protein."""
        if self._rng.random() > self._crossover_rate or len(a) != len(b):
            return list(a)
        return [a[i] if self._rng.random() < 0.5 else b[i] for i in range(len(a))]

    def _mutate(self, genome: list[str]) -> list[str]:
        out = list(genome)
        for i, codon in enumerate(out):
            if self._rng.random() < self._mutation_rate:
                alts = self._synonyms(codon)
                if alts:
                    out[i] = self._rng.choice(alts)
        return out

    # ── Controller interface ──────────────────────────────────────────────────

    def propose_edits(
        self,
        candidate: Candidate,
        diagnostics: list[RegionDiagnostic],
        history: list[EditProposal],
        max_edits: int = 5,
        iteration: int = 0,
    ) -> EditProposal:
        self.observe(candidate)

        current = tuple(candidate.codons)
        if not self._population:
            self._init_population(current)
        elif current not in self._population:
            # Keep the loop's current candidate in the gene pool.
            self._population.append(current)
            if len(self._population) > self._pop_size * 2:
                ranked = self._ranked()
                self._population.sort(key=lambda g: ranked.get(g, (10**6, 0.0))[0])
                self._population = self._population[: self._pop_size]

        ranked = self._ranked()
        child = self._mutate(self._crossover(self._tournament(ranked), self._tournament(ranked)))
        self._population.append(tuple(child))

        # Express the offspring as edits relative to the current candidate. The
        # applicator still validates every one, so the GA cannot bypass the
        # protein-identity guarantee.
        differing = [
            i for i, (orig, new) in enumerate(zip(current, child, strict=True)) if orig != new
        ]
        self._rng.shuffle(differing)
        edits = [
            CodonEdit(
                codon_index=i,
                original_codon=current[i],
                new_codon=child[i],
                reason="NSGA-II offspring codon (rank + crowding selection)",
                targeting_issue=None,
            )
            for i in differing[:max_edits]
        ]

        front_sizes: dict[int, int] = {}
        for rank, _ in ranked.values():
            front_sizes[rank] = front_sizes.get(rank, 0) + 1

        log.event(
            "nsga2_proposal",
            iteration=iteration,
            n_edits=len(edits),
            population=len(self._population),
            evaluated=len(self._fitness),
            front0_size=front_sizes.get(0, 0),
        )
        return EditProposal(
            edits=edits,
            expected_improvement="NSGA-II multi-objective offspring",
            controller_type="nsga2",
            targeting_diagnostics=[],
            iteration=iteration,
            metadata={
                "population": len(self._population),
                "evaluated_genomes": len(self._fitness),
                "front0_size": front_sizes.get(0, 0),
                "objective_set": self.objective_set.name,
            },
        )
