"""
Baselines — non-agent optimisers for comparison.

All baselines implement the BaseController interface so they can be swapped in
and benchmarked by the same evaluation harness.

1. CaiMaxController    — "optimise" by always choosing the max-CAI codon
                         (single-pass, deterministic)
2. RandomController    — random synonymous substitutions (budget-matched)
3. GeneticAlgorithm    — simple tournament + crossover GA on codon arrays
"""

from __future__ import annotations

import random
import math

from mrna_design.controller.base import BaseController
from mrna_design.models.candidate import Candidate
from mrna_design.models.diagnostics import RegionDiagnostic
from mrna_design.models.edits import CodonEdit, EditProposal
from mrna_design.validators.codon_table import (
    HUMAN_FREQUENCIES,
    aa_synonyms,
    codon_to_aa,
)
from mrna_design.logging_utils import get_logger

log = get_logger("baselines")


# ── CAI-max controller ────────────────────────────────────────────────────────

class CaiMaxController(BaseController):
    """
    Greedy CAI-maximising controller.

    On each call: replace the `max_edits` codons with the lowest human usage
    with their highest-usage synonyms. Ignores diagnostics.
    This is the simplest possible optimiser and a natural baseline.
    """

    name = "cai_max"

    def propose_edits(
        self,
        candidate: Candidate,
        diagnostics: list[RegionDiagnostic],
        history: list[EditProposal],
        max_edits: int = 5,
        iteration: int = 0,
    ) -> EditProposal:
        indexed = [
            (i, c) for i, c in enumerate(candidate.codons)
            if aa_synonyms(codon_to_aa(c))
        ]
        # Sort ascending by human frequency (lowest first — most to gain)
        indexed.sort(key=lambda x: HUMAN_FREQUENCIES.get(x[1], 0.0))

        edits: list[CodonEdit] = []
        used: set[int] = set()
        for idx, codon in indexed:
            if len(edits) >= max_edits:
                break
            if idx in used:
                continue
            aa = codon_to_aa(codon)
            synonyms = sorted(
                aa_synonyms(aa),
                key=lambda c: HUMAN_FREQUENCIES.get(c, 0.0),
                reverse=True,
            )
            best = next((c for c in synonyms if c != codon), None)
            if best and HUMAN_FREQUENCIES.get(best, 0.0) > HUMAN_FREQUENCIES.get(codon, 0.0):
                edits.append(CodonEdit(
                    codon_index=idx,
                    original_codon=codon,
                    new_codon=best,
                    reason="CAI-max greedy replacement",
                    targeting_issue=None,
                ))
                used.add(idx)

        if not edits:
            # All codons are already at max frequency — nothing to do
            # Return a trivial edit that the applicator may accept or reject
            for idx, codon in enumerate(candidate.codons):
                aa = codon_to_aa(codon)
                alts = [c for c in aa_synonyms(aa) if c != codon]
                if alts:
                    edits = [CodonEdit(
                        codon_index=idx,
                        original_codon=codon,
                        new_codon=alts[0],
                        reason="CAI-max: already optimal, exploring synonym",
                    )]
                    break

        return EditProposal(
            edits=edits or [_dummy_edit(candidate)],
            expected_improvement="Increase CAI",
            controller_type="rule_based",   # GA/random use same type slot
            targeting_diagnostics=[],
            iteration=iteration,
        )


# ── Random synonymous search ───────────────────────────────────────────────────

class RandomController(BaseController):
    """
    Random synonymous substitution controller.

    Proposes `max_edits` random synonymous replacements, ignoring diagnostics.
    Budget-matched with other controllers for fair comparison.
    """

    name = "random"

    def __init__(self, rng_seed: int = 42) -> None:
        self._rng = random.Random(rng_seed)

    def propose_edits(
        self,
        candidate: Candidate,
        diagnostics: list[RegionDiagnostic],
        history: list[EditProposal],
        max_edits: int = 5,
        iteration: int = 0,
    ) -> EditProposal:
        eligible = [
            (i, c) for i, c in enumerate(candidate.codons)
            if len(aa_synonyms(codon_to_aa(c))) > 1
        ]
        if not eligible:
            return EditProposal(
                edits=[_dummy_edit(candidate)],
                expected_improvement="random",
                controller_type="random",
                iteration=iteration,
            )
        chosen = self._rng.sample(eligible, min(max_edits, len(eligible)))
        edits = []
        for idx, codon in chosen:
            aa = codon_to_aa(codon)
            alts = [c for c in aa_synonyms(aa) if c != codon]
            new = self._rng.choice(alts)
            edits.append(CodonEdit(
                codon_index=idx,
                original_codon=codon,
                new_codon=new,
                reason="Random synonymous substitution",
            ))
        return EditProposal(
            edits=edits,
            expected_improvement="random exploration",
            controller_type="random",
            targeting_diagnostics=[],
            iteration=iteration,
        )


# ── Genetic Algorithm controller ──────────────────────────────────────────────

class GeneticAlgorithmController(BaseController):
    """
    Simple GA-based controller.

    Maintains an internal population of codon arrays.  Each call to propose_edits()
    performs one GA step (selection + crossover + mutation) and returns the
    edits that transform the parent candidate into the GA offspring.

    Parameters
    ----------
    pop_size : int
        GA population size.
    mutation_rate : float
        Per-codon probability of a random synonymous mutation.
    crossover_rate : float
        Probability of single-point crossover.
    """

    name = "ga"

    def __init__(
        self,
        pop_size: int = 20,
        mutation_rate: float = 0.02,
        crossover_rate: float = 0.7,
        rng_seed: int = 42,
    ) -> None:
        self._pop_size = pop_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._rng = random.Random(rng_seed)
        self._population: list[list[str]] | None = None  # list of codon arrays

    def reset(self) -> None:
        self._population = None

    def _init_population(self, seed_codons: list[str]) -> None:
        """Initialise GA population by mutating the seed codon array."""
        self._population = [list(seed_codons)]
        while len(self._population) < self._pop_size:
            individual = list(seed_codons)
            for i in range(len(individual)):
                if self._rng.random() < 0.1:   # 10% random mutation at init
                    aa = codon_to_aa(individual[i])
                    alts = [c for c in aa_synonyms(aa) if c != individual[i]]
                    if alts:
                        individual[i] = self._rng.choice(alts)
            self._population.append(individual)

    def _fitness(self, codons: list[str]) -> float:
        """Simple fitness: CAI as a scalar (GA optimises CAI only; multi-obj done by archive)."""
        log_sum = 0.0
        n = 0
        for c in codons:
            aa = codon_to_aa(c)
            if aa in ("M", "W", "*"):
                continue
            w = HUMAN_FREQUENCIES.get(c, 1e-10)
            log_sum += math.log(max(w, 1e-10))
            n += 1
        return math.exp(log_sum / n) if n > 0 else 0.0

    def _tournament(self, k: int = 3) -> list[str]:
        pool = self._rng.choices(self._population, k=k)
        return max(pool, key=self._fitness)

    def _crossover(self, a: list[str], b: list[str]) -> list[str]:
        if self._rng.random() > self._crossover_rate or len(a) < 2:
            return list(a)
        point = self._rng.randint(1, len(a) - 1)
        return a[:point] + b[point:]

    def _mutate(self, codons: list[str]) -> list[str]:
        result = list(codons)
        for i in range(len(result)):
            if self._rng.random() < self._mutation_rate:
                aa = codon_to_aa(result[i])
                alts = [c for c in aa_synonyms(aa) if c != result[i]]
                if alts:
                    result[i] = self._rng.choice(alts)
        return result

    def propose_edits(
        self,
        candidate: Candidate,
        diagnostics: list[RegionDiagnostic],
        history: list[EditProposal],
        max_edits: int = 5,
        iteration: int = 0,
    ) -> EditProposal:
        seed_codons = list(candidate.codons)

        if self._population is None:
            self._init_population(seed_codons)

        # One GA step: produce next generation
        new_pop = []
        while len(new_pop) < self._pop_size:
            p1 = self._tournament()
            p2 = self._tournament()
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            new_pop.append(child)
        self._population = new_pop

        # Best individual
        best = max(self._population, key=self._fitness)

        # Convert diff between current candidate codons and best to edits
        edits: list[CodonEdit] = []
        for i, (orig, new) in enumerate(zip(seed_codons, best)):
            if orig != new and len(edits) < max_edits:
                edits.append(CodonEdit(
                    codon_index=i,
                    original_codon=orig,
                    new_codon=new,
                    reason="GA offspring codon substitution",
                    targeting_issue=None,
                ))

        if not edits:
            return EditProposal(
                edits=[_dummy_edit(candidate)],
                expected_improvement="ga_step",
                controller_type="ga",
                iteration=iteration,
            )

        return EditProposal(
            edits=edits,
            expected_improvement="GA-guided CAI improvement",
            controller_type="ga",
            targeting_diagnostics=[],
            iteration=iteration,
            metadata={"ga_fitness": round(self._fitness(best), 4)},
        )


# ── Utility ────────────────────────────────────────────────────────────────────

def _dummy_edit(candidate: Candidate) -> CodonEdit:
    """Return a trivial (but valid) edit for when nothing better is available."""
    for i, codon in enumerate(candidate.codons):
        aa = codon_to_aa(codon)
        alts = [c for c in aa_synonyms(aa) if c != codon]
        if alts:
            return CodonEdit(
                codon_index=i,
                original_codon=codon,
                new_codon=alts[0],
                reason="Dummy edit: no better option found",
            )
    raise ValueError("No synonymous edits possible for this candidate.")
