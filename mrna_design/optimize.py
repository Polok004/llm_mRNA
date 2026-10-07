"""
Main optimisation loop.

``optimize()`` is the central function of the pipeline. It:

  1. Seeds a population from the protein sequence.
  2. Evaluates the initial candidates.
  3. Runs the refinement loop until the evaluation budget is spent, the
     iteration cap is reached, or the archive converges:

       for each iteration:
         for each candidate in the population:
           controller.propose_edits()       -> EditProposal
           apply_edit_proposal()            -> validated new Candidate
           score the new candidate          -> charged against the budget
           update the Pareto archive

  4. Writes the archive, the convergence trace and a run manifest.

What is enforced here
---------------------
**Budget, not iterations, is the unit of comparison.** Every full objective
evaluation is charged to an :class:`~mrna_design.budget.EvaluationBudget`.
Repeat evaluations of an identical sequence are served from a cache and are not
charged, because they cannot produce new information. The original loop
re-scored every unchanged population member on every iteration: 45% of its
evaluations were duplicates, which both wasted work and made "same number of
iterations" a meaningless notion of fairness.

**Fallback behaviour is uniform across controllers.** Previously the rule-based
controller had three internal layers of fallback and always emitted at least one
edit, while the LLM controller returned an empty proposal whenever no diagnostic
fired. Those controllers therefore consumed different amounts of budget for
reasons unrelated to the quality of their reasoning — a confound sitting
directly underneath the headline comparison. Empty proposals are now handled by
the loop, identically for every controller, under an explicit policy, and are
counted and reported.

Usage::

    from mrna_design.optimize import optimize, OptimizeConfig

    archive = optimize(
        protein="MVSK...",
        controller=RuleBasedController(),
        config=OptimizeConfig(max_evaluations=2000, population_k=10),
    )
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mrna_design import logging_utils
from mrna_design.budget import BudgetExhausted, EvaluationBudget, ScoreCache
from mrna_design.controller.base import BaseController
from mrna_design.controller.pareto_archive import ParetoArchive
from mrna_design.designer.population import seed_population
from mrna_design.logging_utils import get_logger
from mrna_design.metrics.aggregator import Thresholds, compute_all
from mrna_design.metrics.objective_spec import (
    DEFAULT_OBJECTIVE_SET,
    ObjectiveSet,
    get_objective_set,
)
from mrna_design.models.candidate import Candidate
from mrna_design.models.edits import EditProposal
from mrna_design.validators.codon_applicator import apply_edit_proposal

log = get_logger("optimize")

EmptyProposalPolicy = Literal["noop", "random"]


@dataclass
class OptimizeConfig:
    """Configuration for one optimisation run."""

    # ── Budget ────────────────────────────────────────────────────────────────
    max_evaluations: int | None = None
    """
    Hard cap on full objective evaluations. This is the fair unit of comparison
    between controllers and should be set explicitly for any benchmark. ``None``
    falls back to the iteration cap alone (dry runs and tests only).
    """
    max_iters: int = 100
    """Safety cap on iterations. The budget is normally the binding constraint."""

    population_k: int = 20
    max_edits_per_iter: int = 5
    rng_seed: int = 42
    include_lineardesign: bool = True
    convergence_window: int = 15
    convergence_tol: float = 1e-4

    # ── Objective space ───────────────────────────────────────────────────────
    objective_set: ObjectiveSet | str = DEFAULT_OBJECTIVE_SET
    """Which normalised objective space the archive and hypervolume live in."""

    # ── Controller fairness ───────────────────────────────────────────────────
    empty_proposal_policy: EmptyProposalPolicy = "noop"
    """
    What the loop does when a controller proposes no edits.

    ``"noop"``  — accept it, count it, move on. The controller chose to stop, and
                  that choice costs it nothing but also gains it nothing.
    ``"random"``— substitute a single random synonymous edit, so an idle
                  controller still explores.

    Whichever is chosen applies to *every* controller in the run, which is the
    point: the policy is a property of the experiment, not of the controller.
    """

    # ── Metric computation flags ──────────────────────────────────────────────
    run_safety: bool = True
    run_structure: bool = True  # set False for fast dry runs (skips ViennaRNA)
    run_rnahybrid: bool = False
    run_blast: bool = False
    thresholds: Thresholds = field(default_factory=Thresholds)

    # ── I/O ───────────────────────────────────────────────────────────────────
    output_dir: Path | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def resolved_objective_set(self) -> ObjectiveSet:
        return get_objective_set(self.objective_set)


@dataclass
class RunStats:
    """Bookkeeping for one run, written into the manifest."""

    iterations: int = 0
    proposals: int = 0
    empty_proposals: int = 0
    edits_proposed: int = 0
    edits_accepted: int = 0
    edits_rejected: int = 0
    archive_admissions: int = 0
    stop_reason: str = "unknown"

    @property
    def edit_acceptance_rate(self) -> float:
        return self.edits_accepted / self.edits_proposed if self.edits_proposed else 0.0

    def summary(self) -> dict:
        return {
            "iterations": self.iterations,
            "proposals": self.proposals,
            "empty_proposals": self.empty_proposals,
            "empty_proposal_rate": round(self.empty_proposals / self.proposals, 4)
            if self.proposals
            else 0.0,
            "edits_proposed": self.edits_proposed,
            "edits_accepted": self.edits_accepted,
            "edits_rejected": self.edits_rejected,
            "edit_acceptance_rate": round(self.edit_acceptance_rate, 4),
            "archive_admissions": self.archive_admissions,
            "stop_reason": self.stop_reason,
        }


def optimize(
    protein: str,
    controller: BaseController,
    utr5: str = "",
    utr3: str = "",
    config: OptimizeConfig | None = None,
    progress_callback: Callable[[int, ParetoArchive], None] | None = None,
) -> ParetoArchive:
    """
    Run the agentic mRNA optimisation loop.

    Parameters
    ----------
    protein : str
        Amino acid sequence (single-letter, uppercase).
    controller : BaseController
        The refinement controller to use.
    utr5 / utr3 : str
        UTR sequences (optional).
    config : OptimizeConfig
        Run configuration. Defaults to ``OptimizeConfig()``.
    progress_callback : callable | None
        Called after each iteration with ``(iteration, archive)``.

    Returns
    -------
    ParetoArchive
        The final non-dominated archive. Budget and run statistics are attached
        as ``archive.run_budget`` and ``archive.run_stats``.
    """
    cfg = config or OptimizeConfig()
    oset = cfg.resolved_objective_set()

    # ── Configure logging ─────────────────────────────────────────────────────
    if cfg.output_dir:
        run_dir = Path(cfg.output_dir) / cfg.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        logging_utils.configure(run_dir / "logs")
    else:
        run_dir = None

    budget = EvaluationBudget(max_evaluations=cfg.max_evaluations)
    cache = ScoreCache()
    stats = RunStats()
    archive = ParetoArchive(objective_set=oset)

    log.event(
        "optimize_start",
        run_id=cfg.run_id,
        controller=controller.name,
        objective_set=oset.name,
        n_objectives=len(oset),
        max_evaluations=cfg.max_evaluations,
        max_iters=cfg.max_iters,
        population_k=cfg.population_k,
        empty_proposal_policy=cfg.empty_proposal_policy,
        protein_length=len(protein),
    )
    t_start = time.time()

    def score(candidate: Candidate) -> Candidate:
        """Score a candidate, charging the budget only on a cache miss."""
        cached = cache.get(candidate.sequence_id)
        if cached is not None:
            budget.note_cache_hit()
            scores, diagnostics = cached
        else:
            budget.charge(1)  # raises BudgetExhausted at the cap
            scores, diagnostics = compute_all(
                candidate,
                thresholds=cfg.thresholds,
                run_safety=cfg.run_safety,
                run_structure=cfg.run_structure,
                run_rnahybrid=cfg.run_rnahybrid,
                run_blast=cfg.run_blast,
            )
            cache.put(candidate.sequence_id, (scores, diagnostics))
        archive.note_evaluations(budget.spent - archive.evaluations)
        return candidate.model_copy(update={"scores": scores, "diagnostics": diagnostics})

    # ── Seed population ───────────────────────────────────────────────────────
    log.event("seeding_population", k=cfg.population_k)
    pop = seed_population(
        protein=protein,
        utr5=utr5,
        utr3=utr3,
        k=cfg.population_k,
        rng_seed=cfg.rng_seed,
        include_lineardesign=cfg.include_lineardesign,
    )
    pop.objective_set = oset  # so culling is multi-objective, not CAI-only

    # ── Initial scoring ───────────────────────────────────────────────────────
    log.event("initial_scoring", n_candidates=len(pop))
    iteration = 0
    try:
        for candidate in list(pop):
            scored = score(candidate)
            pop.update(scored)
            if archive.update(scored, iteration=0):
                stats.archive_admissions += 1
    except BudgetExhausted:
        stats.stop_reason = "budget_exhausted_during_seeding"
        log.warn("budget_exhausted_during_seeding", spent=budget.spent)

    log.score(
        "initial_archive",
        size=len(archive),
        hypervolume=archive.hypervolume(),
        normalised_hypervolume=archive.normalised_hypervolume(),
        evaluations=budget.spent,
    )

    # ── Optimisation loop ─────────────────────────────────────────────────────
    history: list[EditProposal] = []
    controller.reset()

    for iteration in range(1, cfg.max_iters + 1):
        if budget.exhausted:
            stats.stop_reason = "budget_exhausted"
            log.event("budget_exhausted", iteration=iteration, spent=budget.spent)
            break

        iter_start = time.time()
        n_improved = 0

        for candidate in list(pop):
            if budget.exhausted:
                break

            # The population already carries scores and diagnostics; scoring it
            # again would return the identical result. The old loop re-scored it
            # anyway, every iteration, for every member.
            scored = candidate if candidate.scores.is_complete() else score(candidate)

            # 1. Controller proposes edits.
            proposal = controller.propose_edits(
                candidate=scored,
                diagnostics=scored.diagnostics,
                history=history[-20:],  # only recent history, to bound prompt size
                max_edits=cfg.max_edits_per_iter,
                iteration=iteration,
            )
            stats.proposals += 1

            # 2. Apply the uniform empty-proposal policy.
            if not proposal.edits:
                stats.empty_proposals += 1
                proposal = _apply_empty_proposal_policy(proposal, scored, cfg, iteration)
                if not proposal.edits:
                    continue

            history.append(proposal)
            stats.edits_proposed += len(proposal.edits)

            # 3. Validate and apply. The applicator is the only code that
            #    touches nucleotides; it rejects anything non-synonymous.
            new_candidate, edit_records = apply_edit_proposal(scored, proposal)
            accepted = [r for r in edit_records if r.accepted]
            stats.edits_accepted += len(accepted)
            stats.edits_rejected += len(edit_records) - len(accepted)

            if not accepted:
                continue

            # 4. Score the new candidate and offer it to the archive.
            try:
                new_candidate = score(new_candidate)
            except BudgetExhausted:
                stats.stop_reason = "budget_exhausted"
                log.event("budget_exhausted", iteration=iteration, spent=budget.spent)
                break

            if archive.update(new_candidate, iteration=iteration):
                stats.archive_admissions += 1
                n_improved += 1
            # Offer to the population regardless: the population is the working
            # set and benefits from diversity, while the archive stays strict.
            pop.update(new_candidate)

        stats.iterations = iteration
        log.event(
            "iteration_done",
            iteration=iteration,
            improved=n_improved,
            archive_size=len(archive),
            hypervolume=round(archive.hypervolume(), 6),
            evaluations=budget.spent,
            cache_hits=budget.served_from_cache,
            wall_ms=round((time.time() - iter_start) * 1000, 1),
        )

        if progress_callback:
            progress_callback(iteration, archive)

        if archive.has_converged(window=cfg.convergence_window, tol=cfg.convergence_tol):
            stats.stop_reason = "converged"
            log.event("converged", iteration=iteration, evaluations=budget.spent)
            break
    else:
        stats.stop_reason = stats.stop_reason if stats.stop_reason != "unknown" else "max_iters"

    if stats.stop_reason == "unknown":
        stats.stop_reason = "budget_exhausted" if budget.exhausted else "max_iters"

    # Attach bookkeeping so callers and the manifest can read it back.
    archive.run_budget = budget
    archive.run_stats = stats
    archive.run_cache = cache

    # ── Save ──────────────────────────────────────────────────────────────────
    if run_dir:
        archive.save(run_dir / "archive.jsonl")
        _save_hv_history(archive, run_dir / "hv_history.json")
        _write_manifest(
            run_dir / "manifest.json",
            cfg,
            controller,
            archive,
            budget,
            cache,
            stats,
            protein,
            time.time() - t_start,
        )

    log.event(
        "optimize_done",
        run_id=cfg.run_id,
        total_iterations=stats.iterations,
        stop_reason=stats.stop_reason,
        final_archive_size=len(archive),
        final_hypervolume=round(archive.hypervolume(), 6),
        normalised_hypervolume=round(archive.normalised_hypervolume(), 6),
        evaluations_spent=budget.spent,
        cache_hit_rate=cache.summary()["hit_rate"],
        empty_proposal_rate=stats.summary()["empty_proposal_rate"],
        degenerate_objectives=archive.degenerate_objective_labels(),
        total_wall_s=round(time.time() - t_start, 2),
    )
    return archive


# ── Helpers ───────────────────────────────────────────────────────────────────


def _apply_empty_proposal_policy(
    proposal: EditProposal,
    candidate: Candidate,
    cfg: OptimizeConfig,
    iteration: int,
) -> EditProposal:
    """
    Handle a controller that proposed nothing, identically for every controller.

    Keeping this in the loop rather than inside individual controllers is what
    makes the budget comparison fair: no controller gets a private fallback that
    quietly buys it extra exploration.
    """
    if cfg.empty_proposal_policy == "noop":
        log.debug("empty_proposal_noop", iteration=iteration, controller=proposal.controller_type)
        return proposal

    # "random": substitute exactly one random synonymous edit.
    import random

    from mrna_design.models.edits import CodonEdit
    from mrna_design.validators.codon_table import aa_synonyms, codon_to_aa

    rng = random.Random(cfg.rng_seed + iteration)
    eligible = [
        (i, c)
        for i, c in enumerate(candidate.codons)
        if len([s for s in aa_synonyms(codon_to_aa(c)) if s != c]) > 0
    ]
    if not eligible:
        return proposal
    idx, codon = rng.choice(eligible)
    alts = [s for s in aa_synonyms(codon_to_aa(codon)) if s != codon]
    log.debug(
        "empty_proposal_random_substitute", iteration=iteration, controller=proposal.controller_type
    )
    return proposal.model_copy(
        update={
            "edits": [
                CodonEdit(
                    codon_index=idx,
                    original_codon=codon,
                    new_codon=rng.choice(alts),
                    reason="Loop-level fallback for an empty proposal (uniform across controllers)",
                    targeting_issue=None,
                )
            ]
        }
    )


def _save_hv_history(archive: ParetoArchive, path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(archive.hv_trace(), f, indent=2)


def _write_manifest(
    path: Path,
    cfg: OptimizeConfig,
    controller: BaseController,
    archive: ParetoArchive,
    budget: EvaluationBudget,
    cache: ScoreCache,
    stats: RunStats,
    protein: str,
    wall_s: float,
) -> None:
    """Write a complete, self-describing record of the run."""
    import json

    from mrna_design.provenance import environment_manifest

    oset = archive.objective_set
    manifest = {
        "run_id": cfg.run_id,
        "controller": controller.name,
        "protein_length": len(protein),
        "wall_seconds": round(wall_s, 2),
        "config": {
            "max_evaluations": cfg.max_evaluations,
            "max_iters": cfg.max_iters,
            "population_k": cfg.population_k,
            "max_edits_per_iter": cfg.max_edits_per_iter,
            "rng_seed": cfg.rng_seed,
            "empty_proposal_policy": cfg.empty_proposal_policy,
            "include_lineardesign": cfg.include_lineardesign,
            "run_safety": cfg.run_safety,
            "run_structure": cfg.run_structure,
            "run_rnahybrid": cfg.run_rnahybrid,
            "run_blast": cfg.run_blast,
            "convergence_window": cfg.convergence_window,
            "convergence_tol": cfg.convergence_tol,
        },
        "objective_set": {
            "name": oset.name,
            "description": oset.description,
            "n_objectives": len(oset),
            "objectives": oset.describe(),
            "reference_point": oset.reference_point(),
            "max_hypervolume": oset.max_hypervolume(),
        },
        "result": {
            "archive_size": len(archive),
            "hypervolume": archive.hypervolume(),
            "normalised_hypervolume": archive.normalised_hypervolume(),
            "degenerate_objectives": archive.degenerate_objective_labels(),
        },
        "budget": budget.summary(),
        "cache": cache.summary(),
        "stats": stats.summary(),
        "environment": environment_manifest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.event("manifest_written", path=str(path))
