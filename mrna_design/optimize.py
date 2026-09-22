"""
Main optimisation loop.

optimize() is the central function of the pipeline.  It:
  1. Seeds a population from the protein sequence.
  2. Evaluates all initial candidates with compute_all().
  3. Runs the refinement loop:
       for each iteration:
         for each candidate in population:
           compute diagnostics
           controller.propose_edits()
           apply_edit_proposal()
           re-score new candidate
           update Pareto archive
  4. Stops when archive has converged or budget is exhausted.

All intermediate results are logged as JSON to the run log directory.

Usage::

    from mrna_design.optimize import optimize, OptimizeConfig

    archive = optimize(
        protein="MVSK...",
        controller=RuleBasedController(),
        config=OptimizeConfig(max_iters=50, population_k=10),
    )
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from mrna_design.controller.base import BaseController
from mrna_design.controller.pareto_archive import ParetoArchive
from mrna_design.designer.population import Population, seed_population
from mrna_design.metrics.aggregator import Thresholds, compute_all
from mrna_design.models.candidate import Candidate
from mrna_design.models.edits import EditProposal
from mrna_design.validators.codon_applicator import apply_edit_proposal
from mrna_design import logging_utils
from mrna_design.logging_utils import get_logger

log = get_logger("optimize")


@dataclass
class OptimizeConfig:
    """Configuration for one optimisation run."""

    max_iters: int = 100
    population_k: int = 20
    max_edits_per_iter: int = 5
    rng_seed: int = 42
    include_lineardesign: bool = True
    convergence_window: int = 15
    convergence_tol: float = 1e-4
    # Metric computation flags
    run_safety: bool = True
    run_structure: bool = True       # set False for fast dry runs (skips ViennaRNA)
    run_rnahybrid: bool = False
    run_blast: bool = False
    thresholds: Thresholds = field(default_factory=Thresholds)
    # I/O
    output_dir: Path | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


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
        Run configuration. Defaults to OptimizeConfig().
    progress_callback : callable | None
        Called after each iteration with (iteration, archive).

    Returns
    -------
    ParetoArchive
        The final non-dominated archive.
    """
    cfg = config or OptimizeConfig()

    # ── Configure logging ─────────────────────────────────────────────────────
    if cfg.output_dir:
        run_dir = Path(cfg.output_dir) / cfg.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        logging_utils.configure(run_dir / "logs")
    else:
        run_dir = None

    log.event(
        "optimize_start",
        run_id=cfg.run_id,
        controller=controller.name,
        max_iters=cfg.max_iters,
        population_k=cfg.population_k,
        protein_length=len(protein),
    )
    t_start = time.time()

    # ── Seed population ────────────────────────────────────────────────────────
    log.event("seeding_population", k=cfg.population_k)
    pop = seed_population(
        protein=protein,
        utr5=utr5,
        utr3=utr3,
        k=cfg.population_k,
        rng_seed=cfg.rng_seed,
        include_lineardesign=cfg.include_lineardesign,
    )

    # ── Initial scoring ────────────────────────────────────────────────────────
    archive = ParetoArchive()
    log.event("initial_scoring", n_candidates=len(pop))
    for candidate in pop:
        scored = _score_candidate(candidate, cfg)
        pop.update(scored)
        archive.update(scored, iteration=0)

    log.score(
        "initial_archive",
        size=len(archive),
        hypervolume=archive.hypervolume(),
    )

    # ── Optimisation loop ─────────────────────────────────────────────────────
    history: list[EditProposal] = []
    controller.reset()

    for iteration in range(1, cfg.max_iters + 1):
        iter_start = time.time()
        n_improved = 0

        for candidate in list(pop):
            # 1. Compute diagnostics
            scored = _score_candidate(candidate, cfg)

            # 2. Controller proposes edits
            proposal = controller.propose_edits(
                candidate=scored,
                diagnostics=scored.diagnostics,
                history=history[-20:],    # only pass recent history
                max_edits=cfg.max_edits_per_iter,
                iteration=iteration,
            )
            history.append(proposal)

            # 3. Apply and validate edits
            new_candidate, edit_records = apply_edit_proposal(scored, proposal)

            # 4. Score the new candidate (only if edits were accepted)
            if any(r.accepted for r in edit_records):
                new_candidate = _score_candidate(new_candidate, cfg)
                added = archive.update(new_candidate, iteration=iteration)
                if added:
                    n_improved += 1
                    pop.update(new_candidate)

        wall_ms = round((time.time() - iter_start) * 1000, 1)
        log.event(
            "iteration_done",
            iteration=iteration,
            improved=n_improved,
            archive_size=len(archive),
            hypervolume=round(archive.hypervolume(), 6),
            wall_ms=wall_ms,
        )

        if progress_callback:
            progress_callback(iteration, archive)

        # Convergence check
        if archive.has_converged(window=cfg.convergence_window, tol=cfg.convergence_tol):
            log.event("converged", iteration=iteration)
            break

    # ── Save ──────────────────────────────────────────────────────────────────
    if run_dir:
        archive.save(run_dir / "archive.jsonl")
        _save_hv_history(archive, run_dir / "hv_history.json")

    log.event(
        "optimize_done",
        run_id=cfg.run_id,
        total_iterations=iteration,
        final_archive_size=len(archive),
        final_hypervolume=round(archive.hypervolume(), 6),
        total_wall_s=round(time.time() - t_start, 2),
    )
    return archive


# ── Helpers ────────────────────────────────────────────────────────────────────

def _score_candidate(candidate: Candidate, cfg: OptimizeConfig) -> Candidate:
    """Compute all objectives and update the candidate."""
    scores, diagnostics = compute_all(
        candidate,
        thresholds=cfg.thresholds,
        run_safety=cfg.run_safety,
        run_structure=cfg.run_structure,
        run_rnahybrid=cfg.run_rnahybrid,
        run_blast=cfg.run_blast,
    )
    return candidate.model_copy(update={"scores": scores, "diagnostics": diagnostics})


def _save_hv_history(archive: ParetoArchive, path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            [{"iteration": it, "hypervolume": hv} for it, hv in archive.hv_history()],
            f,
            indent=2,
        )
