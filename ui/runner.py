"""
Background run management for the Streamlit UI.

Why this exists
---------------
A single optimisation run takes minutes: ViennaRNA folding costs roughly two
seconds per evaluation for a 700-nucleotide construct, so a 500-evaluation run
is a quarter of an hour. Streamlit re-executes the whole script on every
interaction, so the optimisation cannot run on the main thread or the page
freezes and the user has no idea whether anything is happening.

The run therefore goes on a worker thread, and the UI polls a plain
:class:`RunState` object. Nothing in the worker touches Streamlit's API, which
is what makes this safe — Streamlit's session state is not thread-safe, but an
ordinary Python object guarded by a lock is.

Cancellation uses the progress callback that ``optimize()`` already accepts: the
callback raises :class:`RunCancelled`, which unwinds the loop. The callback also
receives the live archive, so the partial front survives a cancel and the user
keeps whatever the run found before they stopped it.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mrna_design.controllers import build_controller
from mrna_design.metrics.objective_spec import get_objective_set


class RunCancelled(RuntimeError):
    """Raised inside the progress callback to unwind a running optimisation."""


#: The 20 standard amino acids, single-letter codes.
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def validate_protein(protein: str) -> str | None:
    """
    Return a human-readable problem with ``protein``, or None if it is usable.

    Worth doing up front: the seeding code catches per-strategy failures and
    logs them, so an unusable sequence otherwise produces an empty population,
    an empty archive, and a run that reports "finished" having done nothing.
    A silent success is the worst possible answer to a typo.
    """
    seq = (protein or "").strip().upper()
    if not seq:
        return "No protein sequence given."

    body = seq.rstrip("*")
    if not body:
        return "The sequence contains only stop characters."

    bad = sorted(set(body) - VALID_AMINO_ACIDS)
    if bad:
        shown = ", ".join(repr(c) for c in bad[:6])
        more = f" (and {len(bad) - 6} more)" if len(bad) > 6 else ""
        return (
            f"Not valid amino-acid codes: {shown}{more}. "
            f"Use single-letter codes from {''.join(sorted(VALID_AMINO_ACIDS))}. "
            f"If you pasted a DNA or RNA sequence, this field wants the protein."
        )
    if "*" in body:
        return "A stop character '*' appears mid-sequence; it may only end it."
    if len(body) < 3:
        return f"Too short to optimise ({len(body)} residues)."
    return None


@dataclass
class RunConfig:
    """Everything the UI collects before launching a run."""

    protein: str
    protein_name: str = "custom"
    utr5: str = ""
    utr3: str = ""
    controller: str = "rule_based"
    objective_set: str = "primary"
    max_evaluations: int = 200
    max_iters: int = 200
    population_k: int = 8
    max_edits_per_iter: int = 5
    rng_seed: int = 42
    run_structure: bool = True
    run_safety: bool = True
    include_lineardesign: bool = False
    empty_proposal_policy: str = "noop"
    llm_model: str = "gpt-4o"
    output_dir: Path = Path("results/ui")


@dataclass
class RunState:
    """
    Mutable, thread-safe-ish view of a run in progress.

    The worker writes; the UI thread reads. Every field is either an immutable
    value or replaced wholesale, so a torn read is impossible for the scalars
    and the lock covers the rest.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "idle"  # idle | running | done | cancelled | error
    config: RunConfig | None = None

    started_at: float | None = None
    finished_at: float | None = None
    iteration: int = 0
    evaluations: int = 0
    max_evaluations: int | None = None
    archive_size: int = 0
    hypervolume: float = 0.0
    normalised_hypervolume: float = 0.0

    archive: Any = None  # ParetoArchive, once the first callback fires
    hv_trace: list[dict] = field(default_factory=list)
    error: str | None = None
    traceback: str | None = None

    _cancel: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ── Control ───────────────────────────────────────────────────────────────

    def request_cancel(self) -> None:
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at

    @property
    def progress(self) -> float:
        """Fraction of the evaluation budget spent, clamped to [0, 1]."""
        if not self.max_evaluations:
            return 0.0
        return min(1.0, self.evaluations / self.max_evaluations)

    def snapshot_from(self, archive: Any, iteration: int) -> None:
        """Copy the live numbers out of the archive. Called from the worker."""
        with self._lock:
            self.archive = archive
            self.iteration = iteration
            self.archive_size = len(archive)
            self.hypervolume = archive.hypervolume()
            self.normalised_hypervolume = archive.normalised_hypervolume()
            self.hv_trace = archive.hv_trace()
            budget = getattr(archive, "run_budget", None)
            # run_budget is only attached when the run finishes, so during the
            # run we read the archive's own evaluation counter instead.
            self.evaluations = budget.spent if budget else archive.evaluations


def start_run(cfg: RunConfig) -> RunState:
    """Launch an optimisation on a worker thread and return its live state."""
    state = RunState(config=cfg, max_evaluations=cfg.max_evaluations)
    thread = threading.Thread(target=_worker, args=(cfg, state), daemon=True)
    state.status = "running"
    state.started_at = time.time()
    thread.start()
    return state


def _worker(cfg: RunConfig, state: RunState) -> None:
    """Thread body. Must never touch the Streamlit API."""
    from mrna_design.optimize import OptimizeConfig, optimize

    def progress(iteration: int, archive: Any) -> None:
        state.snapshot_from(archive, iteration)
        if state.cancel_requested:
            raise RunCancelled()

    try:
        problem = validate_protein(cfg.protein)
        if problem is not None:
            raise ValueError(problem)

        controller = build_controller(
            cfg.controller,
            rng_seed=cfg.rng_seed,
            llm_model=cfg.llm_model,
            objective_set=cfg.objective_set,
        )
        opt_cfg = OptimizeConfig(
            max_evaluations=cfg.max_evaluations,
            max_iters=cfg.max_iters,
            population_k=cfg.population_k,
            max_edits_per_iter=cfg.max_edits_per_iter,
            rng_seed=cfg.rng_seed,
            objective_set=get_objective_set(cfg.objective_set),
            empty_proposal_policy=cfg.empty_proposal_policy,
            include_lineardesign=cfg.include_lineardesign,
            run_structure=cfg.run_structure,
            run_safety=cfg.run_safety,
            output_dir=Path(cfg.output_dir),
            run_id=state.run_id,
        )
        archive = optimize(
            protein=cfg.protein,
            controller=controller,
            utr5=cfg.utr5,
            utr3=cfg.utr3,
            config=opt_cfg,
            progress_callback=progress,
        )
        if len(archive) == 0:
            raise RuntimeError(
                "The run finished with an empty archive \u2014 no candidate could be "
                "seeded or scored. Check the protein sequence and the metric "
                "toggles."
            )
        state.snapshot_from(archive, state.iteration)
        state.status = "done"

    except RunCancelled:
        # The archive captured by the last callback is still valid; keep it.
        state.status = "cancelled"

    except Exception as exc:  # noqa: BLE001 - surfaced in the UI, not swallowed
        state.status = "error"
        state.error = f"{type(exc).__name__}: {exc}"
        state.traceback = traceback.format_exc()

    finally:
        state.finished_at = time.time()


# ── Reading results back ──────────────────────────────────────────────────────


def archive_rows(archive: Any) -> list[dict]:
    """Flatten a Pareto archive into table rows for display."""
    if archive is None:
        return []
    rows = []
    for entry in archive.entries:
        c = entry.candidate
        s = c.scores
        rows.append(
            {
                "id": c.sequence_id[:10],
                "CAI": s.cai,
                "MFE": s.mfe,
                "MFE/nt": (s.mfe / len(c.sequence)) if s.mfe is not None else None,
                "GC": s.gc_content,
                "CpG/100nt": s.cpg_density,
                "UpA/100nt": s.upa_density,
                "U fraction": s.uridine_fraction,
                "AUG unpaired": s.start_unpairing_prob,
                "miRNA hits": s.mirna_seed_hits,
                "GU motifs": s.gu_motif_count,
                "seed": c.seed_strategy,
                "iteration": entry.iteration,
                "evals": entry.evaluations,
                "edits": len(c.accepted_edits),
                "_sequence_id": c.sequence_id,
            }
        )
    return rows


def candidate_by_id(archive: Any, sequence_id: str):
    """Find a candidate in the archive by its full sequence id."""
    if archive is None:
        return None
    for entry in archive.entries:
        if entry.candidate.sequence_id == sequence_id:
            return entry.candidate
    return None


def to_fasta(archive: Any, protein_name: str = "design") -> str:
    """Render the whole front as a FASTA file, 60 characters per line."""
    if archive is None:
        return ""
    chunks = []
    for i, entry in enumerate(archive.entries, start=1):
        c = entry.candidate
        s = c.scores
        # Build the header field by field; implicit concatenation inside a
        # conditional expression binds the whole string, not just the last part.
        header = f">{protein_name}_{i:03d}|{c.sequence_id[:10]}"
        if s.cai is not None:
            header += f"|CAI={s.cai:.4f}"
        if s.mfe is not None:
            header += f"|MFE={s.mfe:.1f}"
        if s.cpg_density is not None:
            header += f"|CpG={s.cpg_density:.2f}"
        seq = c.sequence
        body = "\n".join(seq[j : j + 60] for j in range(0, len(seq), 60))
        chunks.append(f"{header}\n{body}")
    return "\n".join(chunks) + "\n"


def discover_runs(results_root: Path) -> list[dict]:
    """
    Find completed runs under ``results_root`` by reading their manifests.

    A run directory is any directory containing a ``manifest.json``. Older runs
    without one are skipped rather than guessed at.
    """
    import json

    root = Path(results_root)
    if not root.exists():
        return []
    runs = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            m = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        result = m.get("result", {})
        budget = m.get("budget", {})
        stats = m.get("stats", {})
        env = m.get("environment", {})
        runs.append(
            {
                "run_id": m.get("run_id", manifest_path.parent.name),
                "controller": m.get("controller"),
                "objectives": m.get("objective_set", {}).get("name"),
                "HV": result.get("hypervolume"),
                "% ceiling": (result.get("normalised_hypervolume") or 0) * 100,
                "front": result.get("archive_size"),
                "evals": budget.get("evaluations_spent"),
                "cache hit": budget.get("cache_hit_rate"),
                "stop": stats.get("stop_reason"),
                "ViennaRNA": env.get("external_tools", {}).get("viennarna_python_bindings"),
                "degenerate": ", ".join(result.get("degenerate_objectives") or []),
                "wall s": m.get("wall_seconds"),
                "_path": str(manifest_path.parent),
            }
        )
    return runs
