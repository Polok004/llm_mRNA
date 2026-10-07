"""
Metrics aggregator — computes all objectives and raises structured diagnostics.

compute_all(candidate) → (ObjectiveScores, list[RegionDiagnostic])

This is the single entry point called by the main optimisation loop and agents.
It calls each metrics module and packages results into typed objects.

Thresholds for diagnostic flagging are defined here and can be overridden via
the Thresholds dataclass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from mrna_design.logging_utils import get_logger
from mrna_design.metrics.cai import cai, gc3_content, gc_content, sliding_window_cai
from mrna_design.metrics.immunogenicity import compute_immunogenicity, cpg_density_windows
from mrna_design.metrics.safety import compute_safety
from mrna_design.metrics.structure import ensemble_fold, fold_windows, start_codon_unpairing
from mrna_design.models.candidate import Candidate
from mrna_design.models.diagnostics import IssueType, Region, RegionDiagnostic, Severity
from mrna_design.models.objectives import ObjectiveScores
from mrna_design.validators.sequence_validator import check_no_uorfs

log = get_logger("metrics.aggregator")


@dataclass
class Thresholds:
    """
    Adjustable thresholds for raising RegionDiagnostics.
    Defaults are conservative; tighten for publication-quality benchmarks.
    """

    # Structure
    mfe_hairpin_5utr: float = -10.0  # kcal/mol — stable hairpin in 5'UTR
    local_mfe_low: float = -25.0  # kcal/mol per 240-nt window
    ensemble_diversity_low: float = 5.0  # below this → over-compacted
    start_unpairing_low: float = 0.5  # below this → AUG too paired
    # Translation
    cai_desert: float = 0.4  # sliding-window CAI below this
    # Immunogenicity
    cpg_hotspot: float = 3.0  # CpG per 100 nt — above this = hotspot
    upa_hotspot: float = 4.0
    gu_motif_high: int = 3  # absolute count
    # Safety
    mirna_seed_max: int = 0  # 0 = any seed hit is flagged


DEFAULT_THRESHOLDS = Thresholds()


def compute_all(
    candidate: Candidate,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    run_safety: bool = True,
    run_structure: bool = True,
    run_rnahybrid: bool = False,
    run_blast: bool = False,
) -> tuple[ObjectiveScores, list[RegionDiagnostic]]:
    """
    Compute all objectives and produce structured diagnostics for `candidate`.

    Parameters
    ----------
    candidate : Candidate
        The candidate to evaluate (scores and diagnostics fields are ignored).
    thresholds : Thresholds
        Thresholds controlling when diagnostics are raised.
    run_safety : bool
        Whether to run the miRNA seed scanner.
    run_structure : bool
        Whether to run ViennaRNA structure computations (slow). Set False for
        fast integration tests and dry runs.
    run_rnahybrid : bool
        Whether to run RNAhybrid (slow — disabled by default).
    run_blast : bool
        Whether to run BLAST+ (requires local DB — disabled by default).

    Returns
    -------
    (ObjectiveScores, list[RegionDiagnostic])
    """
    t0 = time.time()
    seq = candidate.sequence
    cds = candidate.cds
    diagnostics: list[RegionDiagnostic] = []

    # ── CAI & GC ─────────────────────────────────────────────────────────────
    cai_score = cai(cds)
    gc = gc_content(cds)
    gc3 = gc3_content(cds)

    # CAI codon deserts
    for window in sliding_window_cai(cds, window_codons=30, step_codons=10):
        if window["cai"] < thresholds.cai_desert:
            nt_start = candidate.cds_start + window["codon_start"] * 3
            nt_end = candidate.cds_start + window["codon_end"] * 3
            diagnostics.append(
                RegionDiagnostic(
                    region=Region.CDS,
                    issue=IssueType.CODON_DESERT,
                    severity=Severity.MEDIUM,
                    metric=window["cai"],
                    threshold=thresholds.cai_desert,
                    window_start=nt_start,
                    window_end=nt_end,
                    codon_start=window["codon_start"],
                    codon_end=window["codon_end"] - 1,
                    detail=f"Sliding-window CAI={window['cai']:.3f}",
                    source_agent="aggregator",
                )
            )

    # uORFs
    uorf_result = check_no_uorfs(candidate.utr5)
    uorf_count = 0 if uorf_result.passed else uorf_result.reason.count(",") + 1
    if not uorf_result.passed:
        diagnostics.append(
            RegionDiagnostic(
                region=Region.UTR5,
                issue=IssueType.UORF_DETECTED,
                severity=Severity.HIGH,
                metric=float(uorf_count),
                threshold=0.0,
                detail=uorf_result.reason,
                source_agent="aggregator",
            )
        )

    # ── Structure ─────────────────────────────────────────────────────────────
    mfe: float | None = None
    ens_div: float | None = None
    start_up: float | None = None

    if run_structure:
        ef = ensemble_fold(seq)
        mfe = ef.mfe
        ens_div = ef.ensemble_diversity

    mean_local_mfe: float | None = None

    if run_structure:
        # 5'UTR hairpin check
        if candidate.utr5:
            from mrna_design.metrics.structure import fold as _fold

            utr5_fold = _fold(candidate.utr5 + seq[candidate.cds_start : candidate.cds_start + 30])
            if utr5_fold.mfe < thresholds.mfe_hairpin_5utr:
                diagnostics.append(
                    RegionDiagnostic(
                        region=Region.UTR5,
                        issue=IssueType.STABLE_HAIRPIN,
                        severity=Severity.HIGH,
                        metric=utr5_fold.mfe,
                        threshold=thresholds.mfe_hairpin_5utr,
                        window_start=0,
                        window_end=len(candidate.utr5),
                        detail=f"5'UTR+start MFE={utr5_fold.mfe:.2f} kcal/mol",
                        source_agent="aggregator",
                    )
                )

        # Low ensemble diversity
        if ens_div is not None and ens_div < thresholds.ensemble_diversity_low and ens_div > 0:
            diagnostics.append(
                RegionDiagnostic(
                    region=Region.FULL,
                    issue=IssueType.LOW_ENSEMBLE_DIVERSITY,
                    severity=Severity.LOW,
                    metric=ens_div,
                    threshold=thresholds.ensemble_diversity_low,
                    detail=f"Ensemble diversity={ens_div:.2f}",
                    source_agent="aggregator",
                )
            )

        # Local MFE windows
        window_results = fold_windows(seq)
        for wr in window_results:
            if wr.mfe < thresholds.local_mfe_low:
                diagnostics.append(
                    RegionDiagnostic(
                        region=Region.CDS,
                        issue=IssueType.LOCAL_STABLE_STEM,
                        severity=Severity.MEDIUM,
                        metric=wr.mfe,
                        threshold=thresholds.local_mfe_low,
                        window_start=wr.window_start,
                        window_end=wr.window_end,
                        detail=f"Window MFE={wr.mfe:.2f} kcal/mol",
                        source_agent="aggregator",
                    )
                )

        mean_local_mfe = (
            sum(w.mfe for w in window_results) / len(window_results) if window_results else mfe
        )

        # AUG unpairing
        start_up = start_codon_unpairing(seq, candidate.cds_start)
        if start_up < thresholds.start_unpairing_low:
            diagnostics.append(
                RegionDiagnostic(
                    region=Region.JUNCTION_5_CDS,
                    issue=IssueType.START_CODON_PAIRED,
                    severity=Severity.HIGH,
                    metric=start_up,
                    threshold=thresholds.start_unpairing_low,
                    detail=f"AUG unpairing probability={start_up:.3f}",
                    source_agent="aggregator",
                )
            )

    # ── Immunogenicity ────────────────────────────────────────────────────────
    imm = compute_immunogenicity(seq)

    # CpG hotspot windows
    for window in cpg_density_windows(seq, window=100, step=25):
        if window["cpg_density"] > thresholds.cpg_hotspot:
            diagnostics.append(
                RegionDiagnostic(
                    region=Region.CDS,
                    issue=IssueType.CPG_HOTSPOT,
                    severity=Severity.MEDIUM,
                    metric=window["cpg_density"],
                    threshold=thresholds.cpg_hotspot,
                    window_start=window["start"],
                    window_end=window["end"],
                    detail=f"CpG density={window['cpg_density']:.2f}/100nt",
                    source_agent="aggregator",
                )
            )

    if imm.upa_density > thresholds.upa_hotspot:
        diagnostics.append(
            RegionDiagnostic(
                region=Region.FULL,
                issue=IssueType.UPA_HOTSPOT,
                severity=Severity.LOW,
                metric=imm.upa_density,
                threshold=thresholds.upa_hotspot,
                detail=f"UpA density={imm.upa_density:.2f}/100nt",
                source_agent="aggregator",
            )
        )

    if imm.gu_motif_count > thresholds.gu_motif_high:
        diagnostics.append(
            RegionDiagnostic(
                region=Region.FULL,
                issue=IssueType.GU_TLR_MOTIF,
                severity=Severity.MEDIUM,
                metric=float(imm.gu_motif_count),
                threshold=float(thresholds.gu_motif_high),
                detail=f"GU-rich TLR motif count={imm.gu_motif_count}",
                source_agent="aggregator",
            )
        )

    if imm.long_dsrna_count > 0:
        diagnostics.append(
            RegionDiagnostic(
                region=Region.FULL,
                issue=IssueType.LONG_DSRNA_STEM,
                severity=Severity.HIGH,
                metric=float(imm.long_dsrna_count),
                threshold=0.0,
                detail=f"Long dsRNA stems (≥40bp) count={imm.long_dsrna_count}",
                source_agent="aggregator",
            )
        )

    # ── Safety ────────────────────────────────────────────────────────────────
    mirna_hits = 0
    blast_count = 0
    if run_safety:
        safety = compute_safety(
            seq,
            candidate.cds_start,
            candidate.cds_end,
            utr3=candidate.utr3,
            run_rnahybrid_flag=run_rnahybrid,
            run_blast_flag=run_blast,
        )
        mirna_hits = safety.total_mirna_hits
        blast_count = safety.total_blast_hits

        for hit in safety.mirna_seed_hits:
            diagnostics.append(
                RegionDiagnostic(
                    region=Region.CDS,
                    issue=IssueType.MIRNA_SEED_MATCH,
                    severity=Severity.HIGH if hit.seed_type == "8mer" else Severity.MEDIUM,
                    metric=1.0,
                    threshold=0.0,
                    window_start=hit.target_start,
                    window_end=hit.target_end,
                    detail=f"{hit.mirna_id} ({hit.seed_type})",
                    source_agent="aggregator",
                    extra={"seed_type": hit.seed_type, "seed_sequence": hit.seed_sequence},
                )
            )

        for blast_hit in safety.blast_hits:
            diagnostics.append(
                RegionDiagnostic(
                    region=Region.FULL,
                    issue=IssueType.BLAST_HIT,
                    severity=Severity.MEDIUM,
                    metric=blast_hit.pct_identity,
                    threshold=80.0,
                    detail=(
                        f"{blast_hit.subject_id} ({blast_hit.pct_identity:.1f}% identity, "
                        f"e={blast_hit.evalue:.2e})"
                    ),
                    source_agent="aggregator",
                )
            )

    # ── Assemble ObjectiveScores ───────────────────────────────────────────────
    scores = ObjectiveScores(
        mfe=mfe,
        ensemble_diversity=ens_div,
        gc_content=gc,
        gc3_content=gc3,
        mean_local_mfe=mean_local_mfe,
        start_unpairing_prob=start_up,
        cai=cai_score,
        uorf_count=uorf_count,
        cpg_density=imm.cpg_density,
        upa_density=imm.upa_density,
        gu_motif_count=imm.gu_motif_count,
        uridine_fraction=imm.uridine_fraction,
        long_dsrna_count=imm.long_dsrna_count,
        mirna_seed_hits=mirna_hits,
        blast_hits=blast_count,
        computed_at=time.time(),
    )

    log.score(
        "all_objectives_computed",
        sequence_id=candidate.sequence_id,
        wall_ms=round((time.time() - t0) * 1000, 2),
        n_diagnostics=len(diagnostics),
        **{k: v for k, v in scores.summary_dict().items() if isinstance(v, (int, float))},
    )

    return scores, diagnostics
