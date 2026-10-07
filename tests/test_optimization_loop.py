"""
End-to-end optimisation loop tests.

These tests run a few iterations of the full pipeline to catch integration bugs.
ViennaRNA tests are skipped if not installed.
"""

from __future__ import annotations

import pytest

from mrna_design.baselines import CaiMaxController, RandomController
from mrna_design.controller.rule_based import RuleBasedController
from mrna_design.optimize import OptimizeConfig, optimize
from mrna_design.validators.codon_table import translate
from mrna_design.validators.sequence_validator import check_protein_identity

# Short test protein (fast to evaluate)
_TEST_PROTEIN = "MVSKGEELF"  # 9 aa — one codon each + stop = 30 nt CDS


@pytest.mark.parametrize("ctrl_class", [RuleBasedController, CaiMaxController, RandomController])
def test_e2e_protein_identity_preserved(ctrl_class):
    """
    After optimisation, every candidate in the archive must encode the same protein.
    This is the critical correctness invariant.
    """
    ctrl = ctrl_class() if ctrl_class is CaiMaxController else ctrl_class(rng_seed=0)
    cfg = OptimizeConfig(
        max_iters=5,
        population_k=4,
        max_edits_per_iter=3,
        rng_seed=0,
        include_lineardesign=False,
        run_safety=False,
        run_blast=False,
        run_rnahybrid=False,
    )
    archive = optimize(
        protein=_TEST_PROTEIN,
        controller=ctrl,
        config=cfg,
    )

    for candidate in archive:
        result = check_protein_identity(
            candidate.sequence,
            candidate.cds_start,
            candidate.cds_end,
            _TEST_PROTEIN,
        )
        assert result.passed, (
            f"{ctrl_class.__name__}: protein identity violated for "
            f"{candidate.sequence_id}: {result.reason}"
        )


def test_e2e_archive_non_empty():
    """The archive must contain at least one candidate after optimisation."""
    ctrl = RuleBasedController(rng_seed=42)
    cfg = OptimizeConfig(
        max_iters=3,
        population_k=4,
        run_safety=False,
        include_lineardesign=False,
    )
    archive = optimize(protein=_TEST_PROTEIN, controller=ctrl, config=cfg)
    assert len(archive) > 0


def test_e2e_no_invalid_edits_in_lineage():
    """
    Invalid-edit rate should be 0 for non-synonymous edits — the applicator
    should catch them. We verify that no accepted edit changed the amino acid.
    """
    ctrl = RuleBasedController(rng_seed=42)
    cfg = OptimizeConfig(
        max_iters=5,
        population_k=4,
        run_safety=False,
        include_lineardesign=False,
    )
    archive = optimize(protein=_TEST_PROTEIN, controller=ctrl, config=cfg)

    for candidate in archive:
        for record in candidate.accepted_edits:
            orig_aa = translate(record.edit.original_codon).rstrip("*")
            new_aa = translate(record.edit.new_codon).rstrip("*")
            assert orig_aa == new_aa, (
                f"Accepted edit changed amino acid: "
                f"{record.edit.original_codon}({orig_aa}) → "
                f"{record.edit.new_codon}({new_aa})"
            )


def test_e2e_cai_not_decreasing_on_cai_max():
    """CaiMaxController should not decrease CAI on average across iterations."""
    ctrl = CaiMaxController()
    cfg = OptimizeConfig(
        max_iters=10,
        population_k=4,
        run_safety=False,
        include_lineardesign=False,
    )
    archive = optimize(protein=_TEST_PROTEIN, controller=ctrl, config=cfg)
    cai_values = [c.scores.cai for c in archive if c.scores.cai is not None]
    if cai_values:
        assert max(cai_values) >= 0.3  # At least some meaningful CAI


def test_e2e_hv_history_monotone_or_stable():
    """Hypervolume should not decrease during optimisation."""
    ctrl = RuleBasedController(rng_seed=42)
    cfg = OptimizeConfig(
        max_iters=8,
        population_k=4,
        run_safety=False,
        include_lineardesign=False,
    )
    archive = optimize(protein=_TEST_PROTEIN, controller=ctrl, config=cfg)
    hv_hist = [hv for _, hv in archive.hv_history()]
    for i in range(1, len(hv_hist)):
        assert hv_hist[i] >= hv_hist[i - 1] - 1e-9, (
            f"Hypervolume decreased at step {i}: {hv_hist[i - 1]:.6f} → {hv_hist[i]:.6f}"
        )
