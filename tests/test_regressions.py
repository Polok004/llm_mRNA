"""
Regression tests for defects found in the pre-upgrade audit.

Every test in this file fails against the original implementation. Each one
names the defect it pins down, so a future change that reintroduces the bug
fails with an explanation rather than a bare assertion error.
"""

from __future__ import annotations

import pytest

from mrna_design.budget import BudgetExhausted, EvaluationBudget, ScoreCache
from mrna_design.controller.pareto_archive import ParetoArchive, _dominates
from mrna_design.designer.seeds import seed_candidate
from mrna_design.metrics import safety as safety_mod
from mrna_design.metrics.immunogenicity import reverse_complement
from mrna_design.metrics.objective_spec import EXTENDED, PRIMARY
from mrna_design.models.objectives import ObjectiveScores

PROTEIN = "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"


@pytest.fixture
def candidate():
    return seed_candidate(PROTEIN, strategy="cai_max")


def _scored(candidate, **kwargs):
    base = dict(
        cai=0.8,
        gc_content=0.55,
        cpg_density=2.0,
        uridine_fraction=0.22,
        start_unpairing_prob=0.6,
        mfe=-250.0,
    )
    base.update(kwargs)
    return candidate.model_copy(update={"scores": ObjectiveScores(**base)})


# ── Bug 1: miRNA seed double-counting ─────────────────────────────────────────


class TestMirnaSeedDoubleCounting:
    """
    A single 8mer site used to be reported twice: once as an "8mer" (anchored at
    the A1 position) and again as a "7mer-m8" (anchored one base later). The
    deduplication key included seed_type and target_start, so neither field
    matched and both survived. Every 8mer inflated the miRNA objective by 2x.
    """

    @pytest.fixture(autouse=True)
    def _one_mirna(self, monkeypatch):
        monkeypatch.setattr(safety_mod, "_MIRNA_CACHE", {"hsa-miR-TEST": "UGAGGUAGUAGGUUGUAUAGUU"})

    def _site(self) -> str:
        return reverse_complement("UGAGGUAGUAGGUUGUAUAGUU"[1:8])

    def test_one_8mer_site_yields_exactly_one_hit(self):
        target = "GGGG" + "A" + self._site() + "GGGG"
        hits = safety_mod.scan_mirna_seeds(target)
        assert len(hits) == 1, (
            f"One 8mer site must produce one hit, got "
            f"{[(h.seed_type, h.target_start) for h in hits]}"
        )
        assert hits[0].seed_type == "8mer"

    def test_site_without_a1_is_a_7mer_m8(self):
        target = "GGGG" + "C" + self._site() + "GGGG"
        hits = safety_mod.scan_mirna_seeds(target)
        assert len(hits) == 1
        assert hits[0].seed_type == "7mer-m8"

    def test_stringency_filter_excludes_weaker_classes(self):
        target = "GGGG" + "C" + self._site() + "GGGG"  # a 7mer-m8 site
        assert safety_mod.scan_mirna_seeds(target, min_seed_type="8mer") == []

    def test_invalid_stringency_is_rejected(self):
        with pytest.raises(ValueError, match="min_seed_type"):
            safety_mod.scan_mirna_seeds("ACGU", min_seed_type="6mer")


# ── Bug 2: ragged objective vectors ───────────────────────────────────────────


class TestObjectiveVectorLength:
    """
    ``to_objective_vector`` appended a tenth element only when a surrogate score
    was present. Candidates scored with and without the surrogate produced
    vectors of different lengths, and the ``zip``-based dominance check silently
    compared only the shorter prefix — so the surrogate objective was ignored in
    exactly the comparisons it was supposed to influence.
    """

    def test_legacy_vector_length_is_constant(self):
        without = ObjectiveScores(cai=0.8, gc_content=0.5, cpg_density=1.0)
        with_surrogate = ObjectiveScores(
            cai=0.8, gc_content=0.5, cpg_density=1.0, surrogate_degradation=0.3
        )
        assert len(without.to_objective_vector()) == len(with_surrogate.to_objective_vector()) == 10

    def test_normalised_vector_length_follows_the_objective_set(self):
        scores = ObjectiveScores(cai=0.8, gc_content=0.5, cpg_density=1.0)
        assert len(scores.normalised_vector(1000, PRIMARY)) == len(PRIMARY)
        assert len(scores.normalised_vector(1000, EXTENDED)) == len(EXTENDED)

    def test_dominance_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            _dominates([0.1, 0.2], [0.1, 0.2, 0.3])


# ── Bug 3: surrogate confidence normalised within the batch ───────────────────


class TestSurrogateConfidence:
    """
    Confidence was ``1 - std / std.max()`` over the rows of a single call. For a
    one-row call that is always exactly 0.0, and the same candidate scored
    differently depending on what it was batched with.
    """

    @pytest.fixture(scope="class")
    def model(self):
        from mrna_design.surrogate.model import SurrogateModel
        from mrna_design.surrogate.openvaccine import load_openvaccine

        m = SurrogateModel(n_estimators=40)
        m.fit(load_openvaccine(), cv_folds=0)
        return m

    def test_single_prediction_confidence_is_not_zero(self, model, candidate):
        assert model.predict(candidate).confidence > 0.0

    def test_confidence_is_independent_of_batch_composition(self, model, candidate):
        alone = model.predict(candidate).confidence
        other = seed_candidate(PROTEIN, strategy="harmonised")
        batched = model.predict_batch([candidate, other])[0].confidence
        assert alone == pytest.approx(batched, abs=1e-9)

    def test_confidence_is_bounded(self, model, candidate):
        assert 0.0 <= model.predict(candidate).confidence <= 1.0


# ── Bug 4: controller mislabelling ────────────────────────────────────────────


def test_cai_max_controller_labels_itself_correctly(candidate):
    """CaiMaxController emitted controller_type='rule_based', so every edit it
    made was attributed to a different controller in the logs and lineage."""
    from mrna_design.baselines import CaiMaxController

    proposal = CaiMaxController().propose_edits(candidate, [], [], max_edits=3)
    assert proposal.controller_type == "cai_max"


# ── Hypervolume comparability ─────────────────────────────────────────────────


class TestHypervolumeComparability:
    """
    HV used a reference point derived from the worst value seen *within a run*
    (``global_worst + 0.1``), so two runs were measured against two different
    boxes and their HVs could not legitimately be compared — which is exactly
    what the benchmark's Mann-Whitney test did.
    """

    def test_reference_point_is_fixed_and_independent_of_history(self, candidate):
        a = ParetoArchive(objective_set=PRIMARY)
        b = ParetoArchive(objective_set=PRIMARY)
        a.update(_scored(candidate, cai=0.9))
        for _ in range(3):
            b.update(_scored(candidate, cai=0.9))
        assert a.objective_set.reference_point() == b.objective_set.reference_point()

    def test_identical_fronts_give_identical_hypervolume(self, candidate):
        a, b = ParetoArchive(PRIMARY), ParetoArchive(PRIMARY)
        a.update(_scored(candidate, cai=0.9, cpg_density=1.0))
        b.update(_scored(candidate, cai=0.9, cpg_density=1.0))
        assert a.hypervolume() == pytest.approx(b.hypervolume())

    def test_hypervolume_is_bounded_by_the_objective_set_ceiling(self, candidate):
        archive = ParetoArchive(PRIMARY)
        archive.update(
            _scored(
                candidate,
                cai=1.0,
                cpg_density=0.0,
                uridine_fraction=0.10,
                start_unpairing_prob=1.0,
                mfe=-0.60 * len(candidate.sequence),
            )
        )
        assert 0.0 < archive.hypervolume() <= PRIMARY.max_hypervolume()
        assert 0.0 < archive.normalised_hypervolume() <= 1.0

    def test_better_candidate_gives_larger_hypervolume(self, candidate):
        poor, good = ParetoArchive(PRIMARY), ParetoArchive(PRIMARY)
        poor.update(_scored(candidate, cai=0.4, cpg_density=9.0, uridine_fraction=0.40))
        good.update(_scored(candidate, cai=0.95, cpg_density=0.5, uridine_fraction=0.15))
        assert good.hypervolume() > poor.hypervolume()

    def test_length_invariance_across_targets(self):
        """A short and a long construct of equal per-nt quality score the same."""
        short = seed_candidate("MVSKGEELFTG", strategy="cai_max")
        long_ = seed_candidate(PROTEIN * 3, strategy="cai_max")
        vs = _scored(short, mfe=-0.35 * len(short.sequence)).scores
        vl = _scored(long_, mfe=-0.35 * len(long_.sequence)).scores
        a = PRIMARY.vector(vs, len(short.sequence))
        b = PRIMARY.vector(vl, len(long_.sequence))
        assert a[0] == pytest.approx(b[0]), "MFE density must be length-invariant"


# ── Budget accounting ─────────────────────────────────────────────────────────


class TestEvaluationBudget:
    """
    The research question is posed "under the same tool-call budget", but
    nothing counted evaluations and the loop re-scored unchanged candidates —
    45% of all evaluations in the original loop were duplicates.
    """

    def test_budget_charges_and_exhausts(self):
        b = EvaluationBudget(max_evaluations=3)
        for _ in range(3):
            b.charge()
        assert b.exhausted
        with pytest.raises(BudgetExhausted):
            b.charge()

    def test_cache_hits_are_not_charged(self):
        b = EvaluationBudget(max_evaluations=10)
        cache = ScoreCache()
        cache.put("seq1", ("scores", "diags"))
        assert cache.get("seq1") is not None
        b.note_cache_hit()
        assert b.spent == 0 and b.served_from_cache == 1

    def test_optimisation_never_exceeds_its_budget(self):
        from mrna_design.baselines import RandomController
        from mrna_design.optimize import OptimizeConfig, optimize

        archive = optimize(
            PROTEIN,
            RandomController(rng_seed=3),
            config=OptimizeConfig(
                max_evaluations=40,
                max_iters=100,
                population_k=5,
                run_structure=False,
                run_safety=False,
                include_lineardesign=False,
                convergence_window=10**6,
            ),
        )
        assert archive.run_budget.spent <= 40

    def test_identical_sequences_are_scored_once(self):
        from mrna_design.baselines import CaiMaxController
        from mrna_design.optimize import OptimizeConfig, optimize

        archive = optimize(
            PROTEIN,
            CaiMaxController(),
            config=OptimizeConfig(
                max_evaluations=200,
                max_iters=20,
                population_k=4,
                run_structure=False,
                run_safety=False,
                include_lineardesign=False,
                convergence_window=10**6,
            ),
        )
        summary = archive.run_cache.summary()
        assert archive.run_budget.spent == summary["distinct_sequences"], (
            "every charged evaluation must correspond to a distinct sequence"
        )


# ── Controller fairness ───────────────────────────────────────────────────────


def test_empty_proposal_policy_is_uniform_across_controllers():
    """
    The rule-based controller had internal fallbacks that guaranteed it always
    emitted an edit, while the LLM controller returned nothing when no
    diagnostic fired. That asymmetry changed how much budget each spent, which
    is precisely the variable the comparison holds fixed.
    """
    from mrna_design.models.edits import EditProposal
    from mrna_design.optimize import OptimizeConfig, _apply_empty_proposal_policy

    cand = seed_candidate(PROTEIN, strategy="cai_max")
    empty = EditProposal(edits=[], controller_type="llm", iteration=1)

    noop = _apply_empty_proposal_policy(
        empty, cand, OptimizeConfig(empty_proposal_policy="noop"), 1
    )
    assert noop.edits == []

    rand = _apply_empty_proposal_policy(
        empty, cand, OptimizeConfig(empty_proposal_policy="random"), 1
    )
    assert len(rand.edits) == 1
    edit = rand.edits[0]
    from mrna_design.validators.codon_table import is_synonymous

    assert is_synonymous(edit.original_codon, edit.new_codon)


# ── The invariant that matters most ───────────────────────────────────────────


@pytest.mark.parametrize("controller_name", ["random", "cai_max", "ga", "nsga2", "rule_based"])
def test_protein_identity_is_preserved_by_every_controller(controller_name):
    """
    The pipeline's central safety claim: no controller can change the encoded
    protein, because only the validated applicator writes nucleotides.
    """
    from mrna_design.baselines import (
        CaiMaxController,
        GeneticAlgorithmController,
        NSGA2Controller,
        RandomController,
    )
    from mrna_design.controller.rule_based import RuleBasedController
    from mrna_design.optimize import OptimizeConfig, optimize
    from mrna_design.validators.codon_table import translate

    controllers = {
        "random": RandomController(rng_seed=7),
        "cai_max": CaiMaxController(),
        "ga": GeneticAlgorithmController(rng_seed=7),
        "nsga2": NSGA2Controller(rng_seed=7),
        "rule_based": RuleBasedController(rng_seed=7),
    }
    archive = optimize(
        PROTEIN,
        controllers[controller_name],
        config=OptimizeConfig(
            max_evaluations=60,
            max_iters=15,
            population_k=4,
            run_structure=False,
            run_safety=False,
            include_lineardesign=False,
            convergence_window=10**6,
        ),
    )
    for cand in archive:
        assert translate(cand.cds).rstrip("*") == PROTEIN, (
            f"{controller_name} produced a candidate encoding a different protein"
        )
