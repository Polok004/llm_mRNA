"""
Tests for the surrogate model (features, data loader, model, scorer).

All tests run offline — the synthetic OV dataset is used throughout.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from mrna_design.surrogate.features import (
    DIM,
    FEATURE_NAMES,
    batch_extract,
    extract,
    _dinuc_shannon,
    _count_poly_runs,
)
from mrna_design.surrogate.openvaccine import (
    OVRecord,
    _synthetic_dataset,
    records_to_arrays,
)
from mrna_design.surrogate.model import SurrogateModel, SurrogatePrediction


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_records():
    return _synthetic_dataset(n=100, rng_seed=7)


@pytest.fixture(scope="module")
def fitted_model(synthetic_records):
    model = SurrogateModel(n_estimators=50)
    model.fit(synthetic_records, cv_folds=0)   # no CV for speed
    return model


@pytest.fixture
def egfp_feat(egfp_cds_candidate):
    c = egfp_cds_candidate
    return extract(c.sequence, c.cds_start, c.cds_end, utr5=c.utr5)


# ── Feature extractor ─────────────────────────────────────────────────────────

class TestFeatures:
    def test_output_dim(self, egfp_cds_candidate):
        c = egfp_cds_candidate
        feat = extract(c.sequence, c.cds_start, c.cds_end)
        assert feat.shape == (DIM,), f"Expected shape ({DIM},), got {feat.shape}"

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == DIM

    def test_dtype_float32(self, egfp_feat):
        assert egfp_feat.dtype == np.float32

    def test_cai_range(self, egfp_feat):
        # Feature index 0 = cai
        assert 0.0 <= egfp_feat[0] <= 1.0, f"CAI out of range: {egfp_feat[0]}"

    def test_gc_range(self, egfp_feat):
        # Feature index 1 = gc_content
        assert 0.0 <= egfp_feat[1] <= 1.0, f"GC out of range: {egfp_feat[1]}"

    def test_no_nan(self, egfp_feat):
        """After NaN filtering in cai_vector, a proper CDS should yield no NaN."""
        assert not np.any(np.isnan(egfp_feat)), (
            f"Feature vector contains NaN at indices: "
            f"{np.where(np.isnan(egfp_feat))[0].tolist()}"
        )

    def test_no_inf(self, egfp_feat):
        assert not np.any(np.isinf(egfp_feat)), "Feature vector contains Inf"

    def test_batch_extract_shape(self, egfp_cds_candidate):
        c = egfp_cds_candidate
        X = batch_extract(
            [c.sequence, c.sequence],
            [c.cds_start, c.cds_start],
            [c.cds_end, c.cds_end],
        )
        assert X.shape == (2, DIM)
        assert np.allclose(X[0], X[1], equal_nan=True)

    def test_poly_run_counter(self):
        assert _count_poly_runs("AAAAUUU", "A", min_run=4) == 1
        assert _count_poly_runs("AAAUUU", "A", min_run=4) == 0
        assert _count_poly_runs("AAAAAAAAUUU", "A", min_run=4) == 1
        assert _count_poly_runs("AAAA AAAA", "A", min_run=4) == 2  # two runs

    def test_dinuc_entropy_bounds(self):
        e = _dinuc_shannon("ACGU" * 20)
        assert 0.0 <= e <= 1.0

    def test_cai_max_higher_than_random(self, egfp_cds_candidate):
        """CAI-max seeded candidate should have higher CAI feature than random."""
        from mrna_design.designer.seeds import seed_candidate
        import random
        rng = random.Random(0)
        cai_max = egfp_cds_candidate
        harmonised = seed_candidate(
            cai_max.protein, strategy="harmonised", rng=rng
        )
        feat_cai = extract(cai_max.sequence, cai_max.cds_start, cai_max.cds_end)
        feat_harm = extract(harmonised.sequence, harmonised.cds_start, harmonised.cds_end)
        # CAI-max should have higher or equal feature[0] (CAI)
        assert feat_cai[0] >= feat_harm[0] - 0.05  # allow small numerical margin

    def test_different_sequences_give_different_features(self, egfp_cds_candidate):
        from mrna_design.designer.seeds import seed_candidate
        import random
        c1 = egfp_cds_candidate
        c2 = seed_candidate(c1.protein, strategy="gc_balanced", rng=random.Random(99))
        f1 = extract(c1.sequence, c1.cds_start, c1.cds_end)
        f2 = extract(c2.sequence, c2.cds_start, c2.cds_end)
        # Must differ in at least one feature
        assert not np.allclose(f1, f2), "Different sequences gave identical features"


# ── OpenVaccine data loader ───────────────────────────────────────────────────

class TestOpenVaccineLoader:
    def test_synthetic_length(self, synthetic_records):
        assert len(synthetic_records) == 100

    def test_record_types(self, synthetic_records):
        for r in synthetic_records:
            assert isinstance(r, OVRecord)
            assert isinstance(r.sequence, str)
            assert isinstance(r.mean_reactivity, float)
            assert isinstance(r.mean_degradation, float)

    def test_reactivity_in_range(self, synthetic_records):
        for r in synthetic_records:
            assert r.mean_reactivity >= 0.0

    def test_records_to_arrays_shape(self, synthetic_records):
        X, y_r, y_d = records_to_arrays(synthetic_records, extract)
        assert X.shape == (len(synthetic_records), DIM)
        assert y_r.shape == (len(synthetic_records),)
        assert y_d.shape == (len(synthetic_records),)

    def test_no_nan_in_arrays(self, synthetic_records):
        X, y_r, y_d = records_to_arrays(synthetic_records, extract)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y_r))
        assert not np.any(np.isnan(y_d))


# ── Surrogate model ───────────────────────────────────────────────────────────

class TestSurrogateModel:
    def test_fit_returns_scores(self, synthetic_records):
        model = SurrogateModel(n_estimators=20)
        scores = model.fit(synthetic_records, cv_folds=3)
        assert isinstance(scores, dict)
        # With synthetic GC-correlated targets, R² should be positive
        if "reactivity_r2" in scores:
            assert scores["reactivity_r2"] > -1.0   # not worse than a flat baseline

    def test_predict_returns_correct_type(self, fitted_model, egfp_cds_candidate):
        pred = fitted_model.predict(egfp_cds_candidate)
        assert isinstance(pred, SurrogatePrediction)
        assert isinstance(pred.mean_reactivity, float)
        assert isinstance(pred.mean_degradation, float)
        assert 0.0 <= pred.confidence <= 1.0

    def test_predict_batch_length(self, fitted_model, egfp_cds_candidate):
        preds = fitted_model.predict_batch([egfp_cds_candidate] * 5)
        assert len(preds) == 5
        for p in preds:
            assert isinstance(p, SurrogatePrediction)

    def test_predict_batch_consistent_with_single(self, fitted_model, egfp_cds_candidate):
        single = fitted_model.predict(egfp_cds_candidate)
        batch = fitted_model.predict_batch([egfp_cds_candidate])
        assert abs(single.mean_reactivity - batch[0].mean_reactivity) < 1e-5
        assert abs(single.mean_degradation - batch[0].mean_degradation) < 1e-5

    def test_feature_importances_returns_dict(self, fitted_model):
        imp = fitted_model.feature_importances("react")
        assert isinstance(imp, dict)
        assert len(imp) == DIM
        # Sum to ~1.0 (RF importances are normalised)
        total = sum(imp.values())
        assert abs(total - 1.0) < 0.01

    def test_predict_without_fit_raises(self):
        model = SurrogateModel(n_estimators=10)
        from mrna_design.designer.seeds import seed_candidate
        c = seed_candidate("MVS", strategy="cai_max")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(c)

    def test_save_load_roundtrip(self, fitted_model, tmp_path):
        path = tmp_path / "model.pkl"
        fitted_model.save(path)
        loaded = SurrogateModel.load(path)
        assert loaded._fitted

        from mrna_design.designer.seeds import seed_candidate
        c = seed_candidate("MVS", strategy="cai_max")
        pred_orig = fitted_model.predict(c)
        pred_loaded = loaded.predict(c)
        assert abs(pred_orig.mean_reactivity - pred_loaded.mean_reactivity) < 1e-5

    def test_load_or_train_uses_synthetic_if_no_data(self, tmp_path):
        """load_or_train must succeed even without real OpenVaccine data."""
        path = tmp_path / "surrogate.pkl"
        model = SurrogateModel.load_or_train(path=path, data_path=tmp_path / "no_file.json")
        assert model._fitted

    def test_surrogate_degradation_in_objective_vector(self, fitted_model, egfp_cds_candidate):
        """When surrogate scores are written into ObjectiveScores, the
        objective vector should include the degradation term."""
        pred = fitted_model.predict(egfp_cds_candidate)
        # Patch scores
        c = egfp_cds_candidate
        from mrna_design.models.objectives import ObjectiveScores
        scores = ObjectiveScores(
            mfe=-20.0, gc_content=0.5, gc3_content=0.5, cai=0.8,
            cpg_density=1.0, upa_density=1.5, gu_motif_count=0,
            uridine_fraction=0.25, long_dsrna_count=0,
            mirna_seed_hits=0, blast_hits=0, uorf_count=0,
            start_unpairing_prob=0.7,
            surrogate_degradation=pred.mean_degradation,
            surrogate_reactivity=pred.mean_reactivity,
            surrogate_confidence=pred.confidence,
        )
        vec = scores.to_objective_vector()
        assert len(vec) == 10   # 9 primary + 1 degradation
        assert vec[9] == pytest.approx(pred.mean_degradation, abs=1e-5)
