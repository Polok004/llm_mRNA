"""
Tests for the RNA-FM surrogate feature extractor.

All tests that require the `transformers` library are guarded with
`pytest.importorskip("transformers")` so the suite stays green in CI
without a GPU or HuggingFace access.

Tests that don't require transformers (the availability flag, fallback
behaviour) always run.
"""
from __future__ import annotations

import numpy as np
import pytest

from mrna_design.surrogate.rnafm import RNAFM_AVAILABLE, RNAFM_DIM


# ── Always-running tests ───────────────────────────────────────────────────────

def test_availability_flag_is_bool():
    """RNAFM_AVAILABLE must be a plain bool regardless of environment."""
    assert isinstance(RNAFM_AVAILABLE, bool)


def test_dim_constant():
    """RNA-FM hidden dimension is 640."""
    assert RNAFM_DIM == 640


def test_model_fallback_when_unavailable():
    """SurrogateModel(use_rnafm=True) silently falls back when RNA-FM is absent."""
    from mrna_design.surrogate.model import SurrogateModel
    m = SurrogateModel(use_rnafm=True)
    # If RNA-FM is not installed, _use_rnafm should be False
    if not RNAFM_AVAILABLE:
        assert m._use_rnafm is False
        assert m._input_dim == 22  # feature-only mode
    else:
        assert m._use_rnafm is True
        assert m._input_dim == 22 + 640


# ── RNA-FM-only tests (skipped without transformers) ──────────────────────────

@pytest.fixture(scope="module")
def rnafm():
    """Import the rnafm module, skipping if not available."""
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from mrna_design.surrogate import rnafm as _rnafm
    return _rnafm


@pytest.mark.skipif(not RNAFM_AVAILABLE, reason="transformers not installed")
def test_embed_sequence_shape(rnafm):
    """embed_sequence returns a 640-dim float32 vector."""
    emb = rnafm.embed_sequence("AUGCGAUAG")
    assert emb.shape == (640,)
    assert emb.dtype == np.float32


@pytest.mark.skipif(not RNAFM_AVAILABLE, reason="transformers not installed")
def test_embed_sequence_finite(rnafm):
    """Embedding must be fully finite (no NaN/Inf)."""
    emb = rnafm.embed_sequence("AUGCGAUAG")
    assert np.all(np.isfinite(emb))


@pytest.mark.skipif(not RNAFM_AVAILABLE, reason="transformers not installed")
def test_embed_sequence_deterministic(rnafm):
    """Same sequence must yield identical embeddings on repeated calls."""
    seq = "AUGCGAUAG"
    emb1 = rnafm.embed_sequence(seq)
    emb2 = rnafm.embed_sequence(seq)
    np.testing.assert_array_equal(emb1, emb2)


@pytest.mark.skipif(not RNAFM_AVAILABLE, reason="transformers not installed")
def test_embed_batch_shape(rnafm):
    """embed_batch returns correct shape for a list of sequences."""
    seqs = ["AUGCGAUAG", "UUGGCCAAU", "GCGCGCGCG"]
    embs = rnafm.embed_batch(seqs)
    assert embs.shape == (3, 640)
    assert embs.dtype == np.float32


@pytest.mark.skipif(not RNAFM_AVAILABLE, reason="transformers not installed")
def test_embed_batch_single_matches_single(rnafm):
    """Batch embedding of a single seq must equal standalone embedding."""
    seq = "AUGCGAUAG"
    emb_single = rnafm.embed_sequence(seq)
    emb_batch = rnafm.embed_batch([seq])[0]
    np.testing.assert_allclose(emb_single, emb_batch, rtol=1e-5)


@pytest.mark.skipif(not RNAFM_AVAILABLE, reason="transformers not installed")
def test_surrogate_fit_with_rnafm():
    """SurrogateModel with use_rnafm=True trains on synthetic data without error."""
    from mrna_design.surrogate.model import SurrogateModel
    from mrna_design.surrogate.openvaccine import load_openvaccine

    records = load_openvaccine(max_records=20)
    model = SurrogateModel(n_estimators=10, use_rnafm=True)
    scores = model.fit(records, cv_folds=0)
    assert model._use_rnafm is True
    assert model._input_dim == 22 + 640
    assert isinstance(scores, dict)
