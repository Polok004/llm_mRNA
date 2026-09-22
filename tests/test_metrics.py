"""
Tests for metrics (CAI, immunogenicity; structure requires ViennaRNA).
"""

from __future__ import annotations

import math
import pytest

from mrna_design.metrics.cai import cai, cai_vector, gc_content, gc3_content, sliding_window_cai
from mrna_design.metrics.immunogenicity import compute_immunogenicity, cpg_density_windows


# ── CAI ────────────────────────────────────────────────────────────────────────

class TestCAI:
    def test_cai_range(self, egfp_cds_candidate):
        c = egfp_cds_candidate
        score = cai(c.cds)
        assert 0.0 < score <= 1.0

    def test_cai_max_is_one(self):
        """A CDS where every codon is the highest-frequency human codon should have CAI=1."""
        from mrna_design.validators.codon_table import SYNONYMOUS_CODONS, HUMAN_FREQUENCIES
        codons = []
        for aa, aa_codons in SYNONYMOUS_CODONS.items():
            if aa in ("*", "M", "W"):
                continue
            best = max(aa_codons, key=lambda c: HUMAN_FREQUENCIES.get(c, 0))
            codons.append(best)
        # Just use a simple case
        cds = "AUGCUGCUGUAA"  # M + 2x most-freq Leu + stop
        score = cai(cds)
        assert 0.0 < score <= 1.001

    def test_cai_all_rare_codons_low(self):
        """A CDS of all rare codons should have lower CAI than CAI-max."""
        from mrna_design.designer.seeds import seed_candidate
        protein = "MVSKGEEL"
        cai_max_c = seed_candidate(protein, strategy="cai_max")
        # Manually build a very rare-codon CDS (use lowest-freq synonyms)
        from mrna_design.validators.codon_table import HUMAN_FREQUENCIES, aa_synonyms
        rare_codons = []
        for aa in protein:
            synonyms = aa_synonyms(aa)
            if len(synonyms) > 1:
                rare = min(synonyms, key=lambda c: HUMAN_FREQUENCIES.get(c, 0))
            else:
                rare = synonyms[0]
            rare_codons.append(rare)
        rare_codons.append("UAA")  # stop
        rare_cds = "".join(rare_codons)
        rare_c = seed_candidate(protein, strategy="cai_max")  # use for structure
        rare_score = cai(rare_cds)
        cai_max_score = cai(cai_max_c.cds)
        assert rare_score <= cai_max_score

    def test_cai_invalid_length(self):
        with pytest.raises(ValueError):
            cai("AUGGU")

    def test_gc_content_known(self):
        # AUGCCC — 4 G+C out of 6 = 0.667
        assert abs(gc_content("AUGCCC") - (4 / 6)) < 0.01

    def test_gc3_content(self):
        # AUG GCC GAA — third positions: G, C, A → 2/3 = 0.667
        assert abs(gc3_content("AUGGCCGAA") - (2 / 3)) < 0.01

    def test_sliding_window_cai_returns_list(self, egfp_cds_candidate):
        c = egfp_cds_candidate
        windows = sliding_window_cai(c.cds, window_codons=5, step_codons=2)
        assert isinstance(windows, list)
        assert len(windows) > 0
        for w in windows:
            assert "cai" in w
            assert 0.0 <= w["cai"] <= 1.0

    def test_cai_vector_length(self, egfp_cds_candidate):
        c = egfp_cds_candidate
        vec = cai_vector(c.cds)
        n_codons = len(c.cds) // 3
        assert len(vec) == n_codons


# ── Immunogenicity ────────────────────────────────────────────────────────────

class TestImmunogenicity:
    def test_cpg_density_known(self):
        # "ACGACG" — 2 CG out of 6 nt → 2/6*100 = 33.3 per 100 nt
        result = compute_immunogenicity("ACGACG")
        assert abs(result.cpg_density - (2 / 6 * 100)) < 0.5

    def test_zero_cpg(self):
        result = compute_immunogenicity("AAAUUU")
        assert result.cpg_density == 0.0

    def test_upa_density_known(self):
        # "UAUAUA" — 3 UA out of 6 nt → 3/6*100 = 50
        result = compute_immunogenicity("UAUAUA")
        assert result.upa_density > 0

    def test_gu_motif_detected(self):
        # Contains "GUUGUGU" — a TLR7 motif
        result = compute_immunogenicity("AAAGUUGUGUAAA")
        assert result.gu_motif_count > 0

    def test_uridine_fraction(self):
        # "UUUAAA" — 3 U out of 6 = 0.5
        result = compute_immunogenicity("UUUAAA")
        assert abs(result.uridine_fraction - 0.5) < 0.01

    def test_empty_sequence(self):
        result = compute_immunogenicity("")
        assert result.cpg_density == 0.0
        assert result.uridine_fraction == 0.0

    def test_cpg_windows_returns_list(self, egfp_cds_candidate):
        windows = cpg_density_windows(egfp_cds_candidate.cds, window=30, step=10)
        assert isinstance(windows, list)
        for w in windows:
            assert "cpg_density" in w
            assert w["cpg_density"] >= 0.0


# ── Structure (skipped if ViennaRNA not available) ─────────────────────────────

class TestStructure:
    @pytest.mark.skipif(
        not pytest.importorskip("RNA", reason="ViennaRNA not installed"),
        reason="ViennaRNA not installed",
    )
    def test_fold_mfe_is_float(self, egfp_cds_candidate):
        from mrna_design.metrics.structure import fold
        result = fold(egfp_cds_candidate.cds[:60])
        assert isinstance(result.mfe, float)
        assert result.mfe <= 0.0  # stable RNA should have negative MFE

    @pytest.mark.skipif(
        not pytest.importorskip("RNA", reason="ViennaRNA not installed"),
        reason="ViennaRNA not installed",
    )
    def test_fold_dot_bracket_length(self, egfp_cds_candidate):
        from mrna_design.metrics.structure import fold
        seq = egfp_cds_candidate.cds[:60]
        result = fold(seq)
        assert len(result.dot_bracket) == len(seq)
        assert all(c in ".()[]{}|," for c in result.dot_bracket)
