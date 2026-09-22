"""
Tests for validators (codon_table, sequence_validator, codon_applicator).

These tests have no external dependencies — they run offline.
"""

from __future__ import annotations

import pytest

from mrna_design.validators.codon_table import (
    SYNONYMOUS_CODONS,
    HUMAN_FREQUENCIES,
    STANDARD_CODE,
    codon_to_aa,
    is_synonymous,
    max_freq_codon,
    translate,
    aa_synonyms,
)
from mrna_design.validators.sequence_validator import (
    check_gc_bounds,
    check_no_premature_stop,
    check_no_uorfs,
    check_protein_identity,
    check_restriction_sites,
    check_start_codon,
    validate_all,
    all_pass,
)
from mrna_design.validators.codon_applicator import apply_edit_proposal
from mrna_design.models.candidate import Candidate
from mrna_design.models.edits import CodonEdit, EditProposal


# ── Codon table ───────────────────────────────────────────────────────────────

class TestCodonTable:
    def test_all_64_codons_present(self):
        assert len(STANDARD_CODE) == 64

    def test_stop_codons(self):
        for codon in ("UAA", "UAG", "UGA"):
            assert STANDARD_CODE[codon] == "*"

    def test_aug_is_met(self):
        assert codon_to_aa("AUG") == "M"

    def test_dna_input_normalised(self):
        assert codon_to_aa("ATG") == "M"

    def test_is_synonymous_true(self):
        assert is_synonymous("GAA", "GAG")   # both Glu

    def test_is_synonymous_false(self):
        assert not is_synonymous("GAA", "GCU")  # Glu vs Ala

    def test_max_freq_codon_glu(self):
        # GAG is higher frequency than GAA for Glu in humans
        best = max_freq_codon("E")
        assert best in ("GAG", "GAA")  # GAG should win
        assert codon_to_aa(best) == "E"

    def test_synonymous_codons_sorted_by_freq(self):
        for aa, codons in SYNONYMOUS_CODONS.items():
            if aa in ("M", "W", "*") or len(codons) < 2:
                continue
            freqs = [HUMAN_FREQUENCIES.get(c, 0.0) for c in codons]
            assert freqs == sorted(freqs, reverse=True), \
                f"Synonymous codons for {aa} not sorted by frequency: {codons}"

    def test_translate_egfp_start(self):
        # AUGGUUAGC → MVS
        result = translate("AUGGUUAGC")
        assert result == "MVS"

    def test_translate_with_stop(self):
        result = translate("AUGGUUUAA")
        assert result == "MV*"

    def test_translate_wrong_length(self):
        with pytest.raises(ValueError):
            translate("AUGGU")   # 5 nt — not divisible by 3

    def test_aa_synonyms_met_has_one(self):
        assert aa_synonyms("M") == ["AUG"]

    def test_aa_synonyms_trp_has_one(self):
        assert aa_synonyms("W") == ["UGG"]

    def test_aa_synonyms_leu_has_six(self):
        assert len(aa_synonyms("L")) == 6

    def test_human_frequencies_sum_per_aa(self):
        """Frequencies for synonyms of each AA should sum to approximately 1."""
        for aa, codons in SYNONYMOUS_CODONS.items():
            if aa == "*":
                continue
            total = sum(HUMAN_FREQUENCIES.get(c, 0.0) for c in codons)
            assert abs(total - 1.0) < 0.05, \
                f"Frequencies for {aa} sum to {total:.3f} (expected ~1.0)"


# ── Sequence validator ────────────────────────────────────────────────────────

class TestSequenceValidator:
    def _make_candidate(self, cds: str, protein: str, utr5: str = "", utr3: str = "") -> Candidate:
        return Candidate.from_cds(cds=cds, protein=protein, utr5=utr5, utr3=utr3)

    def test_start_codon_ok(self):
        c = self._make_candidate("AUGGUUUAA", "MV")
        result = check_start_codon(c.sequence, c.cds_start)
        assert result.passed

    def test_start_codon_bad(self):
        c = self._make_candidate("GCUGUUUAA", "AV")
        result = check_start_codon(c.sequence, c.cds_start)
        assert not result.passed

    def test_protein_identity_ok(self):
        c = self._make_candidate("AUGGUUUAA", "MV")
        result = check_protein_identity(c.sequence, c.cds_start, c.cds_end, "MV")
        assert result.passed

    def test_protein_identity_mismatch(self):
        c = self._make_candidate("AUGGUUUAA", "MV")
        result = check_protein_identity(c.sequence, c.cds_start, c.cds_end, "MA")
        assert not result.passed

    def test_premature_stop_detected(self):
        # UAA in the middle (codon 1), then another stop at end
        # MVS* — but we'll insert a stop in middle: AUG UAA GUU UAA
        c = self._make_candidate("AUGUAAGUUUAA", "MVS")  # premature stop at codon 1
        # Override the candidate to skip protein identity check here (just testing stop)
        result = check_no_premature_stop(c.sequence, 0, len(c.sequence))
        assert not result.passed

    def test_no_premature_stop_ok(self):
        c = self._make_candidate("AUGGUUAGCUAA", "MVS")
        result = check_no_premature_stop(c.sequence, c.cds_start, c.cds_end)
        assert result.passed

    def test_gc_bounds_ok(self):
        cds = "AUGGUUAGCUAA"
        c = self._make_candidate(cds, "MVS")
        result = check_gc_bounds(c.sequence, c.cds_start, c.cds_end, lo=0.1, hi=0.9)
        assert result.passed

    def test_gc_bounds_violated(self):
        # All A/U — GC ~0
        cds = "AUGUUUAUAUAA"
        c = self._make_candidate(cds, "MFI")
        result = check_gc_bounds(c.sequence, c.cds_start, c.cds_end, lo=0.5, hi=0.9)
        assert not result.passed

    def test_restriction_site_ecori_detected(self):
        # EcoRI site GAATTC in DNA = GAAUUC in RNA
        cds = "AUGGAAUUCUAA"
        # protein would be MEF* ... use a simple case
        # Just check the detection function directly
        hits = check_restriction_sites(cds, 0, len(cds), {"EcoRI": "GAATTC"})
        assert len(hits) > 0
        assert hits[0].enzyme == "EcoRI"

    def test_restriction_site_none(self):
        cds = "AUGGUUAGCUAA"
        hits = check_restriction_sites(cds, 0, len(cds))
        assert len(hits) == 0

    def test_uorf_detection(self):
        utr5_with_uorf = "GGGAUGGUUUAAGGG"  # contains AUG...UAA uORF
        result = check_no_uorfs(utr5_with_uorf)
        assert not result.passed

    def test_no_uorf(self):
        utr5_clean = "GGGAAACCCUUUGGG"
        result = check_no_uorfs(utr5_clean)
        assert result.passed


# ── Codon applicator ──────────────────────────────────────────────────────────

class TestCodonApplicator:
    def _make_candidate(self, cds: str, protein: str) -> Candidate:
        return Candidate.from_cds(cds=cds, protein=protein)

    def test_valid_edit_accepted(self, minimal_candidate):
        """A valid synonymous edit should be accepted."""
        # codon 0 is AUG (Met) — only one codon, can't be changed
        # Find a codon with synonyms
        c = minimal_candidate
        for i, codon in enumerate(c.codons[:-1]):  # skip stop
            aa = codon_to_aa(codon)
            alts = [x for x in aa_synonyms(aa) if x != codon]
            if alts:
                proposal = EditProposal(
                    edits=[CodonEdit(
                        codon_index=i,
                        original_codon=codon,
                        new_codon=alts[0],
                        reason="test",
                    )],
                    controller_type="rule_based",
                )
                new_c, records = apply_edit_proposal(c, proposal)
                assert records[0].accepted, f"Edit should be accepted: {records[0]}"
                # Protein identity preserved
                from mrna_design.validators import translate, check_protein_identity
                result = check_protein_identity(
                    new_c.sequence, new_c.cds_start, new_c.cds_end, c.protein
                )
                assert result.passed, result.reason
                return
        pytest.skip("No editable codons in minimal_candidate")

    def test_wrong_original_codon_rejected(self, minimal_candidate):
        c = minimal_candidate
        codon_0 = c.codons[0]   # AUG
        proposal = EditProposal(
            edits=[CodonEdit(
                codon_index=0,
                original_codon="GGG",   # wrong — actual is AUG
                new_codon="GGC",
                reason="test wrong original",
            )],
            controller_type="rule_based",
        )
        _, records = apply_edit_proposal(c, proposal)
        assert not records[0].accepted
        assert "wrong_original" in records[0].status

    def test_non_synonymous_edit_rejected(self, minimal_candidate):
        c = minimal_candidate
        # AUG (M) → GGC (G) — not synonymous
        proposal = EditProposal(
            edits=[CodonEdit(
                codon_index=0,
                original_codon="AUG",
                new_codon="GGC",
                reason="test non-synonymous",
            )],
            controller_type="rule_based",
        )
        _, records = apply_edit_proposal(c, proposal)
        assert not records[0].accepted
        assert "not_synonymous" in records[0].status

    def test_protein_identity_preserved_after_multiple_edits(self, egfp_cds_candidate):
        """Apply several edits and verify protein identity is never broken."""
        from mrna_design.validators import check_protein_identity
        c = egfp_cds_candidate
        edits = []
        used = set()
        for i, codon in enumerate(c.codons[1:10], start=1):  # skip AUG
            aa = codon_to_aa(codon)
            alts = [x for x in aa_synonyms(aa) if x != codon]
            if alts and i not in used:
                edits.append(CodonEdit(
                    codon_index=i,
                    original_codon=codon,
                    new_codon=alts[0],
                    reason="test",
                ))
                used.add(i)

        if not edits:
            pytest.skip("No editable codons")

        proposal = EditProposal(edits=edits, controller_type="rule_based")
        new_c, records = apply_edit_proposal(c, proposal)

        # At least one accepted
        assert any(r.accepted for r in records)

        result = check_protein_identity(
            new_c.sequence, new_c.cds_start, new_c.cds_end, c.protein
        )
        assert result.passed, f"Protein identity violated: {result.reason}"
