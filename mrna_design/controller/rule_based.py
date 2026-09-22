"""
RuleBasedController — deterministic diagnostic-guided edit heuristics.

Each IssueType has a corresponding rule that selects a synonymous codon
substitution in the flagged region. Rules are ranked by severity (HIGH first)
and applied up to `max_edits` per iteration.

Rule catalogue
--------------
stable_hairpin / start_codon_paired
    → Reduce G/C at 5' end of the flagged CDS window.
    → Pick the synonymous codon with lowest GC content.

local_stable_stem
    → Increase sequence entropy in the flagged window.
    → Pick a synonymous codon that differs most from current (Hamming distance).

codon_desert
    → Replace lowest-CAI codon in window with the highest-frequency synonym.

cpg_hotspot
    → Replace codons containing "CG" dinucleotide with non-CG synonyms.

upa_hotspot
    → Replace codons containing "UA" dinucleotide with non-UA synonyms.

gu_tlr_motif
    → Replace U or G in flagged codons with synonyms lacking that pattern.

mirna_seed_match
    → Disrupt the seed match: pick a synonymous codon that breaks the
      complementary run at the flagged position.

low_ensemble_diversity
    → Introduce moderate GC variation by mixing high- and low-GC synonyms.
"""

from __future__ import annotations

import random

from mrna_design.controller.base import BaseController
from mrna_design.models.candidate import Candidate
from mrna_design.models.diagnostics import IssueType, RegionDiagnostic, Severity
from mrna_design.models.edits import CodonEdit, EditProposal
from mrna_design.validators.codon_table import (
    HUMAN_FREQUENCIES,
    aa_synonyms,
    codon_to_aa,
    is_synonymous,
)
from mrna_design.logging_utils import get_logger

log = get_logger("controller.rule_based")


def _codon_gc(codon: str) -> float:
    return (codon.count("G") + codon.count("C")) / 3.0


def _codon_hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _lowest_gc_synonym(codon: str, exclude_self: bool = True) -> str | None:
    aa = codon_to_aa(codon)
    synonyms = [c for c in aa_synonyms(aa) if not (exclude_self and c == codon)]
    if not synonyms:
        return None
    return min(synonyms, key=_codon_gc)


def _highest_cai_synonym(codon: str, exclude_self: bool = True) -> str | None:
    aa = codon_to_aa(codon)
    synonyms = [c for c in aa_synonyms(aa) if not (exclude_self and c == codon)]
    if not synonyms:
        return None
    return max(synonyms, key=lambda c: HUMAN_FREQUENCIES.get(c, 0.0))


def _most_different_synonym(codon: str) -> str | None:
    """Synonym with maximum Hamming distance (increases sequence diversity)."""
    aa = codon_to_aa(codon)
    synonyms = [c for c in aa_synonyms(aa) if c != codon]
    if not synonyms:
        return None
    return max(synonyms, key=lambda c: _codon_hamming(codon, c))


def _no_cg_synonym(codon: str) -> str | None:
    """Synonym that avoids the CG dinucleotide within or spanning the codon."""
    aa = codon_to_aa(codon)
    synonyms = [c for c in aa_synonyms(aa) if c != codon and "CG" not in c]
    if not synonyms:
        return None
    return max(synonyms, key=lambda c: HUMAN_FREQUENCIES.get(c, 0.0))


def _no_ua_synonym(codon: str) -> str | None:
    """Synonym that avoids the UA dinucleotide within the codon."""
    aa = codon_to_aa(codon)
    synonyms = [c for c in aa_synonyms(aa) if c != codon and "UA" not in c]
    if not synonyms:
        return None
    return max(synonyms, key=lambda c: HUMAN_FREQUENCIES.get(c, 0.0))


def _disrupts_complement(original: str, new: str, seed: str) -> bool:
    """
    Return True if replacing `original` with `new` disrupts complementarity
    to `seed` (used for miRNA seed disruption).
    """
    from mrna_design.metrics.immunogenicity import _reverse_complement
    rc_seed = _reverse_complement(seed)
    return (original in rc_seed or rc_seed in original) and \
           (new not in rc_seed and rc_seed not in new)


class RuleBasedController(BaseController):
    """
    Deterministic heuristic controller.

    Reads the diagnostics sorted by severity and generates edit proposals
    that target the flagged regions. This controller is the Week 1–6 default
    and the fallback for the LLM controller.
    """

    name = "rule_based"

    def __init__(self, rng_seed: int = 42) -> None:
        self._rng = random.Random(rng_seed)

    def reset(self) -> None:
        pass  # Rule-based has no memory between candidates

    def propose_edits(
        self,
        candidate: Candidate,
        diagnostics: list[RegionDiagnostic],
        history: list[EditProposal],
        max_edits: int = 5,
        iteration: int = 0,
    ) -> EditProposal:
        """Propose up to max_edits synonymous substitutions guided by diagnostics."""

        # Sort: highest severity first, then highest metric
        sorted_diag = sorted(
            diagnostics,
            key=lambda d: (d.severity.value, d.metric),
            reverse=True,
        )

        edits: list[CodonEdit] = []
        used_indices: set[int] = set()
        targeting: list[str] = []

        for diag in sorted_diag:
            if len(edits) >= max_edits:
                break
            new_edits = self._rule_for(diag, candidate, used_indices)
            for edit in new_edits:
                if len(edits) >= max_edits:
                    break
                used_indices.add(edit.codon_index)
                edits.append(edit)
                targeting.append(diag.issue.value)

        # Fallback: if no diagnostics or no rules fired, improve CAI greedily
        if not edits:
            edits = self._fallback_cai_improvement(candidate, max_edits, used_indices)
            targeting = ["cai_improvement"]

        if not edits:
            # Nothing to do: return a no-op proposal with a single no-change note
            # (this should not happen in practice but prevents empty EditProposal)
            edits = self._random_synonymous_edit(candidate, 1)
            targeting = ["random_fallback"]

        proposal = EditProposal(
            edits=edits,
            expected_improvement="; ".join(targeting),
            controller_type="rule_based",
            targeting_diagnostics=list(set(targeting)),
            iteration=iteration,
        )
        log.event(
            "rule_based_proposal",
            iteration=iteration,
            n_edits=len(edits),
            targeting=targeting,
        )
        return proposal

    # ── Rule dispatch ──────────────────────────────────────────────────────────

    def _rule_for(
        self,
        diag: RegionDiagnostic,
        candidate: Candidate,
        used: set[int],
    ) -> list[CodonEdit]:
        """Dispatch to the appropriate rule for a diagnostic."""
        issue = diag.issue

        if issue in (IssueType.STABLE_HAIRPIN, IssueType.START_CODON_PAIRED):
            return self._rule_reduce_gc(diag, candidate, used, n=2)

        elif issue == IssueType.LOCAL_STABLE_STEM:
            return self._rule_increase_entropy(diag, candidate, used, n=2)

        elif issue == IssueType.CODON_DESERT:
            return self._rule_boost_cai(diag, candidate, used, n=3)

        elif issue == IssueType.CPG_HOTSPOT:
            return self._rule_remove_cpg(diag, candidate, used, n=2)

        elif issue == IssueType.UPA_HOTSPOT:
            return self._rule_remove_upa(diag, candidate, used, n=2)

        elif issue in (IssueType.GU_TLR_MOTIF, IssueType.HIGH_URIDINE):
            return self._rule_reduce_u(diag, candidate, used, n=2)

        elif issue == IssueType.MIRNA_SEED_MATCH:
            return self._rule_disrupt_mirna(diag, candidate, used, n=1)

        elif issue == IssueType.LOW_ENSEMBLE_DIVERSITY:
            return self._rule_increase_gc_variation(diag, candidate, used, n=2)

        return []

    # ── Individual rules ───────────────────────────────────────────────────────

    def _codons_in_window(
        self,
        candidate: Candidate,
        window_start: int,
        window_end: int,
        used: set[int],
    ) -> list[tuple[int, str]]:
        """Return (codon_index, codon) pairs that overlap the nt window."""
        results = []
        for i, codon in enumerate(candidate.codons):
            nt_start = candidate.cds_start + i * 3
            nt_end = nt_start + 3
            if nt_end > window_start and nt_start < window_end and i not in used:
                results.append((i, codon))
        return results

    def _make_edit(
        self,
        codon_index: int,
        original: str,
        new_codon: str | None,
        reason: str,
        targeting: str,
    ) -> CodonEdit | None:
        if new_codon is None or new_codon == original:
            return None
        if not is_synonymous(original, new_codon):
            return None
        return CodonEdit(
            codon_index=codon_index,
            original_codon=original,
            new_codon=new_codon,
            reason=reason,
            targeting_issue=targeting,
        )

    def _rule_reduce_gc(
        self, diag: RegionDiagnostic, candidate: Candidate, used: set[int], n: int
    ) -> list[CodonEdit]:
        codons = self._codons_in_window(
            candidate, diag.window_start, diag.window_end, used
        )
        # Sort by GC content of current codon (high GC first)
        codons.sort(key=lambda x: _codon_gc(x[1]), reverse=True)
        edits = []
        for idx, codon in codons[:n]:
            new = _lowest_gc_synonym(codon)
            e = self._make_edit(idx, codon, new, "Reduce GC to destabilise hairpin", diag.issue.value)
            if e:
                edits.append(e)
        return edits

    def _rule_increase_entropy(
        self, diag: RegionDiagnostic, candidate: Candidate, used: set[int], n: int
    ) -> list[CodonEdit]:
        codons = self._codons_in_window(
            candidate, diag.window_start, diag.window_end, used
        )
        edits = []
        for idx, codon in codons[:n]:
            new = _most_different_synonym(codon)
            e = self._make_edit(idx, codon, new, "Increase sequence entropy to disrupt stem", diag.issue.value)
            if e:
                edits.append(e)
        return edits

    def _rule_boost_cai(
        self, diag: RegionDiagnostic, candidate: Candidate, used: set[int], n: int
    ) -> list[CodonEdit]:
        codons = self._codons_in_window(
            candidate, diag.window_start, diag.window_end, used
        )
        # Sort by current CAI weight (lowest first — most to gain)
        from mrna_design.validators.codon_table import HUMAN_FREQUENCIES
        codons.sort(key=lambda x: HUMAN_FREQUENCIES.get(x[1], 0.0))
        edits = []
        for idx, codon in codons[:n]:
            new = _highest_cai_synonym(codon)
            e = self._make_edit(idx, codon, new, "Boost CAI in low-usage window", diag.issue.value)
            if e:
                edits.append(e)
        return edits

    def _rule_remove_cpg(
        self, diag: RegionDiagnostic, candidate: Candidate, used: set[int], n: int
    ) -> list[CodonEdit]:
        codons = self._codons_in_window(
            candidate, diag.window_start, diag.window_end, used
        )
        edits = []
        for idx, codon in codons:
            if "CG" in codon and idx not in used:
                new = _no_cg_synonym(codon)
                e = self._make_edit(idx, codon, new, "Remove CpG dinucleotide", diag.issue.value)
                if e:
                    edits.append(e)
                if len(edits) >= n:
                    break
        return edits

    def _rule_remove_upa(
        self, diag: RegionDiagnostic, candidate: Candidate, used: set[int], n: int
    ) -> list[CodonEdit]:
        codons = self._codons_in_window(
            candidate, diag.window_start, diag.window_end, used
        )
        edits = []
        for idx, codon in codons:
            if "UA" in codon and idx not in used:
                new = _no_ua_synonym(codon)
                e = self._make_edit(idx, codon, new, "Remove UpA dinucleotide", diag.issue.value)
                if e:
                    edits.append(e)
                if len(edits) >= n:
                    break
        return edits

    def _rule_reduce_u(
        self, diag: RegionDiagnostic, candidate: Candidate, used: set[int], n: int
    ) -> list[CodonEdit]:
        codons = self._codons_in_window(
            candidate, diag.window_start, diag.window_end, used
        )
        # Sort by U count in codon (most U first)
        codons.sort(key=lambda x: x[1].count("U"), reverse=True)
        edits = []
        for idx, codon in codons[:n]:
            aa = codon_to_aa(codon)
            synonyms = [c for c in aa_synonyms(aa) if c != codon]
            if not synonyms:
                continue
            new = min(synonyms, key=lambda c: c.count("U"))
            e = self._make_edit(idx, codon, new, "Reduce U content for TLR/immunogenicity", diag.issue.value)
            if e:
                edits.append(e)
        return edits

    def _rule_disrupt_mirna(
        self, diag: RegionDiagnostic, candidate: Candidate, used: set[int], n: int
    ) -> list[CodonEdit]:
        if diag.codon_start is None:
            # Fall back to window-based
            seed = diag.extra.get("seed_sequence", "")
            if not seed:
                return []
            codons = self._codons_in_window(
                candidate, diag.window_start, diag.window_end, used
            )
        else:
            codons = [
                (i, candidate.codons[i])
                for i in range(diag.codon_start, (diag.codon_end or diag.codon_start) + 1)
                if i not in used and i < len(candidate.codons)
            ]
            seed = diag.extra.get("seed_sequence", "")

        edits = []
        for idx, codon in codons[:n]:
            aa = codon_to_aa(codon)
            synonyms = [c for c in aa_synonyms(aa) if c != codon]
            if not synonyms:
                continue
            # Prefer a codon that disrupts the seed complement
            disrupting = [c for c in synonyms if _disrupts_complement(codon, c, seed)]
            new = disrupting[0] if disrupting else self._rng.choice(synonyms)
            e = self._make_edit(idx, codon, new, f"Disrupt miRNA seed match ({diag.detail})", diag.issue.value)
            if e:
                edits.append(e)
        return edits

    def _rule_increase_gc_variation(
        self, diag: RegionDiagnostic, candidate: Candidate, used: set[int], n: int
    ) -> list[CodonEdit]:
        # Sample n random codons and introduce GC variation
        all_codons = [
            (i, c) for i, c in enumerate(candidate.codons)
            if i not in used and aa_synonyms(codon_to_aa(c))
        ]
        self._rng.shuffle(all_codons)
        edits = []
        for idx, codon in all_codons[:n * 2]:
            if len(edits) >= n:
                break
            aa = codon_to_aa(codon)
            synonyms = [c for c in aa_synonyms(aa) if c != codon]
            if not synonyms:
                continue
            # Pick the one with GC furthest from current
            new = max(synonyms, key=lambda c: abs(_codon_gc(c) - _codon_gc(codon)))
            e = self._make_edit(idx, codon, new, "Increase GC variation to improve ensemble diversity", diag.issue.value)
            if e:
                edits.append(e)
        return edits

    # ── Fallbacks ──────────────────────────────────────────────────────────────

    def _fallback_cai_improvement(
        self,
        candidate: Candidate,
        n: int,
        used: set[int],
    ) -> list[CodonEdit]:
        """When no diagnostics fire, greedily improve CAI on the lowest-weight codons."""
        from mrna_design.validators.codon_table import HUMAN_FREQUENCIES
        indexed = [
            (i, c) for i, c in enumerate(candidate.codons)
            if i not in used
        ]
        indexed.sort(key=lambda x: HUMAN_FREQUENCIES.get(x[1], 0.0))
        edits = []
        for idx, codon in indexed[:n * 2]:
            if len(edits) >= n:
                break
            new = _highest_cai_synonym(codon)
            e = self._make_edit(idx, codon, new, "Greedy CAI improvement (no diagnostics)", "cai_improvement")
            if e:
                edits.append(e)
        return edits

    def _random_synonymous_edit(
        self,
        candidate: Candidate,
        n: int,
    ) -> list[CodonEdit]:
        """Last-resort: make n random synonymous substitutions."""
        eligible = [
            (i, c) for i, c in enumerate(candidate.codons)
            if aa_synonyms(codon_to_aa(c))
        ]
        if not eligible:
            return []
        chosen = self._rng.sample(eligible, min(n, len(eligible)))
        edits = []
        for idx, codon in chosen:
            aa = codon_to_aa(codon)
            synonyms = [c for c in aa_synonyms(aa) if c != codon]
            if not synonyms:
                continue
            new = self._rng.choice(synonyms)
            edits.append(CodonEdit(
                codon_index=idx,
                original_codon=codon,
                new_codon=new,
                reason="Random synonymous substitution (last resort)",
                targeting_issue=None,
            ))
        return edits
