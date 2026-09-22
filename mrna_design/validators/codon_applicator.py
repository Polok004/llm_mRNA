"""
CodonApplicator — applies an EditProposal to a Candidate, validates each edit,
and returns a new Candidate with accepted edits folded in.

This is the only code that physically modifies sequences. It is the firewall
between controller outputs and actual nucleotide changes.

Validation order per edit (fail-fast)
--------------------------------------
1. Codon index in bounds.
2. Original codon matches what's actually at that position.
3. New codon is synonymous (same amino acid).
4. After substitution: no premature stop.
5. After substitution: GC bounds satisfied.
6. After substitution: no forbidden restriction sites introduced.

Edits are applied sequentially left-to-right. If an edit fails, it is recorded
as rejected in the lineage but does not block subsequent edits.
"""

from __future__ import annotations

import time

from mrna_design.models.candidate import Candidate
from mrna_design.models.edits import CodonEdit, EditProposal, EditRecord, ValidationStatus
from mrna_design.models.objectives import ObjectiveScores
from mrna_design.validators.codon_table import codon_to_aa, is_synonymous
from mrna_design.validators.sequence_validator import (
    check_gc_bounds,
    check_no_premature_stop,
    check_restriction_sites,
    GC_LO,
    GC_HI,
)
from mrna_design.logging_utils import get_logger

log = get_logger("codon_applicator")


def apply_edit_proposal(
    candidate: Candidate,
    proposal: EditProposal,
    gc_lo: float = GC_LO,
    gc_hi: float = GC_HI,
    forbidden_sites: dict[str, str] | None = None,
) -> tuple[Candidate, list[EditRecord]]:
    """
    Apply all edits in `proposal` to `candidate`, validating each.

    Returns
    -------
    (new_candidate, edit_records)
        new_candidate : Candidate with accepted edits applied (scores cleared).
        edit_records  : List of EditRecord for all edits (accepted + rejected).
    """
    # Work on a mutable list of codons
    codons = list(candidate.codons)    # includes stop codon
    records: list[EditRecord] = []
    t0 = time.time()

    for edit in proposal.edits:
        record = _validate_and_apply_edit(
            edit=edit,
            codons=codons,
            candidate=candidate,
            proposal=proposal,
            gc_lo=gc_lo,
            gc_hi=gc_hi,
            forbidden_sites=forbidden_sites,
        )
        records.append(record)
        if record.accepted:
            codons[edit.codon_index] = edit.new_codon

    # Rebuild CDS from (possibly modified) codon list
    new_cds = "".join(codons)
    accepted_count = sum(1 for r in records if r.accepted)
    rejected_count = len(records) - accepted_count

    log.event(
        "edit_proposal_applied",
        sequence_id=candidate.sequence_id,
        iteration=proposal.iteration,
        total=len(records),
        accepted=accepted_count,
        rejected=rejected_count,
        wall_ms=round((time.time() - t0) * 1000, 2),
    )

    new_candidate = candidate.with_new_cds(
        new_cds=new_cds,
        iteration=proposal.iteration,
        lineage_append=records,
    )
    return new_candidate, records


def _validate_and_apply_edit(
    edit: CodonEdit,
    codons: list[str],
    candidate: Candidate,
    proposal: EditProposal,
    gc_lo: float,
    gc_hi: float,
    forbidden_sites: dict[str, str] | None,
) -> EditRecord:
    """Validate a single CodonEdit against the current codon list."""

    def _reject(status: str, reason: str = "") -> EditRecord:
        log.warn(
            "edit_rejected",
            codon_index=edit.codon_index,
            status=status,
            reason=reason,
        )
        return EditRecord(
            edit=edit,
            status=status,
            iteration=proposal.iteration,
            controller_type=proposal.controller_type,
        )

    # 1. Index bounds
    if edit.codon_index >= len(codons):
        return _reject(
            ValidationStatus.REJECTED_INDEX_OOB,
            f"codon_index {edit.codon_index} >= n_codons {len(codons)}",
        )

    # 2. Original codon matches
    actual = codons[edit.codon_index]
    if actual != edit.original_codon:
        return _reject(
            ValidationStatus.REJECTED_WRONG_ORIGINAL,
            f"Expected '{edit.original_codon}' at index {edit.codon_index}, found '{actual}'",
        )

    # 3. Synonymous check
    if not is_synonymous(edit.original_codon, edit.new_codon):
        original_aa = codon_to_aa(edit.original_codon)
        new_aa = codon_to_aa(edit.new_codon)
        return _reject(
            ValidationStatus.REJECTED_NOT_SYNONYMOUS,
            f"'{edit.original_codon}' → '{edit.new_codon}' changes AA: "
            f"{original_aa} → {new_aa}",
        )

    # 4. Premature stop — apply tentatively
    trial_codons = list(codons)
    trial_codons[edit.codon_index] = edit.new_codon
    trial_cds = "".join(trial_codons)
    trial_seq = candidate.utr5 + trial_cds + candidate.utr3

    stop_check = check_no_premature_stop(
        trial_seq, candidate.cds_start, candidate.cds_end
    )
    if not stop_check:
        return _reject(ValidationStatus.REJECTED_PREMATURE_STOP, stop_check.reason)

    # 5. GC bounds
    gc_check = check_gc_bounds(trial_seq, candidate.cds_start, candidate.cds_end, gc_lo, gc_hi)
    if not gc_check:
        return _reject(ValidationStatus.REJECTED_GC_BOUNDS, gc_check.reason)

    # 6. Restriction sites — only check if forbidden_sites provided
    if forbidden_sites:
        hits = check_restriction_sites(
            trial_seq, candidate.cds_start, candidate.cds_end, forbidden_sites
        )
        if hits:
            enzymes = ", ".join(h.enzyme for h in hits)
            return _reject(
                ValidationStatus.REJECTED_RESTRICTION_SITE,
                f"New restriction sites introduced: {enzymes}",
            )

    # ── ACCEPTED ──────────────────────────────────────────────────────────────
    log.event(
        "edit_accepted",
        codon_index=edit.codon_index,
        old_codon=edit.original_codon,
        new_codon=edit.new_codon,
        targeting=edit.targeting_issue,
    )
    return EditRecord(
        edit=edit,
        status=ValidationStatus.ACCEPTED,
        iteration=proposal.iteration,
        controller_type=proposal.controller_type,
    )
