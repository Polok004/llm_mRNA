"""
Objective specification — the contract that makes multi-objective scores comparable.

Why this module exists
----------------------
Hypervolume (HV) is the headline metric of this project: it is what the benchmark
reports, what the statistical tests compare, and what the research question is
phrased in terms of. HV is only meaningful if three things are pinned down:

1. **Orientation.** Every objective must point the same way. We use the
   all-minimise convention: after normalisation, 0.0 is the best achievable
   value and 1.0 is the worst.

2. **Scale.** HV is a *volume*, i.e. a product across dimensions. If one
   objective is measured in hundreds (MFE, ~-300 kcal/mol for a 1 kb mRNA) and
   another in fractions (CAI, in [0, 1]), the volume is determined almost
   entirely by the large one. Measured on this project's own benchmark output,
   raw MFE accounted for 95.3% of the total numeric range across all objectives
   and CAI for 0.08% — so the "multi-objective" HV was, numerically, a
   single-objective MFE score wearing a disguise. Normalising every objective to
   [0, 1] against documented bounds removes that distortion.

3. **Reference point.** HV is measured relative to a reference point. The
   previous implementation derived it from the worst value *observed in that
   run*, which makes HV a function of the run's own history: two runs, two
   reference points, two incomparable numbers. Because every normalised
   objective now lives in [0, 1], we can fix the reference point at
   ``1 + REF_EPS`` in every dimension, once and for all. HV then becomes an
   absolute quantity in [0, (1 + REF_EPS)^m] that is comparable across
   iterations, seeds, controllers, targets and machines.

Length invariance
-----------------
MFE scales with sequence length, so raw MFE cannot be compared between a 717 nt
EGFP construct and a 1995 nt luciferase construct. We use **MFE density**
(kcal/mol per nucleotide), which is standard practice in the RNA design
literature and is length-invariant. Motif and hit counts are likewise converted
to per-kilobase rates.

Bounds are deliberately generous and are *documented constants*, not values
fitted to the data. A value outside its bounds is clamped, and every clamp is
counted in :data:`CLAMP_COUNTS` so that badly-chosen bounds show up in the run
log instead of silently distorting the front.

Choosing an objective set
-------------------------
More objectives is not better. In high-dimensional objective spaces almost every
point is non-dominated (the "curse of dimensionality" in multi-objective
optimisation), so the Pareto front stops discriminating between candidates and
HV becomes numerically degenerate. This project's own dry run shows the failure
mode: with 9 objectives, three of them (``mirna_seed_hits``, ``uorf_count``,
``long_dsrna_count``) were identically zero across every candidate, contributing
no information while still multiplying the dimensionality.

We therefore expose named objective sets. :data:`PRIMARY` is the default and is
the set the headline claim should be made over; :data:`EXTENDED` reproduces the
original nine for ablation. The chosen set is recorded in each run's manifest so
that any reported HV can be traced to the space it was measured in.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover
    from mrna_design.models.objectives import ObjectiveScores

# Reference point offset. Normalised objectives live in [0, 1]; placing the
# reference just outside that box guarantees every feasible point contributes
# positive volume, including a point that is worst-in-class on some axis.
REF_EPS: float = 0.05

# Diagnostic counter: how often each objective had to be clamped into range.
# Surfaced in the run manifest so out-of-range bounds are visible, not silent.
CLAMP_COUNTS: Counter = Counter()

Direction = Literal["min", "max"]
Transform = Literal["none", "per_nt", "per_kb"]


@dataclass(frozen=True)
class Objective:
    """
    One axis of the objective space.

    Attributes
    ----------
    attr
        Name of the field on :class:`~mrna_design.models.objectives.ObjectiveScores`.
    label
        Human-readable name used in plots and tables.
    direction
        ``"min"`` if lower raw values are better, ``"max"`` if higher are better.
    best, worst
        Documented bounds *in raw units, after `transform` is applied*. ``best``
        maps to normalised 0.0 and ``worst`` maps to normalised 1.0. Note that
        for a ``"min"`` objective ``best < worst``, and for a ``"max"``
        objective ``best > worst``.
    transform
        ``"per_nt"`` divides the raw value by sequence length (used for MFE),
        ``"per_kb"`` converts a count to a rate per 1000 nt, ``"none"`` leaves
        the value alone.
    missing
        Normalised value substituted when the metric was never computed. We use
        1.0 (worst case) rather than 0.0 so that skipping an expensive metric can
        never *flatter* a candidate — an un-run check is pessimistic, not free.
    rationale
        Short justification for the bounds, quoted in the report.
    """

    attr: str
    label: str
    direction: Direction
    best: float
    worst: float
    transform: Transform = "none"
    missing: float = 1.0
    rationale: str = ""

    def raw_to_transformed(self, raw: float, seq_length: int) -> float:
        """Apply the length transform to a raw metric value."""
        if self.transform == "per_nt":
            return raw / max(seq_length, 1)
        if self.transform == "per_kb":
            return raw * 1000.0 / max(seq_length, 1)
        return raw

    def normalise(self, raw: float | None, seq_length: int) -> float:
        """
        Map a raw metric value onto [0, 1] where 0 is best and 1 is worst.

        Returns :attr:`missing` when ``raw`` is None.
        """
        if raw is None:
            return self.missing
        value = self.raw_to_transformed(float(raw), seq_length)
        span = self.worst - self.best
        if span == 0:
            return 0.0
        unit = (value - self.best) / span
        if unit < 0.0 or unit > 1.0:
            CLAMP_COUNTS[self.attr] += 1
        return min(1.0, max(0.0, unit))


# ── The objective catalogue ───────────────────────────────────────────────────
#
# Bounds below are documented design constants chosen to bracket the plausible
# range for human-codon mRNA therapeutics. They are intentionally wider than
# observed data so that normalisation does not depend on the benchmark itself.

MFE_DENSITY = Objective(
    attr="mfe",
    label="MFE density (kcal/mol/nt)",
    direction="min",
    best=-0.60,
    worst=-0.10,
    transform="per_nt",
    rationale=(
        "Raw MFE scales with length and cannot be compared across targets. "
        "Folded mRNA typically sits near -0.3 to -0.45 kcal/mol/nt; the bounds "
        "bracket that with headroom on both sides."
    ),
)

CAI = Objective(
    attr="cai",
    label="CAI",
    direction="max",
    best=1.0,
    worst=0.0,
    rationale="CAI is defined on [0, 1] by construction (Sharp & Li, 1987).",
)

CPG_DENSITY = Objective(
    attr="cpg_density",
    label="CpG per 100 nt",
    direction="min",
    best=0.0,
    worst=12.0,
    rationale=(
        "CpG-depleted therapeutic designs approach 0; unoptimised human CDS "
        "rarely exceeds ~12 CpG per 100 nt."
    ),
)

URIDINE_FRACTION = Objective(
    attr="uridine_fraction",
    label="Uridine fraction",
    direction="min",
    best=0.10,
    worst=0.45,
    rationale=(
        "Uridine content drives TLR7/8 recognition. Synonymous recoding can "
        "move U fraction roughly within 0.10-0.45 for human proteins."
    ),
)

START_UNPAIRING = Objective(
    attr="start_unpairing_prob",
    label="AUG unpairing probability",
    direction="max",
    best=1.0,
    worst=0.0,
    rationale="A probability, already on [0, 1]; higher means a more accessible start codon.",
)

UPA_DENSITY = Objective(
    attr="upa_density",
    label="UpA per 100 nt",
    direction="min",
    best=0.0,
    worst=12.0,
    rationale="UpA is an RNase L substrate motif; same scale convention as CpG.",
)

GU_MOTIFS = Objective(
    attr="gu_motif_count",
    label="GU TLR motifs per kb",
    direction="min",
    best=0.0,
    worst=20.0,
    transform="per_kb",
    rationale="Count normalised to a rate so long and short constructs are comparable.",
)

MIRNA_HITS = Objective(
    attr="mirna_seed_hits",
    label="miRNA seed hits per kb",
    direction="min",
    best=0.0,
    worst=30.0,
    transform="per_kb",
    rationale=(
        "Seed matches to the human miRNA repertoire, as a rate. Requires the "
        "miRBase database; absent data normalises to the worst case, not zero."
    ),
)

UORF_COUNT = Objective(
    attr="uorf_count",
    label="uORFs in 5'UTR",
    direction="min",
    best=0.0,
    worst=5.0,
    rationale="Upstream ORFs sequester scanning ribosomes; more than a handful is pathological.",
)

LONG_DSRNA = Objective(
    attr="long_dsrna_count",
    label="Long dsRNA stems",
    direction="min",
    best=0.0,
    worst=10.0,
    rationale="Stems >=40 bp are RIG-I/MDA5/PKR triggers (Hornung et al., 2006).",
)


@dataclass(frozen=True)
class ObjectiveSet:
    """An ordered, named collection of objectives defining one objective space."""

    name: str
    objectives: tuple[Objective, ...]
    description: str = ""

    def __len__(self) -> int:
        return len(self.objectives)

    @property
    def labels(self) -> list[str]:
        return [o.label for o in self.objectives]

    @property
    def attrs(self) -> list[str]:
        return [o.attr for o in self.objectives]

    def vector(self, scores: ObjectiveScores, seq_length: int) -> list[float]:
        """
        Normalised, fixed-length, all-minimise objective vector.

        The length depends only on the objective set, never on which metrics
        happen to have been computed — the previous implementation appended a
        tenth element only when a surrogate score was present, producing ragged
        vectors that ``zip``-based dominance checks silently truncated.
        """
        return [
            obj.normalise(getattr(scores, obj.attr, None), seq_length) for obj in self.objectives
        ]

    def reference_point(self) -> list[float]:
        """Fixed HV reference point: worst-case corner, nudged outward by REF_EPS."""
        return [1.0 + REF_EPS] * len(self.objectives)

    def max_hypervolume(self) -> float:
        """The HV of a single ideal point at the origin — the theoretical ceiling."""
        return (1.0 + REF_EPS) ** len(self.objectives)

    def describe(self) -> list[dict]:
        """Serialisable description, written into the run manifest."""
        return [
            {
                "attr": o.attr,
                "label": o.label,
                "direction": o.direction,
                "best": o.best,
                "worst": o.worst,
                "transform": o.transform,
                "rationale": o.rationale,
            }
            for o in self.objectives
        ]


PRIMARY = ObjectiveSet(
    name="primary",
    objectives=(MFE_DENSITY, CAI, CPG_DENSITY, URIDINE_FRACTION, START_UNPAIRING),
    description=(
        "Five objectives spanning the three design pressures the pipeline actually "
        "models: stability (MFE density), translation (CAI, start-codon accessibility) "
        "and innate immunogenicity (CpG, uridine). Kept deliberately small — Pareto "
        "dominance loses discriminating power as dimensionality grows."
    ),
)

SAFETY_EXTENDED = ObjectiveSet(
    name="safety_extended",
    objectives=PRIMARY.objectives + (MIRNA_HITS, UPA_DENSITY),
    description=(
        "PRIMARY plus the two off-target/immune axes that require external "
        "databases (miRBase). Use when `make db` has been run."
    ),
)

EXTENDED = ObjectiveSet(
    name="extended",
    objectives=(
        MFE_DENSITY,
        CAI,
        CPG_DENSITY,
        UPA_DENSITY,
        GU_MOTIFS,
        MIRNA_HITS,
        UORF_COUNT,
        START_UNPAIRING,
        LONG_DSRNA,
    ),
    description=(
        "The original nine-objective space, retained for ablation. Expect a large, "
        "weakly-discriminating front: in the reference dry run three of these axes "
        "were identically zero across all candidates."
    ),
)

OBJECTIVE_SETS: dict[str, ObjectiveSet] = {s.name: s for s in (PRIMARY, SAFETY_EXTENDED, EXTENDED)}

DEFAULT_OBJECTIVE_SET = PRIMARY


def get_objective_set(name: str | ObjectiveSet) -> ObjectiveSet:
    """Resolve an objective set by name, or pass one through unchanged."""
    if isinstance(name, ObjectiveSet):
        return name
    try:
        return OBJECTIVE_SETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown objective set {name!r}. Available: {sorted(OBJECTIVE_SETS)}"
        ) from None


def degenerate_objectives(vectors: list[list[float]], tol: float = 1e-12) -> list[int]:
    """
    Return indices of objectives that take a single value across ``vectors``.

    A constant objective adds a dimension without adding information: it cannot
    break a dominance tie, but it does make the front larger and the HV smaller.
    The benchmark reports these so a degenerate experimental design is visible in
    the output rather than buried in it.
    """
    if not vectors:
        return []
    n_obj = len(vectors[0])
    flat: list[int] = []
    for j in range(n_obj):
        column = [v[j] for v in vectors]
        if max(column) - min(column) <= tol:
            flat.append(j)
    return flat
