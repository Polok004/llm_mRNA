"""mrna_design/metrics package."""

from mrna_design.metrics.cai import (
    cai,
    cai_vector,
    gc_content,
    gc3_content,
    sliding_window_cai,
)
from mrna_design.metrics.structure import (
    EnsembleResult,
    FoldResult,
    WindowFoldResult,
    ensemble_fold,
    fold,
    fold_windows,
    start_codon_unpairing,
)
from mrna_design.metrics.immunogenicity import (
    ImmunogenicityResult,
    compute_immunogenicity,
    cpg_density_windows,
)
from mrna_design.metrics.safety import (
    BlastHit,
    MirnaSeedHit,
    RnahybridHit,
    SafetyResult,
    compute_safety,
    scan_mirna_seeds,
)
from mrna_design.metrics.aggregator import (
    DEFAULT_THRESHOLDS,
    Thresholds,
    compute_all,
)

__all__ = [
    "cai", "cai_vector", "gc_content", "gc3_content", "sliding_window_cai",
    "FoldResult", "EnsembleResult", "WindowFoldResult",
    "fold", "ensemble_fold", "fold_windows", "start_codon_unpairing",
    "ImmunogenicityResult", "compute_immunogenicity", "cpg_density_windows",
    "MirnaSeedHit", "RnahybridHit", "BlastHit", "SafetyResult",
    "compute_safety", "scan_mirna_seeds",
    "Thresholds", "DEFAULT_THRESHOLDS", "compute_all",
]
