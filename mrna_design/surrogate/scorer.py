"""
Surrogate-augmented scorer.

Provides surrogate_score(), which blends the cheap surrogate predictions
into the ObjectiveScores already computed by compute_all().

When to use
-----------
- During early-iteration exploration (many candidates, fast feedback).
- When ViennaRNA is unavailable.
- In ablation runs comparing surrogate vs full metrics.

The surrogate prediction only fills scores that are otherwise None or
overwrites them when `override=True`. The fields it writes:
  - scores.mean_reactivity_surrogate  (new field, always written)
  - scores.mean_degradation_surrogate (new field, always written)

The surrogate never overwrites MFE, ensemble_diversity, or CAI —
those come from the primary metrics pipeline.
"""

from __future__ import annotations

from mrna_design.models.candidate import Candidate
from mrna_design.models.objectives import ObjectiveScores
from mrna_design.surrogate.model import SurrogateModel, SurrogatePrediction
from mrna_design.logging_utils import get_logger

log = get_logger("surrogate.scorer")

_MODEL: SurrogateModel | None = None   # module-level singleton


def get_global_model(
    path=None,
    data_path=None,
    force_retrain: bool = False,
) -> SurrogateModel:
    """
    Return (and lazily load/train) the module-level singleton surrogate.
    Thread-safety: not guaranteed; use in single-threaded optimisation loops.
    """
    global _MODEL
    if _MODEL is None or force_retrain:
        _MODEL = SurrogateModel.load_or_train(
            path=path, data_path=data_path, force_retrain=force_retrain
        )
    return _MODEL


def surrogate_score(
    candidate: Candidate,
    model: SurrogateModel | None = None,
) -> SurrogatePrediction:
    """
    Predict surrogate targets for a single candidate.

    Parameters
    ----------
    candidate : Candidate
    model : SurrogateModel | None
        If None, uses the global singleton (loading or training it if needed).

    Returns
    -------
    SurrogatePrediction
    """
    m = model or get_global_model()
    pred = m.predict(candidate)
    log.debug(
        "surrogate_scored",
        sequence_id=candidate.sequence_id,
        reactivity=round(pred.mean_reactivity, 4),
        degradation=round(pred.mean_degradation, 4),
        confidence=round(pred.confidence, 4),
    )
    return pred
