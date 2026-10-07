"""
Lightweight feature-based surrogate model.

Trains a RandomForest and a GradientBoosting regressor on the feature vectors
from mrna_design.surrogate.features and the OpenVaccine targets.

Design rationale
----------------
- No deep learning: fits in seconds, explains itself via feature importances.
- Two targets: `mean_reactivity` and `mean_degradation`.
- Ensemble: a simple average of RF and GB predictions reduces variance.
- Caching: fitted models are pickled to `data/surrogate/` for reuse.
- At inference time, the surrogate replaces compute_all() when ViennaRNA
  is unavailable or when high-throughput screening needs speed.

The SurrogateModel class exposes:
  - fit(records)           — train on OVRecord list
  - predict(candidate)     — returns SurrogatePrediction
  - predict_batch(cands)   — vectorised
  - save / load            — pickle round-trip
  - cv_score               — 5-fold CV R² for both targets (for reporting)
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mrna_design.logging_utils import get_logger
from mrna_design.models.candidate import Candidate
from mrna_design.surrogate.features import DIM, FEATURE_NAMES, extract
from mrna_design.surrogate.openvaccine import OVRecord, records_to_arrays
from mrna_design.surrogate.rnafm import RNAFM_AVAILABLE, RNAFM_DIM
from mrna_design.surrogate.rnafm import embed_batch as rnafm_embed_batch

log = get_logger("surrogate.model")

_DEFAULT_CACHE = Path(__file__).parent.parent.parent / "data" / "surrogate" / "model.pkl"


@dataclass
class SurrogatePrediction:
    """Predicted values from the surrogate for one candidate."""

    mean_reactivity: float  # proxy for structural flexibility (higher = more open)
    mean_degradation: float  # predicted degradation rate (lower is better)
    confidence: float  # rough confidence based on RF tree variance (0–1)


def _make_pipeline(n_estimators: int = 200, max_depth: int = 6) -> dict[str, Pipeline]:
    """Return untrained RF + GB pipelines for both targets."""

    def _rf():
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )

    def _gb():
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        random_state=42,
                    ),
                ),
            ]
        )

    return {
        "rf_react": _rf(),
        "gb_react": _gb(),
        "rf_deg": _rf(),
        "gb_deg": _gb(),
    }


class SurrogateModel:
    """
    Feature-based surrogate predicting mRNA stability & degradation.

    Usage::

        from mrna_design.surrogate.model import SurrogateModel
        from mrna_design.surrogate.openvaccine import load_openvaccine

        records = load_openvaccine()
        model = SurrogateModel()
        cv = model.fit(records)
        print(cv)   # {"reactivity_r2": ..., "degradation_r2": ...}

        pred = model.predict(candidate)
        print(pred.mean_degradation)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        use_rnafm: bool = False,
    ) -> None:
        self._use_rnafm = use_rnafm and RNAFM_AVAILABLE
        if use_rnafm and not RNAFM_AVAILABLE:
            log.event(
                "rnafm_unavailable_fallback",
                msg="RNA-FM not available; falling back to feature-only surrogate.",
            )
        self._input_dim = DIM + (RNAFM_DIM if self._use_rnafm else 0)
        self._pipes = _make_pipeline(n_estimators, max_depth)
        self._fitted = False
        self._feature_importances: dict[str, np.ndarray] = {}
        self._cv_scores: dict[str, float] = {}
        self._target_scale: float = 1.0
        self.n_estimators = n_estimators

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        records: list[OVRecord],
        cv_folds: int = 5,
    ) -> dict[str, float]:
        """
        Fit all four pipelines and compute cross-validated R².

        Parameters
        ----------
        records : list[OVRecord]
        cv_folds : int
            Number of CV folds for evaluation. Set 0 to skip.

        Returns
        -------
        dict with keys 'reactivity_r2' and 'degradation_r2'.
        """
        log.event(
            "surrogate_fit_start",
            n_records=len(records),
            cv_folds=cv_folds,
            use_rnafm=self._use_rnafm,
        )

        X, y_react, y_deg = records_to_arrays(records, extract)
        log.event("features_extracted", X_shape=list(X.shape))

        # Optionally prepend RNA-FM embeddings
        if self._use_rnafm:
            seqs = [r.sequence for r in records]
            embs = rnafm_embed_batch(seqs)
            X = np.concatenate([embs, X], axis=1)  # (N, 640+22)
            log.event("rnafm_embeddings_added", shape=list(X.shape))

        # Fit all four pipelines
        self._pipes["rf_react"].fit(X, y_react)
        self._pipes["gb_react"].fit(X, y_react)
        self._pipes["rf_deg"].fit(X, y_deg)
        self._pipes["gb_deg"].fit(X, y_deg)
        self._fitted = True

        # Absolute reference scale for prediction confidence (see _predict_array).
        # Mean of the two target standard deviations; batch-independent by design.
        self._target_scale = float((np.std(y_react) + np.std(y_deg)) / 2.0) or 1.0

        # Feature importances from RF (not affected by scaling)
        for name, pipe in self._pipes.items():
            rf_step = pipe.named_steps["model"]
            if hasattr(rf_step, "feature_importances_"):
                self._feature_importances[name] = rf_step.feature_importances_

        # Cross-validation
        scores: dict[str, float] = {}
        if cv_folds > 0 and len(records) >= cv_folds * 5:
            for target_name, y in [("reactivity", y_react), ("degradation", y_deg)]:
                pipe_key = f"rf_{target_name[:5]}" if target_name == "reactivity" else "rf_deg"
                cv_r2 = cross_val_score(
                    self._pipes[pipe_key], X, y, cv=cv_folds, scoring="r2", n_jobs=-1
                )
                mean_r2 = float(np.mean(cv_r2))
                scores[f"{target_name}_r2"] = mean_r2
                log.event(
                    "cv_score",
                    target=target_name,
                    mean_r2=round(mean_r2, 4),
                    std_r2=round(float(np.std(cv_r2)), 4),
                )
        self._cv_scores = scores
        log.event("surrogate_fit_done", scores=scores)
        return scores

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, candidate: Candidate) -> SurrogatePrediction:
        """Predict reactivity and degradation for a single Candidate."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        feat = extract(
            candidate.sequence,
            candidate.cds_start,
            candidate.cds_end,
            utr5=candidate.utr5,
        )
        if self._use_rnafm:
            emb = rnafm_embed_batch([candidate.sequence])[0]  # (640,)
            feat = np.concatenate([emb, feat])  # (662,)
        return self._predict_array(feat[np.newaxis, :])[0]

    def predict_batch(self, candidates: Sequence[Candidate]) -> list[SurrogatePrediction]:
        """Vectorised prediction over multiple candidates."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        feat_matrix = np.stack(
            [extract(c.sequence, c.cds_start, c.cds_end, utr5=c.utr5) for c in candidates]
        )
        if self._use_rnafm:
            seqs = [c.sequence for c in candidates]
            embs = rnafm_embed_batch(seqs)  # (N, 640)
            feat_matrix = np.concatenate([embs, feat_matrix], axis=1)  # (N, 662)
        return self._predict_array(feat_matrix)

    def _predict_array(self, X: np.ndarray) -> list[SurrogatePrediction]:
        """Internal: predict from a feature matrix."""
        react_rf = self._pipes["rf_react"].predict(X)
        react_gb = self._pipes["gb_react"].predict(X)
        deg_rf = self._pipes["rf_deg"].predict(X)
        deg_gb = self._pipes["gb_deg"].predict(X)

        # Ensemble: simple average
        react = (react_rf + react_gb) / 2.0
        deg = (deg_rf + deg_gb) / 2.0

        # Confidence from RF tree variance (std of individual trees)
        rf_react_model = self._pipes["rf_react"].named_steps["model"]
        rf_deg_model = self._pipes["rf_deg"].named_steps["model"]
        X_scaled_react = self._pipes["rf_react"].named_steps["scaler"].transform(X)
        X_scaled_deg = self._pipes["rf_deg"].named_steps["scaler"].transform(X)

        tree_preds_react = np.array([t.predict(X_scaled_react) for t in rf_react_model.estimators_])
        tree_preds_deg = np.array([t.predict(X_scaled_deg) for t in rf_deg_model.estimators_])
        std_react = tree_preds_react.std(axis=0)
        std_deg = tree_preds_deg.std(axis=0)
        # Confidence: 1 - normalised combined std (rough heuristic)
        # Confidence from RF tree disagreement.
        #
        # This used to divide by `combined_std.max()`, i.e. it normalised each
        # prediction against the *other rows of the same call*. That made the
        # value meaningless: a single-candidate predict() always produced
        # std/std == 1 -> confidence 0.0, and the same candidate scored
        # differently depending on what it happened to be batched with.
        #
        # We now use an absolute, batch-independent scale: tree-disagreement std
        # is compared against the spread of the training targets, captured at
        # fit() time. confidence = exp(-std / target_scale) falls smoothly from
        # 1 (trees unanimous) toward 0 (trees disagree by a full target sd).
        combined_std = (std_react + std_deg) / 2.0
        scale = float(getattr(self, "_target_scale", 0.0)) or 1.0
        confidence = np.clip(np.exp(-combined_std / scale), 0.0, 1.0)

        return [
            SurrogatePrediction(
                mean_reactivity=float(react[i]),
                mean_degradation=float(deg[i]),
                confidence=float(confidence[i]),
            )
            for i in range(len(X))
        ]

    # ── Feature importance ───────────────────────────────────────────────────

    def feature_importances(self, target: str = "react") -> dict[str, float]:
        """
        Return feature importances from the RF pipeline for the given target.

        Parameters
        ----------
        target : "react" or "deg"
        """
        key = f"rf_{target[:5]}" if target.startswith("react") else "rf_deg"
        imp = self._feature_importances.get(key, np.zeros(self._input_dim))
        # Build feature names: RNA-FM dims first (if used), then hand-crafted
        if self._use_rnafm:
            names = [f"rnafm_{i}" for i in range(RNAFM_DIM)] + list(FEATURE_NAMES)
        else:
            names = list(FEATURE_NAMES)
        return dict(zip(names, imp.tolist(), strict=False))

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Path | str | None = None) -> Path:
        """Pickle the fitted model to disk."""
        path = Path(path) if path else _DEFAULT_CACHE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.event("surrogate_saved", path=str(path))
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> SurrogateModel:
        """Load a pickled SurrogateModel from disk."""
        path = Path(path) if path else _DEFAULT_CACHE
        if not path.exists():
            raise FileNotFoundError(f"No surrogate model at {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)
        log.event("surrogate_loaded", path=str(path))
        return model

    @classmethod
    def load_or_train(
        cls,
        path: Path | str | None = None,
        data_path: Path | str | None = None,
        force_retrain: bool = False,
    ) -> SurrogateModel:
        """
        Load the model from disk if available, otherwise train from scratch.

        Parameters
        ----------
        path : Path | str | None
            Where to load/save the pickled model.
        data_path : Path | str | None
            Path to train.json (used if retraining).
        force_retrain : bool
            If True, always retrain even if a cached model exists.
        """
        path = Path(path) if path else _DEFAULT_CACHE
        if path.exists() and not force_retrain:
            return cls.load(path)

        from mrna_design.surrogate.openvaccine import load_openvaccine

        records = load_openvaccine(path=data_path)
        model = cls()
        model.fit(records)
        model.save(path)
        return model

    def cv_scores(self) -> dict[str, float]:
        """Return the cross-validated R² scores from the last fit() call."""
        return dict(self._cv_scores)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"SurrogateModel({status}, n_estimators={self.n_estimators})"
