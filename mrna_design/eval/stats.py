"""
Statistical analysis for mRNA design benchmarks.

What was wrong before
---------------------
The original ``compare_hypervolumes`` ran a one-sided Mann-Whitney U test per
controller and reported ``p < 0.05`` as "significant". Three problems:

1. **No multiple-comparison control.** With C controllers across T targets you
   run C x T tests. At 4 controllers and 3 targets that is 12 tests, where the
   chance of at least one spurious "significant" result at alpha = 0.05 is
   about 46%. We apply Holm-Bonferroni, which controls the family-wise error
   rate without assuming independence and is uniformly more powerful than plain
   Bonferroni.

2. **"Effect size" was a relative mean difference**, ``(mean_c - mean_b) / mean_b``.
   That is a percentage change, not an effect size: it is unitless only by
   accident, it explodes when the baseline is near zero, and it says nothing
   about overlap between the distributions. We report **Vargha-Delaney A12**,
   the probability that a random run of the candidate beats a random run of the
   baseline. A12 = 0.5 means indistinguishable; 0.56/0.64/0.71 are the
   conventional small/medium/large thresholds. It is the effect size that
   pairs naturally with Mann-Whitney, and it is interpretable without knowing
   the units of hypervolume.

3. **No uncertainty on the estimates.** A mean over 2 seeds with no interval is
   not a result. We report bootstrap percentile confidence intervals and refuse
   to test when there are too few seeds to say anything.

A note on what the test can and cannot tell you: it compares *hypervolumes*, so
it inherits every assumption baked into the objective set and its normalisation
(see :mod:`mrna_design.metrics.objective_spec`). Two runs are only comparable if
they used the same objective set; this module checks that when the information
is available and refuses to compare across mismatched spaces.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# Minimum runs per group before a hypothesis test is meaningful. With n = m = 3
# the smallest attainable one-sided Mann-Whitney p-value is 0.05, so anything
# below that cannot reach significance no matter how large the effect.
MIN_SEEDS_FOR_TEST = 3
RECOMMENDED_SEEDS = 10


def load_summary(summary_path: Path | str) -> list[dict]:
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark summary not found at {path}")
    with open(path) as f:
        return json.load(f)


# ── Effect size ───────────────────────────────────────────────────────────────


def vargha_delaney_a12(treatment: list[float], control: list[float]) -> float:
    """
    Vargha-Delaney A12: P(treatment > control) + 0.5 * P(treatment == control).

    Interpretation
    --------------
    0.5  -> the two are indistinguishable
    >0.5 -> treatment tends to win; <0.5 -> control tends to win

    Conventional magnitude thresholds (Vargha & Delaney, 2000):
    |A12 - 0.5| >= 0.06 small, >= 0.14 medium, >= 0.21 large.
    """
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    if t.size == 0 or c.size == 0:
        return float("nan")
    greater = float(np.sum(t[:, None] > c[None, :]))
    equal = float(np.sum(t[:, None] == c[None, :]))
    return (greater + 0.5 * equal) / (t.size * c.size)


def a12_magnitude(a12: float) -> str:
    """Label an A12 value with its conventional magnitude."""
    if not np.isfinite(a12):
        return "undefined"
    delta = abs(a12 - 0.5)
    if delta < 0.06:
        return "negligible"
    if delta < 0.14:
        return "small"
    if delta < 0.21:
        return "medium"
    return "large"


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))


# ── Multiple-comparison correction ────────────────────────────────────────────


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """
    Holm-Bonferroni step-down correction.

    Sort p-values ascending; compare the k-th smallest against
    ``alpha / (n - k)``. Stop at the first failure and reject nothing after it.
    Controls the family-wise error rate under arbitrary dependence.

    Returns one record per input p-value, in the original order.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    out: list[dict | None] = [None] * n
    still_rejecting = True
    for k, idx in enumerate(order):
        threshold = alpha / (n - k)
        rejected = still_rejecting and p_values[idx] <= threshold
        if not rejected:
            still_rejecting = False
        out[idx] = {
            "p_value": float(p_values[idx]),
            "holm_threshold": float(threshold),
            "significant": bool(rejected),
            "rank": k + 1,
            "n_comparisons": n,
        }
    return out  # type: ignore[return-value]


# ── Main comparison ───────────────────────────────────────────────────────────


def _collect(summary: list[dict]) -> tuple[dict, set[str], list[str]]:
    """Group hypervolumes by (target, controller) and check objective-set consistency."""
    hvs: dict = defaultdict(lambda: defaultdict(list))
    objective_sets: set[str] = set()
    warnings: list[str] = []

    for entry in summary:
        if "error" in entry or "hypervolume" not in entry:
            continue
        hvs[entry["target"]][entry["controller"]].append(float(entry["hypervolume"]))
        oset = entry.get("objective_set")
        if oset:
            objective_sets.add(oset)

    if len(objective_sets) > 1:
        warnings.append(
            f"Runs used different objective sets ({sorted(objective_sets)}). "
            f"Hypervolumes from different objective spaces are not comparable; "
            f"the tests below are invalid until the benchmark is re-run in one space."
        )
    return hvs, objective_sets, warnings


def compare_hypervolumes(
    summary: list[dict],
    baseline_controller: str = "random",
    alpha: float = 0.05,
) -> dict:
    """
    Compare each controller's hypervolume against a baseline.

    For every (target, controller) pair this reports:
      - mean and median hypervolume with a bootstrap 95% CI
      - one-sided Mann-Whitney U p-value (H1: controller > baseline)
      - Vargha-Delaney A12 effect size with a magnitude label
      - Holm-Bonferroni corrected significance across the whole family of tests

    Returns
    -------
    dict with keys ``per_target``, ``warnings`` and ``meta``.
    """
    hvs, objective_sets, warnings = _collect(summary)

    # Pass 1: compute the descriptive and raw inferential statistics.
    records: list[dict] = []
    results: dict[str, dict] = {}

    for target, by_controller in sorted(hvs.items()):
        baseline = by_controller.get(baseline_controller)
        target_result: dict[str, dict] = {}

        if baseline is None:
            warnings.append(
                f"Target {target!r} has no runs for baseline controller "
                f"{baseline_controller!r}; skipped."
            )
            results[target] = {}
            continue

        for controller, values in sorted(by_controller.items()):
            lo, hi = bootstrap_ci(values)
            record = {
                "n_runs": len(values),
                "mean_hv": float(np.mean(values)),
                "median_hv": float(np.median(values)),
                "std_hv": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci95_low": lo,
                "ci95_high": hi,
                "is_baseline": controller == baseline_controller,
            }

            if controller != baseline_controller:
                record["baseline_mean_hv"] = float(np.mean(baseline))
                record["a12"] = vargha_delaney_a12(values, baseline)
                record["a12_magnitude"] = a12_magnitude(record["a12"])
                # Retained for backwards compatibility with existing scripts.
                record["effect_size"] = (record["mean_hv"] - record["baseline_mean_hv"]) / (
                    abs(record["baseline_mean_hv"]) + 1e-12
                )

                if len(values) < MIN_SEEDS_FOR_TEST or len(baseline) < MIN_SEEDS_FOR_TEST:
                    record["p_value"] = None
                    record["significant"] = False
                    record["test_skipped"] = (
                        f"Need at least {MIN_SEEDS_FOR_TEST} runs per group for a "
                        f"one-sided Mann-Whitney test to be able to reach p < {alpha}; "
                        f"have {len(values)} vs {len(baseline)}."
                    )
                else:
                    _, p = stats.mannwhitneyu(values, baseline, alternative="greater")
                    record["p_value"] = float(p)
                    records.append({"target": target, "controller": controller, "p": float(p)})
            target_result[controller] = record
        results[target] = target_result

    # Pass 2: correct across the whole family of tests, then write results back.
    corrected = holm_bonferroni([r["p"] for r in records], alpha=alpha)
    for meta, corr in zip(records, corrected, strict=True):
        rec = results[meta["target"]][meta["controller"]]
        rec["significant"] = corr["significant"]
        rec["holm_threshold"] = corr["holm_threshold"]
        rec["n_comparisons"] = corr["n_comparisons"]
        rec["significant_uncorrected"] = rec["p_value"] <= alpha

    n_runs = [len(v) for by_ctrl in hvs.values() for v in by_ctrl.values()]
    if n_runs and min(n_runs) < RECOMMENDED_SEEDS:
        warnings.append(
            f"Smallest group has {min(n_runs)} run(s). Non-parametric tests over "
            f"fewer than {RECOMMENDED_SEEDS} seeds have very low power; treat "
            f"non-significant results as inconclusive rather than as evidence of "
            f"no effect."
        )

    return {
        "per_target": results,
        "warnings": warnings,
        "meta": {
            "baseline_controller": baseline_controller,
            "alpha": alpha,
            "correction": "holm-bonferroni",
            "effect_size": "vargha-delaney A12",
            "n_tests": len(records),
            "objective_sets": sorted(objective_sets),
        },
    }


def format_comparison_table(comparison: dict) -> str:
    """Render :func:`compare_hypervolumes` output as a plain-text table."""
    lines: list[str] = []
    meta = comparison.get("meta", {})
    lines.append(
        f"Baseline: {meta.get('baseline_controller')}   "
        f"alpha={meta.get('alpha')}   correction={meta.get('correction')}   "
        f"tests={meta.get('n_tests')}"
    )
    header = (
        f"{'target':<10}{'controller':<14}{'n':>3}{'mean HV':>12}"
        f"{'95% CI':>24}{'A12':>7}{'magnitude':>12}{'p':>10}{'sig':>5}"
    )
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    for target, by_controller in comparison.get("per_target", {}).items():
        for controller, r in by_controller.items():
            ci = f"[{r['ci95_low']:.4g}, {r['ci95_high']:.4g}]"
            if r.get("is_baseline"):
                a12 = mag = p = sig = "-"
            else:
                a12 = f"{r['a12']:.3f}"
                mag = r["a12_magnitude"]
                p = "-" if r.get("p_value") is None else f"{r['p_value']:.4f}"
                sig = "yes" if r.get("significant") else "no"
            lines.append(
                f"{target:<10}{controller:<14}{r['n_runs']:>3}{r['mean_hv']:>12.5g}"
                f"{ci:>24}{a12:>7}{mag:>12}{p:>10}{sig:>5}"
            )
    for w in comparison.get("warnings", []):
        lines.append("")
        lines.append(f"WARNING: {w}")
    return "\n".join(lines)
