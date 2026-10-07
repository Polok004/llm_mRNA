"""
Plotting utilities for mRNA design benchmarks.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_hv_histories(summary: list[dict]) -> dict:
    """
    Load HV histories grouped by target and controller.
    Returns: dict[target][controller] = list of (iter_arrays, hv_arrays) for each seed
    """
    histories: dict = defaultdict(lambda: defaultdict(list))
    for entry in summary:
        if "hv_history" in entry:
            target = entry["target"]
            ctrl = entry["controller"]
            history = entry["hv_history"]

            iters = np.array([h["iteration"] for h in history])
            hvs = np.array([h["hv"] for h in history])
            histories[target][ctrl].append((iters, hvs))

    return histories


def plot_hypervolume_progression(summary: list[dict], out_dir: Path | str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = load_hv_histories(summary)

    for target, ctrl_data in histories.items():
        plt.figure(figsize=(10, 6))

        for ctrl, seed_runs in ctrl_data.items():
            # Interpolate to a common iteration grid for mean/std
            max_iter = max(max(iters) for iters, _ in seed_runs if len(iters) > 0)
            common_iters = np.arange(0, max_iter + 1)

            interp_hvs = []
            for iters, hvs in seed_runs:
                if len(iters) > 0:
                    interp_hv = np.interp(common_iters, iters, hvs)
                    interp_hvs.append(interp_hv)

            if interp_hvs:
                mean_hv = np.mean(interp_hvs, axis=0)
                std_hv = np.std(interp_hvs, axis=0)

                line = plt.plot(common_iters, mean_hv, label=ctrl)[0]
                plt.fill_between(
                    common_iters,
                    mean_hv - std_hv,
                    mean_hv + std_hv,
                    alpha=0.2,
                    color=line.get_color(),
                )

        plt.title(f"Hypervolume Progression: {target}")
        plt.xlabel("Iteration")
        plt.ylabel("Hypervolume")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"hv_progression_{target}.png", dpi=300)
        plt.close()


def plot_hypervolume_boxplots(summary: list[dict], out_dir: Path | str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flatten data for seaborn
    data = []
    for entry in summary:
        if "hypervolume" in entry and "error" not in entry:
            data.append(
                {
                    "Target": entry["target"],
                    "Controller": entry["controller"],
                    "Hypervolume": entry["hypervolume"],
                }
            )

    if not data:
        return

    import pandas as pd

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Target", y="Hypervolume", hue="Controller")
    plt.title("Final Hypervolume Distributions")
    plt.tight_layout()
    plt.savefig(out_dir / "hv_boxplots.png", dpi=300)
    plt.close()


def plot_hypervolume_vs_evaluations(summary: list[dict], out_dir: Path | str) -> None:
    """
    Hypervolume against evaluation budget spent — the comparable convergence plot.

    ``plot_hypervolume_progression`` uses iteration index on the x-axis, which is
    not a fair basis for comparing controllers: they generate different numbers of
    candidates per iteration, so equal iteration counts do not mean equal work.
    Evaluations are the resource the benchmark actually holds fixed.

    Each controller's runs are interpolated onto a common budget grid and drawn as
    a mean with a +/-1 s.d. band. A run that stops early is held flat at its final
    value, which is the honest extension: it did not get worse, it simply stopped
    finding new sequences to evaluate.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces: dict = defaultdict(lambda: defaultdict(list))
    for entry in summary:
        if "hv_history" not in entry or "error" in entry:
            continue
        history = entry["hv_history"]
        if not history or "evaluations" not in history[0]:
            continue
        evals = np.array([h["evaluations"] for h in history], dtype=float)
        hvs = np.array([h.get("hypervolume", h.get("hv", 0.0)) for h in history], dtype=float)
        traces[entry["target"]][entry["controller"]].append((evals, hvs))

    for target, by_controller in traces.items():
        plt.figure(figsize=(9, 5.5))
        budgets = [float(e.max()) for runs in by_controller.values() for e, _ in runs if len(e)]
        if not budgets:
            plt.close()
            continue
        grid = np.linspace(0, max(budgets), 120)

        for controller, runs in sorted(by_controller.items()):
            curves = [np.interp(grid, evals, hvs) for evals, hvs in runs if len(evals)]
            if not curves:
                continue
            mean = np.mean(curves, axis=0)
            sd = np.std(curves, axis=0)
            line = plt.plot(grid, mean, label=f"{controller} (n={len(curves)})")[0]
            plt.fill_between(grid, mean - sd, mean + sd, alpha=0.18, color=line.get_color())

        plt.title(f"Hypervolume vs evaluation budget: {target}")
        plt.xlabel("Objective evaluations spent")
        plt.ylabel("Hypervolume (normalised objective space)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"hv_vs_evaluations_{target}.png", dpi=200)
        plt.close()
