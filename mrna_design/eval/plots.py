"""
Plotting utilities for mRNA design benchmarks.
"""
from __future__ import annotations

import json
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
    histories = defaultdict(lambda: defaultdict(list))
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
                    color=line.get_color()
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
            data.append({
                "Target": entry["target"],
                "Controller": entry["controller"],
                "Hypervolume": entry["hypervolume"]
            })
            
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
