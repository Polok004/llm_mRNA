"""
Statistical analysis tools for mRNA design benchmarks.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

def load_summary(summary_path: Path | str) -> list[dict]:
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark summary not found at {path}")
    with open(path) as f:
        return json.load(f)

def compare_hypervolumes(summary: list[dict], baseline_controller: str = "random") -> dict:
    """
    Compare hypervolumes of all controllers against a baseline using Mann-Whitney U test.
    """
    hvs = defaultdict(lambda: defaultdict(list))
    for entry in summary:
        if "hypervolume" in entry and "error" not in entry:
            target = entry["target"]
            ctrl = entry["controller"]
            hvs[target][ctrl].append(entry["hypervolume"])

    results = {}
    for target, ctrl_data in hvs.items():
        if baseline_controller not in ctrl_data:
            continue
        
        baseline_hvs = ctrl_data[baseline_controller]
        if len(baseline_hvs) < 2:
            continue
            
        target_res = {}
        for ctrl, vals in ctrl_data.items():
            if ctrl == baseline_controller or len(vals) < 2:
                continue
            
            # Perform Mann-Whitney U test (non-parametric, suitable for small n)
            # Alternative="greater" tests if ctrl > baseline
            stat, p_val = stats.mannwhitneyu(vals, baseline_hvs, alternative="greater")
            
            mean_baseline = np.mean(baseline_hvs)
            mean_ctrl = np.mean(vals)
            effect_size = (mean_ctrl - mean_baseline) / (mean_baseline + 1e-9)
            
            target_res[ctrl] = {
                "mean_hv": float(mean_ctrl),
                "baseline_mean_hv": float(mean_baseline),
                "effect_size": float(effect_size),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05)
            }
        results[target] = target_res
        
    return results
