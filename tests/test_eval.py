"""
Tests for evaluation utilities.
"""

from __future__ import annotations

import pytest

from mrna_design.eval.plots import (
    load_hv_histories,
    plot_hypervolume_boxplots,
    plot_hypervolume_progression,
)
from mrna_design.eval.stats import compare_hypervolumes


@pytest.fixture
def mock_summary():
    return [
        {
            "target": "EGFP",
            "controller": "random",
            "seed": 42,
            "hypervolume": 0.5,
            "hv_history": [
                {"iteration": 0, "hv": 0.1},
                {"iteration": 10, "hv": 0.3},
                {"iteration": 50, "hv": 0.5},
            ],
        },
        {
            "target": "EGFP",
            "controller": "random",
            "seed": 43,
            "hypervolume": 0.55,
            "hv_history": [
                {"iteration": 0, "hv": 0.1},
                {"iteration": 10, "hv": 0.35},
                {"iteration": 50, "hv": 0.55},
            ],
        },
        {
            "target": "EGFP",
            "controller": "rule_based",
            "seed": 42,
            "hypervolume": 0.8,
            "hv_history": [
                {"iteration": 0, "hv": 0.1},
                {"iteration": 10, "hv": 0.6},
                {"iteration": 50, "hv": 0.8},
            ],
        },
        {
            "target": "EGFP",
            "controller": "rule_based",
            "seed": 43,
            "hypervolume": 0.85,
            "hv_history": [
                {"iteration": 0, "hv": 0.1},
                {"iteration": 10, "hv": 0.65},
                {"iteration": 50, "hv": 0.85},
            ],
        },
    ]


def test_compare_hypervolumes(mock_summary):
    report = compare_hypervolumes(mock_summary, baseline_controller="random")
    assert set(report) == {"per_target", "warnings", "meta"}
    assert report["meta"]["correction"] == "holm-bonferroni"
    res = report["per_target"]

    assert "EGFP" in res
    assert "rule_based" in res["EGFP"]

    rb_stats = res["EGFP"]["rule_based"]
    assert rb_stats["mean_hv"] == 0.825
    assert rb_stats["baseline_mean_hv"] == 0.525
    assert rb_stats["effect_size"] > 0
    assert "p_value" in rb_stats
    assert "significant" in rb_stats


def test_load_hv_histories(mock_summary):
    histories = load_hv_histories(mock_summary)

    assert "EGFP" in histories
    assert "random" in histories["EGFP"]
    assert "rule_based" in histories["EGFP"]

    random_runs = histories["EGFP"]["random"]
    assert len(random_runs) == 2

    iters, hvs = random_runs[0]
    assert len(iters) == 3
    assert iters[-1] == 50
    assert hvs[-1] == 0.5


def test_plot_hypervolume_progression(mock_summary, tmp_path):
    plot_hypervolume_progression(mock_summary, tmp_path)

    expected_file = tmp_path / "hv_progression_EGFP.png"
    assert expected_file.exists()


def test_plot_hypervolume_boxplots(mock_summary, tmp_path):
    plot_hypervolume_boxplots(mock_summary, tmp_path)

    expected_file = tmp_path / "hv_boxplots.png"
    assert expected_file.exists()
