"""
CLI command for training and evaluating the surrogate model.

Appended to mrna_design/cli.py as a new command group.
Run with:
    mrna-design surrogate train
    mrna-design surrogate eval
    mrna-design surrogate importances
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group("surrogate")
def surrogate_group() -> None:
    """Surrogate model commands."""
    pass


@surrogate_group.command("train")
@click.option("--data", default=None, help="Path to OpenVaccine train.json (auto-downloaded if absent).")
@click.option("--out", "-o", default="data/surrogate/model.pkl", show_default=True)
@click.option("--n-estimators", default=200, show_default=True)
@click.option("--cv", default=5, show_default=True, help="Cross-validation folds (0 to skip).")
@click.option("--force", is_flag=True, default=False, help="Retrain even if cached model exists.")
@click.option("--use-rnafm", is_flag=True, default=False,
              help="Prepend RNA-FM 640-dim embeddings to hand-crafted features (requires GPU + transformers).")
def train(data, out, n_estimators, cv, force, use_rnafm) -> None:
    """Train the surrogate model on OpenVaccine data."""
    from mrna_design.surrogate.openvaccine import load_openvaccine
    from mrna_design.surrogate.model import SurrogateModel

    console.print("[bold]Loading OpenVaccine data...[/bold]")
    records = load_openvaccine(path=data)
    console.print(f"  → {len(records)} records loaded")

    model = SurrogateModel(n_estimators=n_estimators, use_rnafm=use_rnafm)
    label = "RF + GB + RNA-FM" if use_rnafm else "RF + GB"
    console.print(f"[bold]Training surrogate ({label}, {n_estimators} estimators each)...[/bold]")
    scores = model.fit(records, cv_folds=cv)

    table = Table(title="Surrogate Cross-Validation R²")
    table.add_column("Target", style="cyan")
    table.add_column("R²", justify="right", style="green")
    for k, v in scores.items():
        table.add_row(k.replace("_r2", ""), f"{v:.4f}")
    console.print(table)

    path = model.save(out)
    console.print(f"[bold green]Model saved to {path}[/bold green]")


@surrogate_group.command("eval")
@click.option("--model-path", default=None, help="Path to pickled model (default: data/surrogate/model.pkl).")
@click.option("--data", default=None, help="Path to OpenVaccine train.json.")
@click.option("--n", default=20, show_default=True, help="Number of test records to show.")
def eval_cmd(model_path, data, n) -> None:
    """Evaluate the surrogate on held-out data samples."""
    from mrna_design.surrogate.model import SurrogateModel
    from mrna_design.surrogate.openvaccine import load_openvaccine, records_to_arrays
    from mrna_design.surrogate.features import extract

    model = SurrogateModel.load(model_path)
    records = load_openvaccine(path=data, max_records=n * 5)

    # Evaluate on last n records (rough hold-out)
    test = records[-n:]
    X, y_react, y_deg = records_to_arrays(test, extract)

    react_pred = (model._pipes["rf_react"].predict(X) + model._pipes["gb_react"].predict(X)) / 2.0
    deg_pred   = (model._pipes["rf_deg"].predict(X)   + model._pipes["gb_deg"].predict(X))   / 2.0

    import numpy as np
    from sklearn.metrics import r2_score
    r2_r = r2_score(y_react, react_pred)
    r2_d = r2_score(y_deg, deg_pred)

    table = Table(title=f"Surrogate Predictions on {len(test)} samples")
    table.add_column("ID", style="dim")
    table.add_column("React actual", justify="right")
    table.add_column("React pred", justify="right")
    table.add_column("Deg actual", justify="right")
    table.add_column("Deg pred", justify="right")
    for i, r in enumerate(test):
        table.add_row(
            r.sequence_id[:12],
            f"{y_react[i]:.3f}", f"{react_pred[i]:.3f}",
            f"{y_deg[i]:.3f}",   f"{deg_pred[i]:.3f}",
        )
    console.print(table)
    console.print(f"[bold]R² reactivity:[/bold] {r2_r:.4f}  |  [bold]R² degradation:[/bold] {r2_d:.4f}")


@surrogate_group.command("importances")
@click.option("--model-path", default=None)
@click.option("--target", default="react", type=click.Choice(["react", "deg"]))
def importances(model_path, target) -> None:
    """Print feature importances from the Random Forest surrogate."""
    from mrna_design.surrogate.model import SurrogateModel

    model = SurrogateModel.load(model_path)
    imp = model.feature_importances(target=target)
    sorted_imp = sorted(imp.items(), key=lambda x: -x[1])

    table = Table(title=f"Feature Importances ({target})")
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", justify="right")
    for feat, val in sorted_imp:
        table.add_row(feat, f"{val:.4f}")
    console.print(table)
