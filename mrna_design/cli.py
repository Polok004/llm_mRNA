"""
Click CLI entry points for the mRNA design pipeline.

Commands:
  optimize   Run optimisation on a single protein.
  bench      Run the full benchmark across all targets.
  analyze    Print stats from a saved archive.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from mrna_design.surrogate.cli_commands import surrogate_group

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """Agentic mRNA Design Pipeline."""
    pass


# ── optimize ──────────────────────────────────────────────────────────────────

@click.command()
@click.option("--protein", "-p", required=False, help="Amino acid sequence (single-letter).")
@click.option("--protein-name", default=None, help="Named protein from benchmark targets (e.g. EGFP).")
@click.option("--controller", "-c", default="rule_based",
              type=click.Choice(["rule_based", "cai_max", "random", "ga", "llm"]),
              help="Controller to use.")
@click.option("--iters", "-n", default=50, show_default=True, help="Max optimisation iterations.")
@click.option("--pop", default=10, show_default=True, help="Population size.")
@click.option("--max-edits", default=5, show_default=True, help="Max edits per iteration.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--out", "-o", default="results/", show_default=True, help="Output directory.")
@click.option("--no-safety", is_flag=True, default=False, help="Skip miRNA safety scan.")
@click.option("--no-structure", is_flag=True, default=False, help="Skip ViennaRNA structure computation (for fast dry runs).")
@click.option("--no-lineardesign", is_flag=True, default=False, help="Skip LinearDesign seeding.")
@click.option("--targets-file", default=None, help="Path to benchmark_proteins.json (for --protein-name).")
@click.option("--llm-model", default="gpt-4o", help="Model name if using the llm controller.")
def optimize(
    protein, protein_name, controller, iters, pop, max_edits, seed,
    out, no_safety, no_structure, no_lineardesign, targets_file, llm_model
) -> None:
    """Optimise a single mRNA sequence."""
    from mrna_design.optimize import OptimizeConfig, optimize as _optimize
    from mrna_design.controller.rule_based import RuleBasedController
    from mrna_design.controller.llm import LLMController
    from mrna_design.baselines import CaiMaxController, RandomController, GeneticAlgorithmController

    # Resolve protein sequence
    if protein_name:
        tf = targets_file or str(
            Path(__file__).parent.parent / "data" / "targets" / "benchmark_proteins.json"
        )
        with open(tf) as f:
            targets = json.load(f)["targets"]
        match = next((t for t in targets if t["name"].lower() == protein_name.lower()), None)
        if not match:
            console.print(f"[red]Unknown protein name '{protein_name}'. "
                          f"Available: {[t['name'] for t in targets]}[/red]")
            sys.exit(1)
        protein = match["protein"]
        utr5 = match.get("utr5", "")
        utr3 = match.get("utr3", "")
        console.print(f"[green]Using {protein_name}[/green] ({len(protein)} aa)")
    else:
        if not protein:
            console.print("[red]Provide --protein or --protein-name.[/red]")
            sys.exit(1)
        utr5 = ""
        utr3 = ""

    # Build controller
    ctrl_map = {
        "rule_based": RuleBasedController(rng_seed=seed),
        "cai_max": CaiMaxController(),
        "random": RandomController(rng_seed=seed),
        "ga": GeneticAlgorithmController(rng_seed=seed),
        "llm": LLMController(model_name=llm_model),
    }
    ctrl = ctrl_map[controller]

    cfg = OptimizeConfig(
        max_iters=iters,
        population_k=pop,
        max_edits_per_iter=max_edits,
        rng_seed=seed,
        include_lineardesign=not no_lineardesign,
        run_safety=not no_safety,
        run_structure=not no_structure,
        output_dir=Path(out),
    )

    console.print(f"[bold]Controller:[/bold] {controller}  |  "
                  f"[bold]Iters:[/bold] {iters}  |  [bold]Pop:[/bold] {pop}")

    t0 = time.time()
    archive = _optimize(protein, ctrl, utr5=utr5, utr3=utr3, config=cfg)
    wall = round(time.time() - t0, 1)

    # Summary table
    table = Table(title=f"Pareto Archive ({len(archive)} candidates, {wall}s)")
    table.add_column("ID", style="dim")
    table.add_column("CAI", justify="right")
    table.add_column("MFE", justify="right")
    table.add_column("CpG/100nt", justify="right")
    table.add_column("miRNA hits", justify="right")
    table.add_column("Iter")

    for entry in archive._entries[:20]:
        c = entry.candidate
        s = c.scores
        table.add_row(
            c.sequence_id[:8],
            f"{s.cai:.3f}" if s.cai else "—",
            f"{s.mfe:.1f}" if s.mfe else "—",
            f"{s.cpg_density:.2f}" if s.cpg_density is not None else "—",
            str(s.mirna_seed_hits) if s.mirna_seed_hits is not None else "—",
            str(entry.iteration),
        )

    console.print(table)
    console.print(f"[bold green]Hypervolume:[/bold green] {archive.hypervolume():.6f}")
    console.print(f"[bold]Archive saved to:[/bold] {Path(out) / cfg.run_id}")


# ── bench ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--targets", "-t",
              default="data/targets/benchmark_proteins.json",
              show_default=True,
              help="Path to benchmark targets JSON.")
@click.option("--controllers", default="rule_based,cai_max,random",
              show_default=True,
              help="Comma-separated list of controllers to benchmark.")
@click.option("--iters", default=50, show_default=True)
@click.option("--pop", default=10, show_default=True)
@click.option("--seeds", default=1, show_default=True, help="Number of random seeds.")
@click.option("--out", default="results/bench/", show_default=True)
@click.option("--no-safety", is_flag=True, default=False)
@click.option("--no-structure", is_flag=True, default=False, help="Skip ViennaRNA folding (for fast dry runs).")
@click.option("--llm-model", default="gpt-4o", help="Model name if using the llm controller.")
def bench(targets, controllers, iters, pop, seeds, out, no_safety, no_structure, llm_model) -> None:
    """Run benchmark across multiple proteins and controllers."""
    from mrna_design.optimize import OptimizeConfig, optimize as _optimize
    from mrna_design.controller.rule_based import RuleBasedController
    from mrna_design.controller.llm import LLMController
    from mrna_design.baselines import CaiMaxController, RandomController, GeneticAlgorithmController

    ctrl_classes = {
        "rule_based": RuleBasedController,
        "cai_max": CaiMaxController,
        "random": RandomController,
        "ga": GeneticAlgorithmController,
        "llm": LLMController,
    }

    with open(targets) as f:
        target_list = json.load(f)["targets"]

    controller_names = [c.strip() for c in controllers.split(",")]
    out_dir = Path(out)
    results_summary = []

    for target in target_list:
        for ctrl_name in controller_names:
            for seed_i in range(seeds):
                rng_seed = 42 + seed_i
                ctrl_cls = ctrl_classes.get(ctrl_name)
                if not ctrl_cls:
                    console.print(f"[yellow]Unknown controller '{ctrl_name}', skipping.[/yellow]")
                    continue
                if ctrl_name in ("rule_based", "random", "ga"):
                    ctrl = ctrl_cls(rng_seed=rng_seed)
                elif ctrl_name == "llm":
                    ctrl = ctrl_cls(model_name=llm_model)
                else:
                    ctrl = ctrl_cls()
                cfg = OptimizeConfig(
                    max_iters=iters,
                    population_k=pop,
                    rng_seed=rng_seed,
                    run_safety=not no_safety,
                    run_structure=not no_structure,
                    output_dir=out_dir / target["name"] / ctrl_name / f"seed{rng_seed}",
                )
                console.print(
                    f"[bold]{target['name']}[/bold] | {ctrl_name} | seed={rng_seed}"
                )
                try:
                    archive = _optimize(
                        protein=target["protein"],
                        controller=ctrl,
                        utr5=target.get("utr5", ""),
                        utr3=target.get("utr3", ""),
                        config=cfg,
                    )
                    results_summary.append({
                        "target": target["name"],
                        "controller": ctrl_name,
                        "seed": rng_seed,
                        "archive_size": len(archive),
                        "hypervolume": archive.hypervolume(),
                        "hv_history": [{"iteration": it, "hv": hv}
                                       for it, hv in archive.hv_history()],
                    })
                except Exception as exc:
                    console.print(f"[red]FAILED: {exc}[/red]")
                    results_summary.append({
                        "target": target["name"],
                        "controller": ctrl_name,
                        "seed": rng_seed,
                        "error": str(exc),
                    })

    summary_path = out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    console.print(f"[bold green]Benchmark complete. Summary: {summary_path}[/bold green]")
    
    # ── Evaluation ────────────────────────────────────────────────────────────
    console.print("[bold]Running evaluation (stats and plots)...[/bold]")
    try:
        from mrna_design.eval.stats import compare_hypervolumes
        from mrna_design.eval.plots import plot_hypervolume_progression, plot_hypervolume_boxplots
        
        # 1. Statistical tests (vs random by default, or the first controller if random is not present)
        baseline = "random" if "random" in controller_names else controller_names[-1]
        stats_res = compare_hypervolumes(results_summary, baseline_controller=baseline)
        
        stats_path = out_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats_res, f, indent=2)
        console.print(f"  → Statistical test results saved to {stats_path}")
        
        # 2. Plots
        plots_dir = out_dir / "plots"
        plot_hypervolume_progression(results_summary, plots_dir)
        plot_hypervolume_boxplots(results_summary, plots_dir)
        console.print(f"  → Plots saved to {plots_dir}/")
        
    except ImportError as e:
        console.print(f"[yellow]Skipping evaluation plots (matplotlib/seaborn/pandas required): {e}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")


# ── analyze ───────────────────────────────────────────────────────────────────

@click.command()
@click.argument("archive_path")
def analyze(archive_path: str) -> None:
    """Print statistics for a saved archive (JSONL file)."""
    path = Path(archive_path)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))

    if not entries:
        console.print("[yellow]Archive is empty.[/yellow]")
        return

    table = Table(title=f"Archive — {path.name} ({len(entries)} entries)")
    keys = ["sequence_id", "iteration", "cai", "mfe", "cpg_density",
            "mirna_seed_hits", "gu_motif_count"]
    for k in keys:
        table.add_column(k, justify="right" if k not in ("sequence_id",) else "left")

    for e in entries:
        s = e.get("scores", {})
        table.add_row(
            e.get("sequence_id", "?")[:12],
            str(e.get("iteration", "?")),
            f"{s.get('cai', 0):.3f}",
            f"{s.get('mfe', 0):.1f}",
            f"{s.get('cpg_density', 0):.2f}",
            str(s.get("mirna_seed_hits", "?")),
            str(s.get("gu_motif_count", "?")),
        )
    console.print(table)


main.add_command(optimize)
main.add_command(bench)
main.add_command(analyze)
main.add_command(surrogate_group)

if __name__ == "__main__":
    main()
