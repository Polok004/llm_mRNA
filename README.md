# Agentic mRNA Design Pipeline

> **Research question:** Does an LLM controller that reads region-level diagnostics and proposes targeted synonymous edits reach a better Pareto front than non-LLM optimizers under the same tool-call budget?

## Overview

A multi-objective mRNA sequence optimization pipeline with:
- **Multi-objective optimisation**: CAI, MFE, GC content, immunogenicity, miRNA off-targets.
- **Agentic architecture**: Structure, Safety, and Immunogenicity agents produce structured JSON diagnostics; a pluggable Refinement Controller proposes codon edits.
- **Hard guarantees**: The LLM never writes nucleotides — it proposes edit operations from a synonymous-codon menu; a deterministic validator enforces protein identity.
- **Honest ablation**: Rule-based, GA, CAI-max, random, and LinearDesign baselines; LLM controller added in Weeks 7–8.

## Quick Start

```bash
# 1. Create venv and install
python -m venv .venv && source .venv/bin/activate
make install

# 2. Download databases (miRBase, GENCODE, makeblastdb)
make db

# 3. Run a quick optimisation on EGFP (rule-based controller, 20 iterations)
mrna-design optimize \
  --protein EGFP \
  --controller rule_based \
  --iters 20 \
  --out results/egfp_test/

# 4. Run the full benchmark
make bench-full
```

## Project Structure

```
mrna_design/
├── models/         # Pydantic data models (Candidate, ObjectiveScores, …)
├── validators/     # Codon table, protein-identity guard, GC/stop validators
├── metrics/        # CAI, ViennaRNA structure, immunogenicity, safety (miRNA/BLAST)
├── designer/       # Seeding strategies (CAI-max, GC-balanced, harmonisation, LinearDesign)
├── agents/         # Structure, Safety, Immunogenicity agents → structured JSON diagnostics
├── controller/     # BaseController, RuleBasedController, ParetoArchive
├── baselines/      # GA, random search, CAI-max optimiser
├── evaluation/     # Benchmark runner, statistics, plots
├── optimize.py     # Main optimisation loop
└── cli.py          # Click CLI entry points
data/
├── codon_tables/   # Human codon usage frequencies
└── targets/        # Benchmark protein sequences (EGFP, Luc, EPO)
scripts/
└── setup_databases.sh   # Downloads miRBase + GENCODE, builds BLAST DBs
tests/              # pytest suite
notebooks/          # Jupyter exploratory notebooks
results/            # Benchmark outputs (gitignored)
logs/               # JSON event logs (gitignored)
```

## Dependencies

- **ViennaRNA** ≥ 2.6 (Python bindings via pip)
- **BLAST+** (installed separately — see `scripts/setup_databases.sh`)
- **Biopython**, NumPy, pandas, pymoo, Pydantic v2, Click, Rich

See `pyproject.toml` for full list.

## Baselines

| Baseline | Description |
|---|---|
| Wild-type | Original CDS, no changes |
| CAI-max | Greedy highest-frequency human codon |
| GC-balanced | Random codon sampling targeting 55 % GC |
| Codon harmonised | Matches human codon usage ratios |
| LinearDesign | (Optional) Joint MFE + CAI DP solver |
| Random search | Random synonymous substitutions (budget-matched) |
| GA | Genetic algorithm, same budget |
| Rule-based agent | Diagnostic-guided heuristic edits (this work) |

## Citation

If you use this pipeline, please cite: *(paper in preparation)*.

## License

MIT
