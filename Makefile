.PHONY: env install install-llm install-ui ui test test-fast test-regressions lint fmt bench bench-quick bench-full bench-ablation check-env db clean

# ── Environment ──────────────────────────────────────────────────────────────
env:
	python -m venv .venv
	@echo "Run: source .venv/bin/activate"

install:
	pip install -e ".[dev,surrogate]"

install-llm:
	pip install -e ".[dev,surrogate,llm]"

install-ui:
	pip install -e ".[dev,surrogate,ui]"

# ── User interface ────────────────────────────────────────────────────────────
# Local web app: configure a run, watch it converge, explore the Pareto front.
ui:
	streamlit run ui/app.py

# ── Quality ───────────────────────────────────────────────────────────────────
fmt:
	ruff format mrna_design tests

lint:
	ruff check mrna_design tests
	mypy mrna_design

test:
	pytest tests/ -v --tb=short --cov=mrna_design --cov-report=term-missing

test-fast:
	pytest tests/ -v --tb=short -m "not slow"

# Regression tests for previously-fixed defects. Each one fails against the
# pre-upgrade implementation.
test-regressions:
	pytest tests/test_regressions.py -v --tb=short

# Print which optional external tools and databases are available. All of them
# fail soft, so a run can silently measure something other than it claims to.
check-env:
	python -c "from mrna_design.provenance import warnings_for_missing_tools as w; \
	  ws = w(); print('\n'.join('! ' + x for x in ws) if ws else 'All optional tools present.')"

# ── Data ──────────────────────────────────────────────────────────────────────
db:
	bash scripts/setup_databases.sh

# ── Benchmarks ────────────────────────────────────────────────────────────────
#
# --max-evals is the unit of fairness: every controller gets the same number of
# full objective evaluations. Capping by iterations instead is not comparable,
# because controllers differ in how many candidates they generate per iteration.
#
# --seeds 10 is the minimum for the statistics to have usable power. With fewer
# than 3 seeds a one-sided Mann-Whitney test cannot reach p < 0.05 at all,
# whatever the effect size.

bench-quick:
	mrna-design bench \
	  --targets data/targets/benchmark_proteins.json \
	  --controllers rule_based,nsga2,random \
	  --max-evals 200 \
	  --objective-set primary \
	  --seeds 3 \
	  --out results/quick/

bench-full:
	mrna-design bench \
	  --targets data/targets/benchmark_proteins.json \
	  --controllers rule_based,nsga2,ga,cai_max,random \
	  --max-evals 1000 \
	  --objective-set primary \
	  --seeds 10 \
	  --out results/full/

# Ablation: the same comparison in the original nine-objective space, to show
# what high dimensionality does to the front.
bench-ablation:
	mrna-design bench \
	  --targets data/targets/benchmark_proteins.json \
	  --controllers rule_based,nsga2,random \
	  --max-evals 1000 \
	  --objective-set extended \
	  --seeds 10 \
	  --out results/ablation_extended/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache dist build *.egg-info
