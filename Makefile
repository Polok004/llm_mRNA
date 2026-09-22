.PHONY: env install test lint fmt bench db clean

# ── Environment ──────────────────────────────────────────────────────────────
env:
	python -m venv .venv
	@echo "Run: source .venv/bin/activate"

install:
	pip install -e ".[dev,surrogate]"

install-llm:
	pip install -e ".[dev,surrogate,llm]"

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

# ── Data ──────────────────────────────────────────────────────────────────────
db:
	bash scripts/setup_databases.sh

# ── Benchmarks ────────────────────────────────────────────────────────────────
bench-quick:
	mrna-design bench \
	  --targets data/targets/benchmark_proteins.json \
	  --controllers rule_based,cai_max,random \
	  --iters 20 \
	  --out results/quick/

bench-full:
	mrna-design bench \
	  --targets data/targets/benchmark_proteins.json \
	  --controllers rule_based,cai_max,ga,random \
	  --iters 100 \
	  --seeds 5 \
	  --out results/full/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache dist build *.egg-info
