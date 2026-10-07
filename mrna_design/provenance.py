"""
Run provenance — everything needed to reproduce or audit a result.

A benchmark number is only as trustworthy as the record of how it was produced.
Two runs of this pipeline can differ because of the random seed, the objective
set, the thresholds, the package versions, or — most easily missed — whether an
optional external tool was actually present. ViennaRNA, BLAST+, RNAhybrid and
LinearDesign all degrade *silently*: if they are missing the pipeline logs a
warning and carries on with a zero or a fallback. A run without ViennaRNA still
produces a hypervolume; it is simply not the hypervolume you think it is.

Every run therefore writes a manifest recording the code version, the package
versions, and which external tools were actually available. If a reviewer asks
"was ViennaRNA installed when you produced Figure 3?", the answer is in the file
next to the figure.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

_PACKAGES = [
    "ViennaRNA",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "pymoo",
    "pydantic",
    "matplotlib",
    "seaborn",
    "biopython",
    "torch",
    "transformers",
    "openai",
    "anthropic",
]

_EXTERNAL_TOOLS = ["RNAfold", "LinearFold", "LinearDesign", "blastn", "RNAhybrid", "makeblastdb"]


@lru_cache(maxsize=1)
def git_revision() -> dict:
    """Current git commit and working-tree cleanliness, if this is a repo."""
    repo_root = Path(__file__).resolve().parent.parent

    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    sha = _run("rev-parse", "HEAD")
    if sha is None:
        return {"available": False}
    status = _run("status", "--porcelain")
    return {
        "available": True,
        "commit": sha,
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "uncommitted_files": len(status.splitlines()) if status else 0,
    }


@lru_cache(maxsize=1)
def package_versions() -> dict:
    """Installed versions of the packages that can change numerical results."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


@lru_cache(maxsize=1)
def external_tools() -> dict:
    """
    Which optional external binaries are on PATH.

    These all fail soft inside the pipeline, so their absence changes results
    without raising. Recording availability makes that visible after the fact.
    """
    tools = {name: shutil.which(name) for name in _EXTERNAL_TOOLS}

    # ViennaRNA is normally used through its Python bindings, not the binary.
    try:
        import RNA  # noqa: F401

        vienna_bindings = True
    except ImportError:
        vienna_bindings = False

    return {
        "on_path": {k: (v is not None) for k, v in tools.items()},
        "paths": tools,
        "viennarna_python_bindings": vienna_bindings,
    }


def database_availability() -> dict:
    """Whether the optional sequence databases have been downloaded."""
    from mrna_design.metrics import safety

    mirbase = Path(safety._MIRBASE_FASTA)
    gencode = Path(str(safety._GENCODE_DB) + ".nhr")
    return {
        "mirbase_fasta": {"path": str(mirbase), "present": mirbase.exists()},
        "gencode_blast_db": {"path": str(gencode), "present": gencode.exists()},
    }


def environment_manifest() -> dict:
    """Full environment record, embedded in every run manifest."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git_revision(),
        "packages": package_versions(),
        "external_tools": external_tools(),
        "databases": database_availability(),
    }


def warnings_for_missing_tools() -> list[str]:
    """
    Human-readable warnings about silently-degrading capabilities.

    Printed by the CLI before a benchmark so that a run producing meaningless
    structure metrics announces itself up front instead of in a log file.
    """
    msgs: list[str] = []
    tools = external_tools()
    dbs = database_availability()

    if not tools["viennarna_python_bindings"]:
        msgs.append(
            "ViennaRNA Python bindings are NOT installed. All structure metrics "
            "(MFE, ensemble diversity, start-codon unpairing) will be zero or "
            "missing, and any hypervolume computed over them is meaningless. "
            "Install with: pip install ViennaRNA"
        )
    if not dbs["mirbase_fasta"]["present"]:
        msgs.append(
            "miRBase FASTA not found. The miRNA seed scanner will report zero "
            "hits for every candidate, so that objective is constant and adds "
            "no information. Run: make db"
        )
    if not dbs["gencode_blast_db"]["present"]:
        msgs.append(
            "GENCODE BLAST database not found. Off-target BLAST checks will be "
            "skipped. Run: make db"
        )
    if not tools["on_path"].get("LinearDesign"):
        msgs.append(
            "LinearDesign binary not on PATH. The LinearDesign seed silently "
            "falls back to cai_max, so that baseline is NOT being tested."
        )
    return msgs
