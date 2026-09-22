"""
mrna_design — Agentic multi-objective mRNA sequence design pipeline.

Public re-exports for the most commonly used symbols.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mrna-design")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = ["__version__"]
