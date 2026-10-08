"""
Packaging invariants: the package must import with only its core dependencies.

Why this exists
---------------
Optional extras are a promise: ``pip install mrna-design`` should give you a
working pipeline, and ``[llm]``, ``[surrogate]``, ``[rnafm]`` and ``[ui]`` should
add capability rather than be required for the package to load at all.

That promise was broken twice, and both breaks were invisible in a developer
environment where every extra happened to be installed:

* ``controller/llm_client.py`` imported ``tenacity`` (the ``[llm]`` extra) at
  module level, and ``controller/__init__.py`` imports ``LLMController``. So
  ``import mrna_design.controller`` — needed for the rule-based controller, the
  NSGA-II baseline and the Pareto archive — raised ``ModuleNotFoundError``
  without the LLM extra. This failed CI on every push.
* ``surrogate/model.py`` imported scikit-learn (the ``[surrogate]`` extra) at
  module level, with the same consequence for ``import mrna_design.surrogate``.

A static check is the right tool here: it catches the defect in any environment,
including one where every extra is installed, which is exactly where a runtime
import test would pass and tell you nothing.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "mrna_design"

#: Top-level module name -> the extra that provides it.
OPTIONAL_DEPENDENCIES: dict[str, str] = {
    "tenacity": "llm",
    "openai": "llm",
    "anthropic": "llm",
    "google": "llm",
    "sklearn": "surrogate",
    "torch": "rnafm",
    "transformers": "rnafm",
    "streamlit": "ui",
}


def _module_level_imports(path: pathlib.Path) -> list[tuple[str, int]]:
    """
    Return ``(top_level_module, lineno)`` for every import executed at import time.

    Only ``col_offset == 0`` counts: an import nested inside a function, a
    ``try``/``except ImportError`` guard or an ``if TYPE_CHECKING`` block is
    indented, and does not run (or does not fail) when the module is imported.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        # Check the node type first: ast.walk also yields Module, which has no
        # col_offset at all.
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if node.col_offset != 0:
            continue
        if isinstance(node, ast.Import):
            found += [(alias.name.split(".")[0], node.lineno) for alias in node.names]
        elif node.level == 0 and node.module:
            found.append((node.module.split(".")[0], node.lineno))
    return found


SOURCE_FILES = sorted(PACKAGE_ROOT.rglob("*.py"))


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: str(p.name))
def test_no_optional_dependency_imported_at_module_level(path: pathlib.Path) -> None:
    """No module may require an optional extra just to be imported."""
    offenders = [
        (name, lineno, OPTIONAL_DEPENDENCIES[name])
        for name, lineno in _module_level_imports(path)
        if name in OPTIONAL_DEPENDENCIES
    ]
    if offenders:
        detail = "; ".join(
            f"{name} (extra [{extra}]) at line {lineno}" for name, lineno, extra in offenders
        )
        pytest.fail(
            f"{path.relative_to(PACKAGE_ROOT.parent)} imports an optional dependency "
            f"at module level: {detail}. Move it inside the function that needs it, "
            f"or guard it with try/except ImportError, so the package still imports "
            f"without that extra."
        )


def test_core_subpackages_import_without_optional_extras() -> None:
    """
    The subpackages the pipeline cannot run without must import cleanly.

    This is the runtime half of the check. It only proves anything in an
    environment missing the extras, but it costs nothing and documents intent.
    """
    import importlib

    for name in (
        "mrna_design.controller",
        "mrna_design.baselines",
        "mrna_design.metrics",
        "mrna_design.models",
        "mrna_design.validators",
        "mrna_design.designer",
        "mrna_design.controllers",
        "mrna_design.budget",
        "mrna_design.provenance",
        "mrna_design.optimize",
    ):
        importlib.import_module(name)


def test_every_declared_extra_is_real() -> None:
    """Each extra named in the test map actually exists in pyproject.toml."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        pytest.skip("tomllib requires Python 3.11+")

    pyproject = PACKAGE_ROOT.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    declared = set(data["project"]["optional-dependencies"])
    referenced = set(OPTIONAL_DEPENDENCIES.values())
    missing = referenced - declared
    assert not missing, f"Extras referenced by this test but absent from pyproject: {missing}"
