"""
Controller registry — one place that knows how to build every controller.

The CLI and the UI must construct controllers identically, or a run launched
from one is not comparable with a run launched from the other. Both import
:func:`build_controller` from here rather than each keeping its own mapping.

Note that controllers do not take a uniform constructor signature: some are
seeded, some are stateless, the LLM controller needs a model name, and NSGA-II
needs to know which objective space it is being scored in (otherwise the
baseline optimises something different from what the archive measures). This
function hides that asymmetry behind one call.
"""

from __future__ import annotations

from mrna_design.controller.base import BaseController
from mrna_design.metrics.objective_spec import DEFAULT_OBJECTIVE_SET, ObjectiveSet

#: Controller names accepted by the CLI and the UI, in the order they are shown.
CONTROLLER_CHOICES: list[str] = ["rule_based", "nsga2", "ga", "cai_max", "random", "llm"]

#: One-line descriptions, used for CLI help text and UI tooltips.
CONTROLLER_DESCRIPTIONS: dict[str, str] = {
    "rule_based": "Diagnostic-guided heuristic edits (the non-LLM agent)",
    "nsga2": "True multi-objective GA: non-dominated sorting + crowding distance",
    "ga": "Single-objective GA, fitness = CAI only (naive reference)",
    "cai_max": "Greedy highest-frequency codon substitution",
    "random": "Random synonymous substitution, budget-matched (the floor)",
    "llm": "Language model reads diagnostics and proposes targeted edits",
}


def build_controller(
    name: str,
    rng_seed: int = 42,
    llm_model: str = "gpt-4o",
    objective_set: ObjectiveSet | str = DEFAULT_OBJECTIVE_SET,
) -> BaseController:
    """
    Instantiate a controller by name.

    Parameters
    ----------
    name
        One of :data:`CONTROLLER_CHOICES`.
    rng_seed
        Seed for the controllers that are stochastic. Ignored by the
        deterministic ones.
    llm_model
        Model identifier, used only by the ``llm`` controller.
    objective_set
        The objective space the run will be scored in. Passed to NSGA-II so it
        sorts in the same space the archive measures.

    Raises
    ------
    ValueError
        If ``name`` is not a known controller.
    """
    from mrna_design.baselines import (
        CaiMaxController,
        GeneticAlgorithmController,
        NSGA2Controller,
        RandomController,
    )
    from mrna_design.controller.llm import LLMController
    from mrna_design.controller.rule_based import RuleBasedController

    if name == "rule_based":
        return RuleBasedController(rng_seed=rng_seed)
    if name == "random":
        return RandomController(rng_seed=rng_seed)
    if name == "ga":
        return GeneticAlgorithmController(rng_seed=rng_seed)
    if name == "nsga2":
        return NSGA2Controller(rng_seed=rng_seed, objective_set=objective_set)
    if name == "cai_max":
        return CaiMaxController()
    if name == "llm":
        return LLMController(model_name=llm_model)
    raise ValueError(f"Unknown controller {name!r}. Available: {', '.join(CONTROLLER_CHOICES)}")
