"""
BaseController — abstract interface for all refinement controllers.

All controllers (rule-based, LLM, GA-wrapper) implement this interface.
The optimisation loop calls only `propose_edits()`; the controller type is
swappable without changing any downstream code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mrna_design.models.candidate import Candidate
from mrna_design.models.diagnostics import RegionDiagnostic
from mrna_design.models.edits import EditProposal


class BaseController(ABC):
    """Abstract refinement controller."""

    name: str = "base"

    @abstractmethod
    def propose_edits(
        self,
        candidate: Candidate,
        diagnostics: list[RegionDiagnostic],
        history: list[EditProposal],
        max_edits: int = 5,
        iteration: int = 0,
    ) -> EditProposal:
        """
        Propose a set of synonymous codon edits for `candidate`.

        Parameters
        ----------
        candidate : Candidate
            The current candidate (scores and diagnostics already populated).
        diagnostics : list[RegionDiagnostic]
            Structured diagnostics from all agents (merged).
        history : list[EditProposal]
            All proposals made in previous iterations (for loop detection).
        max_edits : int
            Maximum number of edits to propose in one call.
        iteration : int
            Current optimisation iteration (for logging).

        Returns
        -------
        EditProposal
            The proposed edits. The CodonApplicator will validate and apply them.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state between runs. Override if needed."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
