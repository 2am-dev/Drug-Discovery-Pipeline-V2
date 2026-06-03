"""
agents/__init__.py — Public re-exports for the agents package.

All agent classes are imported from their respective modules here,
giving the rest of the codebase a single stable import path:

    from agents import PlannerAgent, RetrieverAgent, HypothesisAgent
    from agents import MoleculeDesignerAgent, DockingEvaluatorAgent
    from agents import SynthesisEvaluatorAgent, ReportCompilerAgent

Note on SynthesisEvaluatorAgent
────────────────────────────────
The synthesis evaluator is an optional module controlled by
FeatureFlags.ENABLE_CHEMICAL_SYNTHESIS. It is imported with a try/except
here so that a missing or broken synthesis module never prevents the rest
of the pipeline from loading.
"""

from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.hypothesis import HypothesisAgent
from agents.molecule_designer import MoleculeDesignerAgent
from agents.docking_evaluator import DockingEvaluatorAgent
from agents.report_compiler import ReportCompilerAgent

__all__ = [
    "PlannerAgent",
    "RetrieverAgent",
    "HypothesisAgent",
    "MoleculeDesignerAgent",
    "DockingEvaluatorAgent",
    "ReportCompilerAgent",
]

try:
    from agents.synthesis_evaluator import SynthesisEvaluatorAgent
    __all__.append("SynthesisEvaluatorAgent")
except ImportError as _exc:
    import warnings
    warnings.warn(
        f"SynthesisEvaluatorAgent could not be imported: {_exc}. "
        "Set ENABLE_CHEMICAL_SYNTHESIS=false to suppress this warning.",
        ImportWarning,
        stacklevel=2,
    )