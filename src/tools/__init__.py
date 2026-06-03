"""
tools/__init__.py — Public re-exports for all pipeline tool modules.

Tools are stateless utility classes used by agents to interact with
external APIs, run subprocesses, and calculate molecular properties.
Unlike agents, tools do not call the LLM directly — they provide
structured data that agents pass to the LLM.

Import pattern:
    from tools import LiteratureSearch, PatentSearch, TargetLookup
    from tools import DockingTool, MoleculePropertyCalculator
    from tools import SynthesisChecker   # optional
"""

from tools.literature_search import LiteratureSearch
from tools.patent_search import PatentSearch
from tools.target_lookup import TargetLookup
from tools.docking import DockingTool
from tools.molecule_generator import MoleculePropertyCalculator

__all__ = [
    "LiteratureSearch",
    "PatentSearch",
    "TargetLookup",
    "DockingTool",
    "MoleculePropertyCalculator",
]

# SynthesisChecker imported conditionally — requires RDKit SA score module
try:
    from tools.synthesis_checker import SynthesisChecker
    __all__.append("SynthesisChecker")
except ImportError:
    pass