"""
schemas/__init__.py — Public re-exports for all Pydantic v2 schema models.

Every agent imports its input/output schemas from here, giving a single
stable import path across the codebase:

    from schemas import HypothesisResponse, MoleculeDesignResponse
    from schemas import DockingResponse, ReportResponse
    from schemas import PlannerResponse, RetrieverResponse

Design principles
─────────────────
- All schemas use Pydantic v2 with strict type validation.
- Every field has a description (used in JSON schema generation and error messages).
- Optional fields use Optional[T] with explicit None defaults so that
  partial LLM responses can still be validated.
- All schemas implement a .to_prompt_dict() method that strips verbose fields
  before the dict is serialised into the next agent's prompt, keeping
  inter-agent token usage minimal.
- Numeric fields use field validators to clamp values to valid ranges
  (e.g. confidence scores to [0.0, 1.0]).
"""

# ── Planner schemas ───────────────────────────────────────────────────────────
from schemas.planner import (
    PipelinePhase,
    PlannerResponse,
)

# ── Retriever schemas ─────────────────────────────────────────────────────────
from schemas.retriever import (
    TargetCandidate,
    BatchSummary,
    RetrieverResponse,
)

# ── Hypothesis schemas ────────────────────────────────────────────────────────
from schemas.hypothesis import (
    SelectedTarget,
    HypothesisDetail,
    AlternativeTarget,
    HypothesisResponse,
)

# ── Molecule schemas ──────────────────────────────────────────────────────────
from schemas.molecule import (
    GeneratedMolecule,
    LLMMoleculeOutput,
    FilteringCriteria,
    MoleculeDesignResponse,
    SynthesisRoute,
    SynthesisResponse,
)

# ── Docking schemas ───────────────────────────────────────────────────────────
from schemas.docking import (
    DockingResult,
    DockingResponse,
    MockDockingResult,
)

# ── Report schemas ────────────────────────────────────────────────────────────
from schemas.report import (
    ReportMetadata,
    ReportSection,
    ReportSections,
    ReportResponse,
)

__all__ = [
    # Planner
    "PipelinePhase",
    "PlannerResponse",
    # Retriever
    "TargetCandidate",
    "BatchSummary",
    "RetrieverResponse",
    # Hypothesis
    "SelectedTarget",
    "HypothesisDetail",
    "AlternativeTarget",
    "HypothesisResponse",
    # Molecule
    "GeneratedMolecule",
    "FilteringCriteria",
    "LLMMoleculeOutput",
    "MoleculeDesignResponse",
    "SynthesisRoute",
    "SynthesisResponse",
    # Docking
    "DockingResult",
    "DockingResponse",
    "MockDockingResult",
    # Report
    "ReportMetadata",
    "ReportSection",
    "ReportSections",
    "ReportResponse",
]