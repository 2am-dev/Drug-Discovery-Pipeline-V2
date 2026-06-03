"""
schemas/planner.py — Pydantic v2 schemas for the Planner agent.

The Planner agent produces a task graph that describes which pipeline
phases should run, in what order, and with what dependencies.

Expected LLM JSON output example:
──────────────────────────────────
{
  "task_id": "a3f8c2d1-4e56-7890-abcd-ef1234567890",
  "disease_or_target": "non-small cell lung cancer",
  "phases": [
    {"phase": "retrieval",       "status": "pending", "dependencies": []},
    {"phase": "hypothesis",      "status": "pending", "dependencies": ["retrieval"]},
    {"phase": "molecule_design", "status": "pending", "dependencies": ["hypothesis"]},
    {"phase": "docking",         "status": "pending", "dependencies": ["molecule_design"]},
    {"phase": "synthesis",       "status": "skipped", "dependencies": ["molecule_design"]},
    {"phase": "report",          "status": "pending", "dependencies": ["docking"]}
  ],
  "estimated_duration_minutes": 45,
  "pipeline_notes": "Docking enabled; synthesis skipped per configuration."
}
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Valid phase names ─────────────────────────────────────────────────────────
PhaseName = Literal[
    "retrieval",
    "hypothesis",
    "molecule_design",
    "docking",
    "synthesis",
    "report",
]

# ── Valid phase statuses ──────────────────────────────────────────────────────
PhaseStatus = Literal["pending", "running", "complete", "skipped", "failed"]


class PipelinePhase(BaseModel):
    """
    Represents a single phase in the drug discovery pipeline task graph.

    Each phase has a name, an initial status, and a list of phases that
    must complete before this phase can begin.
    """

    phase: PhaseName = Field(
        ...,
        description=(
            "Unique phase identifier. One of: retrieval, hypothesis, "
            "molecule_design, docking, synthesis, report."
        ),
    )
    status: PhaseStatus = Field(
        default="pending",
        description=(
            "Execution status. Use 'skipped' when the phase is disabled "
            "via feature flags. Use 'pending' for phases that will run."
        ),
    )
    dependencies: list[PhaseName] = Field(
        default_factory=list,
        description=(
            "List of phase names that must complete before this phase starts. "
            "Empty list means the phase has no dependencies (runs first)."
        ),
    )

    @field_validator("dependencies", mode="before")
    @classmethod
    def coerce_dependencies(cls, v: object) -> list:
        """
        Accept None or missing dependencies as an empty list.

        Args:
            v: Raw dependency value from JSON.

        Returns:
            list: Empty list if None, otherwise the original value.
        """
        if v is None:
            return []
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "phase": "docking",
                "status": "pending",
                "dependencies": ["molecule_design"],
            }
        }
    }


class PlannerResponse(BaseModel):
    """
    Full response schema for the Planner agent.

    This is the first JSON output produced by the pipeline. It defines
    the execution graph for all subsequent agents and is logged to
    outputs/pipeline_log.jsonl at the start of every run.
    """

    task_id: str = Field(
        ...,
        description="UUID4 string uniquely identifying this pipeline run.",
        min_length=1,
    )
    disease_or_target: str = Field(
        ...,
        description=(
            "The disease indication (e.g. 'non-small cell lung cancer') or "
            "biological target (e.g. 'EGFR') provided as pipeline input."
        ),
        min_length=1,
    )
    phases: list[PipelinePhase] = Field(
        ...,
        description=(
            "Ordered list of all pipeline phases. Must contain exactly 6 "
            "phases: retrieval, hypothesis, molecule_design, docking, "
            "synthesis, report."
        ),
        min_length=6,
        max_length=6,
    )
    estimated_duration_minutes: int = Field(
        ...,
        description=(
            "Estimated total pipeline run time in minutes. "
            "Sum of per-phase estimates for enabled phases only."
        ),
        ge=1,
        le=300,
    )
    pipeline_notes: Optional[str] = Field(
        default=None,
        description=(
            "Optional free-text notes about the planned execution, such as "
            "which features are enabled or disabled."
        ),
    )

    @field_validator("phases", mode="after")
    @classmethod
    def validate_phase_names(cls, phases: list[PipelinePhase]) -> list[PipelinePhase]:
        """
        Ensure all six required phases are present in the response.

        Args:
            phases: List of PipelinePhase objects.

        Returns:
            list[PipelinePhase]: Validated phase list.

        Raises:
            ValueError: If any required phase is missing.
        """
        required = {
            "retrieval", "hypothesis", "molecule_design",
            "docking", "synthesis", "report",
        }
        present = {p.phase for p in phases}
        missing = required - present
        if missing:
            raise ValueError(
                f"Missing required pipeline phases: {sorted(missing)}. "
                f"All 6 phases must be present (use status='skipped' to disable)."
            )
        return phases

    @model_validator(mode="after")
    def validate_dependency_graph(self) -> "PlannerResponse":
        """
        Verify that all listed dependencies reference valid phase names.

        Raises:
            ValueError: If any dependency references a non-existent phase.
        """
        valid_phases = {p.phase for p in self.phases}
        for phase in self.phases:
            for dep in phase.dependencies:
                if dep not in valid_phases:
                    raise ValueError(
                        f"Phase '{phase.phase}' has unknown dependency '{dep}'. "
                        f"Valid phases: {sorted(valid_phases)}"
                    )
        return self

    def get_phase(self, name: PhaseName) -> Optional[PipelinePhase]:
        """
        Look up a phase by name.

        Args:
            name: Phase name to look up.

        Returns:
            PipelinePhase | None: Matching phase or None.
        """
        for phase in self.phases:
            if phase.phase == name:
                return phase
        return None

    def is_enabled(self, name: PhaseName) -> bool:
        """
        Check whether a phase is scheduled to run (i.e. not skipped).

        Args:
            name: Phase name to check.

        Returns:
            bool: True if the phase status is 'pending' (will run).
        """
        phase = self.get_phase(name)
        return phase is not None and phase.status == "pending"

    def to_prompt_dict(self) -> dict:
        """
        Return a minimal dict representation for inclusion in agent prompts.

        Strips large or redundant fields to conserve token budget.

        Returns:
            dict: Slim representation with task_id, disease_or_target,
                  enabled phases, and estimated_duration_minutes.
        """
        return {
            "task_id": self.task_id,
            "disease_or_target": self.disease_or_target,
            "enabled_phases": [
                p.phase for p in self.phases if p.status == "pending"
            ],
            "skipped_phases": [
                p.phase for p in self.phases if p.status == "skipped"
            ],
            "estimated_duration_minutes": self.estimated_duration_minutes,
        }

    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "a3f8c2d1-4e56-7890-abcd-ef1234567890",
                "disease_or_target": "non-small cell lung cancer",
                "phases": [
                    {"phase": "retrieval", "status": "pending", "dependencies": []},
                    {"phase": "hypothesis", "status": "pending",
                     "dependencies": ["retrieval"]},
                    {"phase": "molecule_design", "status": "pending",
                     "dependencies": ["hypothesis"]},
                    {"phase": "docking", "status": "pending",
                     "dependencies": ["molecule_design"]},
                    {"phase": "synthesis", "status": "skipped",
                     "dependencies": ["molecule_design"]},
                    {"phase": "report", "status": "pending",
                     "dependencies": ["docking"]},
                ],
                "estimated_duration_minutes": 43,
                "pipeline_notes": "Synthesis disabled; docking enabled.",
            }
        }
    }