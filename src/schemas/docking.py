"""
schemas/docking.py — Pydantic v2 schemas for the Docking Evaluator agent.

The Docking Evaluator runs AutoDock Vina (or generates mock scores when
Vina is unavailable) and produces binding affinity predictions for the
top-shortlisted molecules.

Expected LLM JSON output example (DockingResponse):
────────────────────────────────────────────────────
{
  "docking_results": [
    {
      "smiles": "CCN(CC)CCNC(=O)c1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1",
      "binding_affinity_kcal_mol": -8.4,
      "ligand_efficiency": 0.35,
      "pose_file": "outputs/poses/pose_001.pdbqt",
      "key_interactions": ["H-bond with Met793", "Pi-stacking with Phe723"],
      "binding_mode_summary": "Quinazoline ring occupies the adenine binding pocket.",
      "rank": 1
    }
  ],
  "lead_compound_smiles": "CCN(CC)CCNC(=O)c1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1",
  "lead_compound_rationale": "Best affinity (-8.4 kcal/mol) with favourable LE.",
  "receptor_pdb": "1M17",
  "docking_software": "AutoDock Vina 1.2.3"
}
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class DockingResult(BaseModel):
    """
    Docking result for a single ligand-receptor pair.

    Contains both Vina-calculated binding affinity and LLM-interpreted
    interaction analysis. The LLM is used to identify key interactions
    from Vina's output (log file + pose PDBQT), not to calculate scores.
    """

    smiles: str = Field(
        ...,
        description="SMILES string of the docked ligand.",
        min_length=2,
    )
    binding_affinity_kcal_mol: float = Field(
        ...,
        description=(
            "Best docking binding affinity in kcal/mol. "
            "More negative = stronger predicted binding. "
            "Typical drug-like range: -6 to -12 kcal/mol."
        ),
        ge=-25.0,
        le=0.0,
    )
    ligand_efficiency: float = Field(
        ...,
        description=(
            "Ligand efficiency = |binding_affinity| / heavy_atom_count. "
            "Values > 0.3 kcal/mol/atom are considered good. "
            "Corrects for molecule size."
        ),
        ge=0.0,
        le=2.0,
    )
    pose_file: Optional[str] = Field(
        default=None,
        description=(
            "Path to the output PDBQT pose file. "
            "Null if docking was mocked (Vina not available)."
        ),
    )
    key_interactions: list[str] = Field(
        default_factory=list,
        description=(
            "Key intermolecular interactions predicted for the best pose. "
            "Format: '<interaction type> with <residue>' "
            "(e.g. 'H-bond with Met793', 'Hydrophobic contact with Leu718')."
        ),
        max_length=10,
    )
    binding_mode_summary: Optional[str] = Field(
        default=None,
        description=(
            "1-2 sentence description of how the ligand binds in the receptor. "
            "Which pocket it occupies, how it orients."
        ),
        max_length=500,
    )
    rank: int = Field(
        ...,
        description=(
            "Rank among all docked compounds (1 = best binding affinity). "
            "Used for shortlisting in the report."
        ),
        ge=1,
    )
    is_mock: bool = Field(
        default=False,
        description=(
            "True if this result uses mock/estimated scores rather than "
            "real Vina docking (set when Vina binary is not available)."
        ),
    )

    @field_validator("smiles", mode="before")
    @classmethod
    def strip_smiles(cls, v: str) -> str:
        """Strip whitespace from SMILES."""
        return v.strip()

    @field_validator("binding_affinity_kcal_mol", mode="before")
    @classmethod
    def coerce_affinity(cls, v: object) -> float:
        """
        Ensure affinity is negative and within realistic range.

        Some LLMs may return positive values (forgetting the sign convention).
        We negate positive values to maintain the convention that lower = better.

        Args:
            v: Raw affinity value.

        Returns:
            float: Negative binding affinity.
        """
        try:
            val = float(v)
            # If positive, negate (assume LLM forgot the minus sign)
            if val > 0:
                val = -val
            return max(-25.0, min(0.0, val))
        except (TypeError, ValueError):
            return -6.0   # Default: weak but plausible

    @field_validator("ligand_efficiency", mode="before")
    @classmethod
    def coerce_le(cls, v: object) -> float:
        """Clamp ligand efficiency to [0.0, 2.0]."""
        try:
            return max(0.0, min(2.0, float(v)))
        except (TypeError, ValueError):
            return 0.25

    @field_validator("key_interactions", mode="before")
    @classmethod
    def coerce_interactions(cls, v: object) -> list:
        """Accept None as empty list."""
        if v is None:
            return []
        return v

    def to_prompt_dict(self) -> dict:
        """
        Return a compact dict for Report Compiler prompt.

        Returns:
            dict: Essential docking data for the report section.
        """
        return {
            "smiles": self.smiles,
            "binding_affinity_kcal_mol": self.binding_affinity_kcal_mol,
            "ligand_efficiency": self.ligand_efficiency,
            "key_interactions": self.key_interactions[:5],  # top 5 only
            "rank": self.rank,
            "is_mock": self.is_mock,
        }


class MockDockingResult(BaseModel):
    """
    Simplified schema for mock docking results generated when AutoDock Vina
    is not available.

    Mock scores are estimated using a simple function of molecular properties
    (MW, LogP, QED) with added Gaussian noise for realism. The pipeline
    clearly marks these as mock in the final report.
    """

    smiles: str = Field(..., description="SMILES of the molecule.")
    estimated_affinity_kcal_mol: float = Field(
        ...,
        description=(
            "Estimated binding affinity (mock). "
            "Calculated from molecular properties, NOT from Vina."
        ),
        ge=-15.0,
        le=-4.0,
    )
    estimation_method: str = Field(
        default="property_based_estimate",
        description="Method used to generate the mock score.",
    )
    warning: str = Field(
        default=(
            "MOCK SCORE: AutoDock Vina not available. "
            "This estimate is based on molecular properties only "
            "and should not be used for decision-making."
        ),
        description="Mandatory warning that this is a mock result.",
    )

    def to_docking_result(self, rank: int) -> DockingResult:
        """
        Convert a MockDockingResult to a DockingResult for unified downstream use.

        Args:
            rank: Rank of this compound in the shortlist.

        Returns:
            DockingResult: Full result object with is_mock=True.
        """
        return DockingResult(
            smiles=self.smiles,
            binding_affinity_kcal_mol=self.estimated_affinity_kcal_mol,
            ligand_efficiency=0.0,    # Cannot estimate without heavy atom count
            pose_file=None,
            key_interactions=[],
            binding_mode_summary="Mock result — Vina not available.",
            rank=rank,
            is_mock=True,
        )


class DockingResponse(BaseModel):
    """
    Full output schema for the Docking Evaluator agent.

    Contains docking results for all shortlisted molecules plus the
    identified lead compound for the report.
    """

    docking_results: list[DockingResult] = Field(
        ...,
        description=(
            "Docking results for all shortlisted molecules, "
            "sorted by binding affinity (best first)."
        ),
        min_length=1,
    )
    lead_compound_smiles: str = Field(
        ...,
        description=(
            "SMILES of the best-ranked compound (rank=1), "
            "selected as the lead for the project proposal."
        ),
        min_length=2,
    )
    lead_compound_rationale: str = Field(
        ...,
        description=(
            "1-2 sentence justification for selecting this compound as lead: "
            "affinity, ligand efficiency, interaction profile."
        ),
        min_length=10,
        max_length=500,
    )
    receptor_pdb: str = Field(
        ...,
        description="PDB ID of the receptor structure used for docking.",
        min_length=4,
        max_length=4,
    )
    docking_software: str = Field(
        default="AutoDock Vina",
        description=(
            "Name and version of the docking software used. "
            "Set to 'MockDocking (property-based)' if Vina was unavailable."
        ),
    )
    contains_mock_results: bool = Field(
        default=False,
        description=(
            "True if ANY result in docking_results is a mock score. "
            "Set to True when Vina is not available. "
            "Propagated to the final report as a caveat."
        ),
    )

    @model_validator(mode="after")
    def set_mock_flag(self) -> "DockingResponse":
        """
        Automatically set contains_mock_results based on individual results.

        Returns:
            DockingResponse: Self with contains_mock_results updated.
        """
        self.contains_mock_results = any(r.is_mock for r in self.docking_results)
        return self

    @model_validator(mode="after")
    def validate_lead_in_results(self) -> "DockingResponse":
        """
        Verify that the lead compound SMILES appears in docking_results.

        Raises:
            ValueError: If lead_compound_smiles is not found in any result.
        """
        smiles_set = {r.smiles for r in self.docking_results}
        if self.lead_compound_smiles not in smiles_set:
            # Try to recover by using the rank-1 compound
            rank1 = next(
                (r for r in self.docking_results if r.rank == 1), None
            )
            if rank1:
                object.__setattr__(self, "lead_compound_smiles", rank1.smiles)
            else:
                raise ValueError(
                    f"lead_compound_smiles '{self.lead_compound_smiles}' "
                    f"is not in docking_results SMILES set."
                )
        return self

    @field_validator("receptor_pdb", mode="before")
    @classmethod
    def uppercase_pdb(cls, v: str) -> str:
        """Uppercase PDB IDs."""
        return v.strip().upper()

    @field_validator("docking_results", mode="before")
    @classmethod
    def coerce_results(cls, v: object) -> list:
        """Ensure results is a non-empty list."""
        if v is None:
            return []
        return v

    def get_lead(self) -> DockingResult:
        """
        Return the rank-1 (lead) docking result.

        Returns:
            DockingResult: The best-ranked docking result.

        Raises:
            ValueError: If no rank-1 result exists.
        """
        for result in self.docking_results:
            if result.rank == 1:
                return result
        raise ValueError("No rank-1 docking result found.")

    def to_prompt_dict(self) -> dict:
        """
        Return a compact dict for Report Compiler prompt.

        Returns:
            dict: Essential docking output for the report section.
        """
        lead = None
        for r in self.docking_results:
            if r.rank == 1:
                lead = r
                break

        return {
            "lead_compound_smiles": self.lead_compound_smiles,
            "lead_compound_rationale": self.lead_compound_rationale,
            "lead_binding_affinity": lead.binding_affinity_kcal_mol if lead else None,
            "lead_ligand_efficiency": lead.ligand_efficiency if lead else None,
            "lead_interactions": lead.key_interactions[:5] if lead else [],
            "receptor_pdb": self.receptor_pdb,
            "docking_software": self.docking_software,
            "contains_mock_results": self.contains_mock_results,
            "total_compounds_docked": len(self.docking_results),
        }

    model_config = {
        "json_schema_extra": {
            "example": {
                "docking_results": [
                    {
                        "smiles": (
                            "CCN(CC)CCNC(=O)c1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1"
                        ),
                        "binding_affinity_kcal_mol": -8.4,
                        "ligand_efficiency": 0.35,
                        "pose_file": "outputs/poses/pose_001.pdbqt",
                        "key_interactions": [
                            "H-bond with Met793",
                            "Pi-stacking with Phe723",
                        ],
                        "binding_mode_summary": (
                            "Quinazoline ring occupies the adenine binding pocket "
                            "with amine chain extending toward the solvent region."
                        ),
                        "rank": 1,
                        "is_mock": False,
                    }
                ],
                "lead_compound_smiles": (
                    "CCN(CC)CCNC(=O)c1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1"
                ),
                "lead_compound_rationale": (
                    "Highest binding affinity (-8.4 kcal/mol) with good "
                    "ligand efficiency (0.35) and favourable H-bond network."
                ),
                "receptor_pdb": "1M17",
                "docking_software": "AutoDock Vina 1.2.3",
                "contains_mock_results": False,
            }
        }
    }