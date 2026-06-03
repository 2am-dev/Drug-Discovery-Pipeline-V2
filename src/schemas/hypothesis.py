"""
schemas/hypothesis.py — Pydantic v2 schemas for the Hypothesis agent.

The Hypothesis agent selects the best target from the Retriever's candidates
and formulates a mechanistic drug discovery hypothesis.

Expected LLM JSON output example:
──────────────────────────────────
{
  "selected_target": {
    "gene_name": "EGFR",
    "uniprot_id": "P00533",
    "pdb_id": "1M17",
    "binding_site_residues": ["L718", "V726", "A743", "M793"],
    "target_class": "kinase",
    "disease_relevance": "Activating mutations in EGFR drive proliferation in NSCLC."
  },
  "hypothesis": {
    "mechanism": "Selective inhibition of the EGFR ATP-binding pocket prevents
                  autophosphorylation, blocking RAS-MAPK and PI3K-AKT cascades.",
    "rationale": "Clinical data from 15 Phase III trials demonstrates OS benefit
                  with first-generation TKIs; third-generation agents overcome
                  T790M resistance.",
    "therapeutic_modality": "small_molecule",
    "confidence_score": 0.87
  },
  "alternative_targets": [
    {
      "gene_name": "ALK",
      "uniprot_id": "Q9UM73",
      "rationale": "ALK fusions present in ~5% of NSCLC; approved inhibitors exist."
    }
  ]
}
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── Valid target classes ───────────────────────────────────────────────────────
TargetClass = Literal[
    "kinase",
    "GPCR",
    "protease",
    "nuclear_receptor",
    "ion_channel",
    "phosphatase",
    "epigenetic",
    "transcription_factor",
    "E3_ligase",
    "other",
]

# ── Valid therapeutic modalities ───────────────────────────────────────────────
TherapeuticModality = Literal[
    "small_molecule",
    "antibody",
    "PROTAC",
    "antisense_oligonucleotide",
    "siRNA",
    "peptide",
    "other",
]


class SelectedTarget(BaseModel):
    """
    The single primary drug target selected by the Hypothesis agent.

    Contains all structural and biological information needed by the
    Molecule Designer and Docking Evaluator agents.
    """

    gene_name: str = Field(
        ...,
        description="HGNC-approved gene symbol in uppercase (e.g. 'EGFR').",
        min_length=1,
        max_length=20,
    )
    uniprot_id: str = Field(
        ...,
        description=(
            "UniProt accession number (e.g. 'P00533'). "
            "Required — the Hypothesis agent must confirm this from UniProt data."
        ),
        min_length=5,
        max_length=10,
    )
    pdb_id: str = Field(
        ...,
        description=(
            "PDB ID of the best available crystal structure for docking "
            "(e.g. '1M17'). Must be a 4-character PDB identifier."
        ),
        min_length=4,
        max_length=4,
    )
    binding_site_residues: list[str] = Field(
        ...,
        description=(
            "Key residues in the target's binding site relevant to drug design. "
            "Format: single-letter amino acid + chain position (e.g. 'M793', 'L718'). "
            "Include 3-8 residues."
        ),
        min_length=1,
        max_length=20,
    )
    target_class: TargetClass = Field(
        ...,
        description=(
            "Structural/functional class of the target. "
            "Determines the drug design strategy used by the Molecule Designer."
        ),
    )
    disease_relevance: str = Field(
        ...,
        description=(
            "One sentence explaining why this target is relevant to the "
            "disease indication."
        ),
        min_length=10,
        max_length=500,
    )

    @field_validator("gene_name", mode="before")
    @classmethod
    def uppercase_gene(cls, v: str) -> str:
        """Ensure gene names are uppercase."""
        return v.strip().upper()

    @field_validator("pdb_id", mode="before")
    @classmethod
    def uppercase_pdb(cls, v: str) -> str:
        """Ensure PDB IDs are uppercase."""
        return v.strip().upper()

    @field_validator("uniprot_id", mode="before")
    @classmethod
    def uppercase_uniprot(cls, v: str) -> str:
        """Ensure UniProt IDs are uppercase."""
        return v.strip().upper()

    @field_validator("binding_site_residues", mode="before")
    @classmethod
    def coerce_residues(cls, v: object) -> list:
        """
        Accept None or string (space-separated) as a list.

        Args:
            v: Raw residues value from LLM.

        Returns:
            list[str]: List of residue strings.
        """
        if v is None:
            return []
        if isinstance(v, str):
            # Handle comma-separated or space-separated strings
            import re
            return [r.strip() for r in re.split(r"[,\s]+", v) if r.strip()]
        return list(v)

    @field_validator("target_class", mode="before")
    @classmethod
    def normalise_target_class(cls, v: str) -> str:
        """
        Normalise target class to a valid literal value.

        Maps common synonyms to canonical values.

        Args:
            v: Raw target class string from LLM.

        Returns:
            str: Normalised target class.
        """
        synonyms = {
            "protein kinase": "kinase",
            "receptor tyrosine kinase": "kinase",
            "rtk": "kinase",
            "g protein coupled receptor": "GPCR",
            "g-protein coupled receptor": "GPCR",
            "gpcr": "GPCR",
            "serine protease": "protease",
            "cysteine protease": "protease",
            "aspartyl protease": "protease",
            "nuclear receptor": "nuclear_receptor",
            "nr": "nuclear_receptor",
            "ion channel": "ion_channel",
            "channel": "ion_channel",
            "ubiquitin ligase": "E3_ligase",
            "e3 ligase": "E3_ligase",
            "e3-ligase": "E3_ligase",
            "histone deacetylase": "epigenetic",
            "hdac": "epigenetic",
            "bromodomain": "epigenetic",
            "tf": "transcription_factor",
        }
        normalised = synonyms.get(v.lower().strip(), v.strip())
        valid = {
            "kinase", "GPCR", "protease", "nuclear_receptor",
            "ion_channel", "phosphatase", "epigenetic",
            "transcription_factor", "E3_ligase", "other",
        }
        if normalised not in valid:
            return "other"
        return normalised

    def to_prompt_dict(self) -> dict:
        """
        Return a compact dict for inclusion in Molecule Designer prompt.

        Returns:
            dict: Essential target information for molecule design.
        """
        return {
            "gene_name": self.gene_name,
            "uniprot_id": self.uniprot_id,
            "pdb_id": self.pdb_id,
            "binding_site_residues": self.binding_site_residues,
            "target_class": self.target_class,
            "disease_relevance": self.disease_relevance,
        }


class HypothesisDetail(BaseModel):
    """
    The mechanistic hypothesis formulated for the selected target.

    Provides the scientific rationale for pursuing the selected target
    with a small-molecule therapeutic approach.
    """

    mechanism: str = Field(
        ...,
        description=(
            "Detailed mechanistic hypothesis: how inhibiting/modulating the "
            "target would produce a therapeutic effect. 2-4 sentences. "
            "Should reference the target's specific biological role."
        ),
        min_length=50,
        max_length=2000,
    )
    rationale: str = Field(
        ...,
        description=(
            "Evidence-based rationale citing key studies, clinical data, or "
            "genetic evidence that supports the hypothesis. 2-3 sentences."
        ),
        min_length=30,
        max_length=2000,
    )
    therapeutic_modality: TherapeuticModality = Field(
        ...,
        description=(
            "Proposed therapeutic modality. 'small_molecule' is the default "
            "for this pipeline; other options are supported for report completeness."
        ),
    )
    confidence_score: float = Field(
        ...,
        description=(
            "Confidence in the hypothesis (0.0-1.0). "
            "Based on strength of clinical/genetic evidence. "
            "0.9+ = strong genetic validation; 0.7-0.9 = solid preclinical; "
            "0.5-0.7 = early/circumstantial; <0.5 = speculative."
        ),
        ge=0.0,
        le=1.0,
    )

    @field_validator("therapeutic_modality", mode="before")
    @classmethod
    def normalise_modality(cls, v: str) -> str:
        """
        Normalise therapeutic modality to a valid literal value.

        Args:
            v: Raw modality string from LLM.

        Returns:
            str: Normalised modality.
        """
        synonyms = {
            "small molecule": "small_molecule",
            "small-molecule": "small_molecule",
            "smi": "small_molecule",
            "monoclonal antibody": "antibody",
            "mab": "antibody",
            "antibody drug conjugate": "antibody",
            "adc": "antibody",
            "targeted protein degradation": "PROTAC",
            "protac": "PROTAC",
            "antisense": "antisense_oligonucleotide",
            "aso": "antisense_oligonucleotide",
            "sirna": "siRNA",
            "si rna": "siRNA",
        }
        normalised = synonyms.get(v.lower().strip(), v.strip())
        valid = {
            "small_molecule", "antibody", "PROTAC",
            "antisense_oligonucleotide", "siRNA", "peptide", "other",
        }
        if normalised not in valid:
            return "small_molecule"   # default for this pipeline
        return normalised

    @field_validator("confidence_score", mode="before")
    @classmethod
    def coerce_confidence(cls, v: object) -> float:
        """
        Clamp confidence score to [0.0, 1.0].

        Some LLMs return values like 87 instead of 0.87.

        Args:
            v: Raw confidence value.

        Returns:
            float: Clamped score between 0.0 and 1.0.
        """
        try:
            score = float(v)
            # Handle percentage values (e.g. 87 → 0.87)
            if score > 1.0:
                score = score / 100.0
            return max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            return 0.5   # neutral default


class AlternativeTarget(BaseModel):
    """
    A backup drug target considered but not selected as the primary candidate.

    Included in the hypothesis output for scientific completeness and
    to support a fallback strategy if the primary target fails.
    """

    gene_name: str = Field(
        ...,
        description="HGNC-approved gene symbol in uppercase.",
        min_length=1,
        max_length=20,
    )
    uniprot_id: Optional[str] = Field(
        default=None,
        description="UniProt accession (may be null if not confirmed).",
    )
    rationale: str = Field(
        ...,
        description=(
            "Brief explanation of why this target was considered and "
            "why it was not selected as the primary (1-2 sentences)."
        ),
        min_length=10,
        max_length=500,
    )

    @field_validator("gene_name", mode="before")
    @classmethod
    def uppercase_gene(cls, v: str) -> str:
        """Ensure gene names are uppercase."""
        return v.strip().upper()

    @field_validator("uniprot_id", mode="before")
    @classmethod
    def coerce_uniprot(cls, v: object) -> Optional[str]:
        """Accept null-like strings as None."""
        if v is None:
            return None
        s = str(v).strip()
        if s.lower() in {"null", "none", "n/a", "unknown", ""}:
            return None
        return s.upper()


class HypothesisResponse(BaseModel):
    """
    Full output schema for the Hypothesis agent.

    This is the central scientific output of the pipeline — it defines
    the target and mechanism that all subsequent agents build upon.
    """

    selected_target: SelectedTarget = Field(
        ...,
        description="The single best drug target selected from the retriever candidates.",
    )
    hypothesis: HypothesisDetail = Field(
        ...,
        description="The mechanistic hypothesis for targeting the selected target.",
    )
    alternative_targets: list[AlternativeTarget] = Field(
        default_factory=list,
        description=(
            "2-3 alternative targets considered but not selected. "
            "Included for scientific due diligence."
        ),
        max_length=5,
    )

    @field_validator("alternative_targets", mode="before")
    @classmethod
    def coerce_alternatives(cls, v: object) -> list:
        """Accept None as empty list."""
        if v is None:
            return []
        return v

    def to_prompt_dict(self) -> dict:
        """
        Return a compact dict for inclusion in Molecule Designer prompt.

        Strips the alternative targets (not needed downstream) and
        truncates long text fields.

        Returns:
            dict: Token-efficient representation for prompt injection.
        """
        return {
            "selected_target": self.selected_target.to_prompt_dict(),
            "mechanism": self.hypothesis.mechanism[:400],
            "confidence_score": self.hypothesis.confidence_score,
            "therapeutic_modality": self.hypothesis.therapeutic_modality,
        }

    model_config = {
        "json_schema_extra": {
            "example": {
                "selected_target": {
                    "gene_name": "EGFR",
                    "uniprot_id": "P00533",
                    "pdb_id": "1M17",
                    "binding_site_residues": ["L718", "V726", "A743", "M793"],
                    "target_class": "kinase",
                    "disease_relevance": (
                        "Activating EGFR mutations drive proliferation in NSCLC."
                    ),
                },
                "hypothesis": {
                    "mechanism": (
                        "Selective inhibition of the EGFR ATP-binding pocket "
                        "prevents autophosphorylation of Y1068 and Y1173, "
                        "blocking downstream RAS-MAPK and PI3K-AKT signalling."
                    ),
                    "rationale": (
                        "15 Phase III trials demonstrate OS benefit with EGFR TKIs "
                        "in EGFR-mutant NSCLC. Third-generation agents overcome "
                        "T790M resistance via covalent C797 modification."
                    ),
                    "therapeutic_modality": "small_molecule",
                    "confidence_score": 0.87,
                },
                "alternative_targets": [
                    {
                        "gene_name": "ALK",
                        "uniprot_id": "Q9UM73",
                        "rationale": (
                            "ALK fusions present in ~5% of NSCLC; approved "
                            "inhibitors exist, reducing novelty."
                        ),
                    }
                ],
            }
        }
    }