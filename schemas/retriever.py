"""
schemas/retriever.py — Pydantic v2 schemas for the Retriever agent.

The Retriever agent mines PubMed, arXiv, and PatentsView to identify
druggable target candidates. It produces two intermediate schemas
(BatchSummary, per batch of 50 abstracts) and one final output schema
(RetrieverResponse) that is passed to the Hypothesis agent.

Expected LLM JSON output example (RetrieverResponse):
──────────────────────────────────────────────────────
{
  "target_candidates": [
    {
      "gene_name": "EGFR",
      "uniprot_id": "P00533",
      "pdb_ids": ["1M17", "2ITY"],
      "evidence_summary": "Overexpressed in 85% of NSCLC; activating mutations
                           drive proliferation via RAS-MAPK and PI3K-AKT.",
      "literature_citations": ["PMID:12345678", "PMID:87654321"],
      "patent_count": 234,
      "druggability_score": 0.92,
      "novelty_score": 0.45
    }
  ],
  "total_papers_reviewed": 150,
  "total_patents_reviewed": 45,
  "retrieval_timestamp": "2025-01-15T10:30:00+00:00"
}
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TargetFinding(BaseModel):
    """
    A single target finding extracted from one abstract batch.

    Used within BatchSummary — not exposed directly to downstream agents.
    """

    gene_name: str = Field(
        ...,
        description=(
            "HGNC-approved gene symbol in uppercase (e.g. 'EGFR', 'KRAS'). "
            "Use the canonical human gene symbol."
        ),
        min_length=1,
        max_length=20,
    )
    evidence_summary: str = Field(
        ...,
        description=(
            "1-2 sentence mechanistic summary of this target's role in the "
            "disease, grounded in the abstract content."
        ),
        min_length=10,
        max_length=500,
    )
    evidence_type: str = Field(
        ...,
        description=(
            "Quality of the supporting evidence. One of: "
            "'clinical', 'preclinical', 'in_vitro', 'review', 'computational'."
        ),
    )
    citations: list[str] = Field(
        default_factory=list,
        description=(
            "PubMed IDs mentioned in the abstract. "
            "Format: 'PMID:XXXXXXXX' (8 digits). May be empty."
        ),
    )

    @field_validator("evidence_type", mode="before")
    @classmethod
    def normalise_evidence_type(cls, v: str) -> str:
        """
        Normalise evidence type to a valid lowercase value.

        Accepts common synonyms and maps them to canonical values.

        Args:
            v: Raw evidence type string from LLM.

        Returns:
            str: Normalised evidence type.
        """
        valid = {"clinical", "preclinical", "in_vitro", "review", "computational"}
        synonyms = {
            "in vitro": "in_vitro",
            "invitro": "in_vitro",
            "in-vitro": "in_vitro",
            "pre-clinical": "preclinical",
            "pre_clinical": "preclinical",
            "literature review": "review",
            "meta-analysis": "review",
            "in silico": "computational",
            "computational": "computational",
        }
        normalised = synonyms.get(v.lower().strip(), v.lower().strip())
        if normalised not in valid:
            # Default to "review" rather than failing validation
            return "review"
        return normalised

    @field_validator("citations", mode="before")
    @classmethod
    def coerce_citations(cls, v: object) -> list:
        """
        Accept None citations as an empty list.

        Args:
            v: Raw citations value.

        Returns:
            list: Empty list or original list.
        """
        if v is None:
            return []
        return v

    @field_validator("gene_name", mode="before")
    @classmethod
    def uppercase_gene_name(cls, v: str) -> str:
        """
        Ensure gene names are uppercase (standard HGNC convention).

        Args:
            v: Raw gene name string.

        Returns:
            str: Uppercased gene name.
        """
        return v.strip().upper()


class BatchSummary(BaseModel):
    """
    Summary output from processing one batch of ~50 literature abstracts.

    The Retriever agent calls the LLM once per batch to extract target
    findings. All batch summaries are then merged and re-ranked by a
    second LLM call that produces the final RetrieverResponse.
    """

    batch_number: int = Field(
        ...,
        description="1-indexed batch number (e.g. 1 of 3).",
        ge=1,
    )
    targets_found: list[TargetFinding] = Field(
        default_factory=list,
        description=(
            "List of druggable targets identified in this batch. "
            "May be empty if the batch contained no relevant papers."
        ),
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 5 key mechanistic findings from this batch, as short "
            "bullet-point strings (max 150 chars each)."
        ),
        max_length=5,
    )
    papers_processed: int = Field(
        ...,
        description="Number of abstracts processed in this batch.",
        ge=0,
    )

    @field_validator("key_findings", mode="before")
    @classmethod
    def coerce_key_findings(cls, v: object) -> list:
        """Accept None as empty list."""
        if v is None:
            return []
        return v

    @field_validator("targets_found", mode="before")
    @classmethod
    def coerce_targets_found(cls, v: object) -> list:
        """Accept None as empty list."""
        if v is None:
            return []
        return v


class TargetCandidate(BaseModel):
    """
    A fully characterised drug target candidate produced by the Retriever.

    This is the primary data object passed to the Hypothesis agent.
    It aggregates evidence from all literature batches and patent search
    into a single, ranked candidate profile.
    """

    gene_name: str = Field(
        ...,
        description="HGNC-approved gene symbol in uppercase (e.g. 'EGFR').",
        min_length=1,
        max_length=20,
    )
    uniprot_id: Optional[str] = Field(
        default=None,
        description=(
            "UniProt accession number (e.g. 'P00533'). "
            "Null if not identified from literature."
        ),
        pattern=r"^[A-Z0-9]{6,10}$",
    )
    pdb_ids: list[str] = Field(
        default_factory=list,
        description=(
            "List of relevant PDB structure IDs for this target "
            "(e.g. ['1M17', '2ITY']). Used by docking agent."
        ),
    )
    evidence_summary: str = Field(
        ...,
        description=(
            "Comprehensive 2-3 sentence summary of the target's role in the "
            "disease indication, synthesised from all literature evidence."
        ),
        min_length=20,
        max_length=1000,
    )
    literature_citations: list[str] = Field(
        default_factory=list,
        description=(
            "PubMed IDs supporting this target. "
            "Format: 'PMID:XXXXXXXX'."
        ),
    )
    patent_count: int = Field(
        default=0,
        description=(
            "Number of patents related to this target found in PatentsView. "
            "High patent count indicates competitive landscape."
        ),
        ge=0,
    )
    druggability_score: float = Field(
        default=0.5,
        description=(
            "Estimated druggability score (0.0-1.0). "
            "Considers binding site accessibility, published ligands, "
            "and structural data availability."
        ),
        ge=0.0,
        le=1.0,
    )
    novelty_score: float = Field(
        default=0.5,
        description=(
            "Estimated novelty score (0.0-1.0). "
            "Lower = more competitive (many existing drugs); "
            "Higher = more novel (few or no approved drugs)."
        ),
        ge=0.0,
        le=1.0,
    )

    @field_validator("gene_name", mode="before")
    @classmethod
    def uppercase_gene(cls, v: str) -> str:
        """Ensure gene names are uppercase."""
        return v.strip().upper()

    @field_validator("pdb_ids", mode="before")
    @classmethod
    def coerce_pdb_ids(cls, v: object) -> list:
        """Accept None as empty list; uppercase PDB IDs."""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(pid).upper().strip() for pid in v]
        return []

    @field_validator("literature_citations", mode="before")
    @classmethod
    def coerce_citations(cls, v: object) -> list:
        """Accept None as empty list."""
        if v is None:
            return []
        return v

    @field_validator("uniprot_id", mode="before")
    @classmethod
    def coerce_uniprot(cls, v: object) -> Optional[str]:
        """
        Accept 'null', 'none', empty string as None.

        Args:
            v: Raw UniProt ID from LLM.

        Returns:
            str | None: Cleaned accession or None.
        """
        if v is None:
            return None
        s = str(v).strip()
        if s.lower() in {"null", "none", "n/a", "unknown", ""}:
            return None
        return s.upper()

    def to_prompt_dict(self) -> dict:
        """
        Return a compact dict for inclusion in Hypothesis agent prompt.

        Excludes citation lists to conserve tokens (kept in full retrieval result).

        Returns:
            dict: Slim representation for prompt injection.
        """
        return {
            "gene_name": self.gene_name,
            "uniprot_id": self.uniprot_id,
            "pdb_ids": self.pdb_ids[:3],   # max 3 PDB IDs in prompt
            "evidence_summary": self.evidence_summary[:300],  # truncate
            "patent_count": self.patent_count,
            "druggability_score": self.druggability_score,
            "novelty_score": self.novelty_score,
        }


class RetrieverResponse(BaseModel):
    """
    Final output schema for the Retriever agent.

    Contains all ranked target candidates and retrieval statistics.
    This object is stored in pipeline state and passed (compressed) to
    the Hypothesis agent.
    """

    target_candidates: list[TargetCandidate] = Field(
        ...,
        description=(
            "Ranked list of druggable target candidates. "
            "Sorted by combined druggability + evidence score, best first. "
            "Minimum 1 candidate required."
        ),
        min_length=1,
    )
    total_papers_reviewed: int = Field(
        ...,
        description="Total number of literature abstracts processed across all batches.",
        ge=0,
    )
    total_patents_reviewed: int = Field(
        default=0,
        description="Total number of patents reviewed (0 if patent search disabled).",
        ge=0,
    )
    retrieval_timestamp: str = Field(
        ...,
        description="ISO-8601 UTC timestamp when retrieval completed.",
        min_length=10,
    )

    @field_validator("target_candidates", mode="before")
    @classmethod
    def coerce_candidates(cls, v: object) -> list:
        """Ensure candidates is a list."""
        if v is None:
            return []
        return v

    def get_top_candidates(self, n: int = 5) -> list[TargetCandidate]:
        """
        Return the top N candidates by druggability score.

        Args:
            n: Number of candidates to return.

        Returns:
            list[TargetCandidate]: Top N candidates.
        """
        sorted_candidates = sorted(
            self.target_candidates,
            key=lambda c: c.druggability_score,
            reverse=True,
        )
        return sorted_candidates[:n]

    def to_prompt_dict(self) -> dict:
        """
        Return a compressed dict for Hypothesis agent prompt.

        Includes only the top 5 candidates (slim representation each)
        and aggregate statistics.

        Returns:
            dict: Token-efficient state representation.
        """
        return {
            "target_candidates": [
                c.to_prompt_dict() for c in self.get_top_candidates(5)
            ],
            "total_papers_reviewed": self.total_papers_reviewed,
            "total_patents_reviewed": self.total_patents_reviewed,
            "retrieval_timestamp": self.retrieval_timestamp,
        }

    model_config = {
        "json_schema_extra": {
            "example": {
                "target_candidates": [
                    {
                        "gene_name": "EGFR",
                        "uniprot_id": "P00533",
                        "pdb_ids": ["1M17", "2ITY"],
                        "evidence_summary": "Overexpressed in 85% of NSCLC cases.",
                        "literature_citations": ["PMID:12345678"],
                        "patent_count": 234,
                        "druggability_score": 0.92,
                        "novelty_score": 0.45,
                    }
                ],
                "total_papers_reviewed": 150,
                "total_patents_reviewed": 45,
                "retrieval_timestamp": "2025-01-15T10:30:00+00:00",
            }
        }
    }