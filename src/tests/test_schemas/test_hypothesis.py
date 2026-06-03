"""
tests/test_schemas/test_hypothesis.py
Place at: drug_discovery_pipeline/tests/test_schemas/test_hypothesis.py
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.hypothesis import (
    AlternativeTarget,
    HypothesisDetail,
    HypothesisResponse,
    SelectedTarget,
)


class TestSelectedTarget:
    """Tests for the SelectedTarget schema."""

    def test_valid_target(self):
        target = SelectedTarget(
            gene_name="egfr",   # lowercase — should be uppercased
            uniprot_id="p00533",
            pdb_id="1m17",
            binding_site_residues=["M793", "L718"],
            target_class="kinase",
            disease_relevance="EGFR drives NSCLC proliferation.",
        )
        assert target.gene_name == "EGFR"
        assert target.uniprot_id == "P00533"
        assert target.pdb_id == "1M17"

    def test_invalid_target_class_falls_back_to_other(self):
        target = SelectedTarget(
            gene_name="EGFR",
            uniprot_id="P00533",
            pdb_id="1M17",
            binding_site_residues=["M793"],
            target_class="completely_unknown_class",
            disease_relevance="Test.",
        )
        assert target.target_class == "other"

    def test_synonym_target_class_normalised(self):
        target = SelectedTarget(
            gene_name="EGFR",
            uniprot_id="P00533",
            pdb_id="1M17",
            binding_site_residues=["M793"],
            target_class="receptor tyrosine kinase",
            disease_relevance="Test.",
        )
        assert target.target_class == "kinase"

    def test_comma_separated_residues_parsed(self):
        target = SelectedTarget(
            gene_name="EGFR",
            uniprot_id="P00533",
            pdb_id="1M17",
            binding_site_residues="M793, L718, V726",  # type: ignore
            target_class="kinase",
            disease_relevance="Test.",
        )
        assert len(target.binding_site_residues) == 3
        assert "M793" in target.binding_site_residues

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            SelectedTarget(
                gene_name="EGFR",
                # uniprot_id missing
                pdb_id="1M17",
                binding_site_residues=["M793"],
                target_class="kinase",
                disease_relevance="Test.",
            )


class TestHypothesisDetail:
    """Tests for HypothesisDetail schema."""

    def test_confidence_score_clamped_from_percentage(self):
        detail = HypothesisDetail(
            mechanism="Test mechanism.",
            rationale="Test rationale.",
            therapeutic_modality="small_molecule",
            confidence_score=87,   # percentage — should become 0.87
        )
        assert detail.confidence_score == pytest.approx(0.87, abs=0.001)

    def test_confidence_score_clamped_above_1(self):
        detail = HypothesisDetail(
            mechanism="Test mechanism.",
            rationale="Test rationale.",
            therapeutic_modality="small_molecule",
            confidence_score=1.5,
        )
        assert detail.confidence_score == 1.0

    def test_modality_synonym_normalised(self):
        detail = HypothesisDetail(
            mechanism="Test mechanism.",
            rationale="Test rationale.",
            therapeutic_modality="small molecule",   # space variant
            confidence_score=0.8,
        )
        assert detail.therapeutic_modality == "small_molecule"


class TestHypothesisResponse:
    """Tests for the full HypothesisResponse schema."""

    def test_full_valid_response(self, sample_hypothesis_result):
        response = HypothesisResponse.model_validate(sample_hypothesis_result)
        assert response.selected_target.gene_name == "EGFR"
        assert response.hypothesis.confidence_score == pytest.approx(0.87)
        assert len(response.alternative_targets) == 1

    def test_none_alternatives_defaults_to_empty_list(self):
        data = {
            "selected_target": {
                "gene_name": "EGFR",
                "uniprot_id": "P00533",
                "pdb_id": "1M17",
                "binding_site_residues": ["M793"],
                "target_class": "kinase",
                "disease_relevance": "Test.",
            },
            "hypothesis": {
                "mechanism": "Test mechanism.",
                "rationale": "Test rationale.",
                "therapeutic_modality": "small_molecule",
                "confidence_score": 0.8,
            },
            "alternative_targets": None,   # None should be coerced to []
        }
        response = HypothesisResponse.model_validate(data)
        assert response.alternative_targets == []

    def test_to_prompt_dict_is_compact(self, sample_hypothesis_result):
        response = HypothesisResponse.model_validate(sample_hypothesis_result)
        prompt_dict = response.to_prompt_dict()
        assert "selected_target" in prompt_dict
        assert "mechanism" in prompt_dict
        # alternative_targets should NOT be in prompt dict (saves tokens)
        assert "alternative_targets" not in prompt_dict