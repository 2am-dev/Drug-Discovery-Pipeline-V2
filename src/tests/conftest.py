"""
tests/conftest.py — Shared pytest fixtures for the full test suite.
Place at: drug_discovery_pipeline/tests/conftest.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Minimal pipeline state for testing ───────────────────────────────────────
@pytest.fixture
def minimal_state() -> dict:
    """Minimal pipeline state dict used by most agent tests."""
    return {
        "task_id": "test-task-001",
        "pipeline_version": "1.0.0",
        "started_at": "2025-01-15T10:00:00+00:00",
        "input_type": "indication",
        "indication_or_target": "non-small cell lung cancer",
        "llm_model": "gemma4:31b-it-q8_0",
        "enable_synthesis": False,
        "enable_docking": True,
        "enable_patents": True,
        "plan": None,
        "retrieval_result": None,
        "hypothesis_result": None,
        "molecule_design_result": None,
        "docking_result": None,
        "synthesis_result": None,
        "report_result": None,
        "errors": [],
        "phase_timings": {},
    }


@pytest.fixture
def sample_retrieval_result() -> dict:
    """Sample RetrieverResponse dict for testing downstream agents."""
    return {
        "target_candidates": [
            {
                "gene_name": "EGFR",
                "uniprot_id": "P00533",
                "pdb_ids": ["1M17", "2ITY"],
                "evidence_summary": (
                    "EGFR is overexpressed in 85% of NSCLC cases and "
                    "drives proliferation via RAS-MAPK and PI3K-AKT."
                ),
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


@pytest.fixture
def sample_hypothesis_result() -> dict:
    """Sample HypothesisResponse dict for testing downstream agents."""
    return {
        "selected_target": {
            "gene_name": "EGFR",
            "uniprot_id": "P00533",
            "pdb_id": "1M17",
            "binding_site_residues": ["L718", "V726", "A743", "M793"],
            "target_class": "kinase",
            "disease_relevance": "Activating EGFR mutations drive proliferation in NSCLC.",
        },
        "hypothesis": {
            "mechanism": (
                "Selective inhibition of the EGFR ATP-binding pocket prevents "
                "autophosphorylation of Y1068 and Y1173."
            ),
            "rationale": "15 Phase III trials demonstrate OS benefit with EGFR TKIs.",
            "therapeutic_modality": "small_molecule",
            "confidence_score": 0.87,
        },
        "alternative_targets": [
            {
                "gene_name": "ALK",
                "uniprot_id": "Q9UM73",
                "rationale": "ALK fusions present in ~5% of NSCLC; less novel.",
            }
        ],
    }


@pytest.fixture
def sample_molecule_result() -> dict:
    """Sample MoleculeDesignResponse dict for testing downstream agents."""
    smiles = "CCN(CC)CCNC(=O)c1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1"
    mol = {
        "smiles": smiles,
        "name": "Compound-001",
        "generation_method": "scaffold_decoration",
        "design_rationale": "Quinazoline scaffold targeting EGFR ATP pocket.",
        "predicted_interactions": ["H-bond with Met793"],
        "molecular_weight": 445.9,
        "logP": 4.2,
        "qed_score": 0.72,
        "sa_score": 2.8,
        "lipinski_violations": 0,
        "hbd": 2,
        "hba": 6,
        "tpsa": 82.4,
        "heavy_atom_count": 32,
        "passes_filters": True,
    }
    return {
        "generated_molecules": [mol],
        "shortlisted_molecules": [mol],
        "shortlisted_count": 1,
        "filtering_criteria": {
            "max_molecular_weight": 500.0,
            "max_logp": 5.0,
            "max_hbd": 5,
            "max_hba": 10,
            "min_qed_score": 0.5,
            "max_sa_score": 4.0,
            "max_lipinski_violations": 1,
        },
        "design_strategy": "Scaffold decoration of quinazoline core.",
        "reference_scaffold": "c1cnc2ccccc2n1",
        "generation_failures": 2,
    }


@pytest.fixture
def mock_ollama_client() -> MagicMock:
    """
    Mock OllamaClient that returns valid JSON strings for any chat() call.
    Tests can override the return_value for specific scenarios.
    """
    client = MagicMock()
    client.chat = AsyncMock(return_value='{"key": "value"}')
    client.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]] * 10)
    client.get_active_endpoint = AsyncMock(return_value="http://localhost:11434/v1")
    return client


@pytest.fixture(autouse=True)
def no_file_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Redirect all file I/O (JSONL logs, report files) to tmp_path during tests.
    Prevents tests from polluting the real outputs/ directory.
    """
    monkeypatch.setattr(
        "config.PipelineConfig.PIPELINE_LOG",
        tmp_path / "pipeline_log.jsonl",
    )
    monkeypatch.setattr(
        "config.PipelineConfig.ERROR_LOG",
        tmp_path / "error_log.jsonl",
    )