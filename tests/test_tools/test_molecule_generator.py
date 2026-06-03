"""
tests/test_tools/test_molecule_generator.py
Place at: drug_discovery_pipeline/tests/test_tools/test_molecule_generator.py
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_rdkit


@pytest.fixture
def calculator():
    """Import and instantiate MoleculePropertyCalculator."""
    try:
        from tools.molecule_generator import MoleculePropertyCalculator
        return MoleculePropertyCalculator()
    except ImportError:
        pytest.skip("RDKit not installed")


class TestMoleculePropertyCalculator:
    """Tests for RDKit property calculation."""

    ETHANOL_SMILES = "CCO"
    ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
    INVALID_SMILES  = "not_a_smiles_$$$$"
    ERLOTINIB_SMILES = (
        "C#Cc1cccc(Nc2ncnc3cc(OCC)c(OCC)cc23)c1"
    )

    def test_valid_smiles_returns_dict(self, calculator):
        result = calculator.calculate(self.ETHANOL_SMILES)
        assert result is not None
        assert isinstance(result, dict)

    def test_invalid_smiles_returns_none(self, calculator):
        result = calculator.calculate(self.INVALID_SMILES)
        assert result is None

    def test_ethanol_properties(self, calculator):
        result = calculator.calculate(self.ETHANOL_SMILES)
        assert result is not None
        assert result["molecular_weight"] == pytest.approx(46.07, abs=0.1)
        assert result["heavy_atom_count"] == 3
        assert result["hbd"] == 1
        assert result["hba"] == 1

    def test_lipinski_violations_zero_for_ethanol(self, calculator):
        result = calculator.calculate(self.ETHANOL_SMILES)
        assert result["lipinski_violations"] == 0

    def test_qed_score_in_range(self, calculator):
        result = calculator.calculate(self.ERLOTINIB_SMILES)
        assert result is not None
        assert 0.0 <= result["qed_score"] <= 1.0

    def test_sa_score_in_range(self, calculator):
        result = calculator.calculate(self.ERLOTINIB_SMILES)
        assert result is not None
        assert 1.0 <= result["sa_score"] <= 10.0

    def test_all_required_keys_present(self, calculator):
        required_keys = [
            "molecular_weight", "logP", "hbd", "hba",
            "tpsa", "heavy_atom_count", "qed_score",
            "sa_score", "lipinski_violations",
        ]
        result = calculator.calculate(self.ASPIRIN_SMILES)
        assert result is not None
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_canonicalize_smiles(self, calculator):
        canonical = calculator.canonicalize_smiles("c1ccccc1")
        assert canonical == "c1ccccc1"

    def test_canonicalize_invalid_returns_none(self, calculator):
        result = calculator.canonicalize_smiles(self.INVALID_SMILES)
        assert result is None

    def test_tanimoto_identical_molecules(self, calculator):
        score = calculator.tanimoto_similarity(
            self.ETHANOL_SMILES, self.ETHANOL_SMILES
        )
        assert score == pytest.approx(1.0)

    def test_tanimoto_different_molecules(self, calculator):
        score = calculator.tanimoto_similarity(
            self.ETHANOL_SMILES, self.ERLOTINIB_SMILES
        )
        assert 0.0 <= score < 0.5

    def test_filter_diverse_set(self, calculator):
        smiles_list = [
            self.ETHANOL_SMILES,
            self.ASPIRIN_SMILES,
            self.ERLOTINIB_SMILES,
        ]
        diverse = calculator.filter_diverse_set(smiles_list)
        assert len(diverse) >= 1
        assert diverse[0] == self.ETHANOL_SMILES   # first always kept