"""
tools/molecule_generator.py — RDKit-based molecular property calculator.

Provides MoleculePropertyCalculator, which calculates all physicochemical
properties needed for drug-likeness filtering:

  - Molecular weight (Descriptors.MolWt)
  - LogP (Wildman-Crippen: Descriptors.MolLogP)
  - H-bond donors (Descriptors.NumHDonors)
  - H-bond acceptors (Descriptors.NumHAcceptors)
  - Rotatable bonds (Descriptors.NumRotatableBonds)
  - TPSA (Descriptors.TPSA)
  - Heavy atom count (Descriptors.HeavyAtomCount)
  - QED (quantitative estimate of drug-likeness: rdkit.Chem.QED)
  - SA score (Synthetic Accessibility: custom RDKit calculation)
  - Lipinski violations (count of Ro5 rule violations)

Also provides:
  - canonicalize_smiles() — normalise SMILES representation.
  - generate_scaffold()   — extract Murcko scaffold SMILES.
  - draw_molecule()       — generate a PNG structure image (optional).
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger


class MoleculePropertyCalculator:
    """
    Calculates physicochemical properties for drug candidate molecules.

    All calculations use RDKit. If RDKit is not installed, an ImportError
    is raised at instantiation time — the caller (MoleculeDesignerAgent)
    catches this and falls back to the mock score path.

    Usage:
        calc = MoleculePropertyCalculator()
        props = calc.calculate("CCO")   # returns dict or None
        if props:
            print(props["qed_score"], props["sa_score"])
    """

    def __init__(self) -> None:
        """
        Initialise the calculator and verify RDKit is available.

        Raises:
            ImportError: If RDKit is not installed.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED, rdMolDescriptors
            self._Chem = Chem
            self._Descriptors = Descriptors
            self._QED = QED
            self._rdMolDescriptors = rdMolDescriptors
            logger.debug("[MoleculePropertyCalculator] RDKit loaded successfully.")
        except ImportError as exc:
            raise ImportError(
                "RDKit is required for molecular property calculation. "
                "Install: pip install rdkit  or  conda install -c conda-forge rdkit"
            ) from exc

        # Load SA score module (bundled with RDKit contrib)
        self._sa_scorer = self._load_sa_scorer()

    @staticmethod
    def _load_sa_scorer() -> Optional[Any]:
        """
        Attempt to load the RDKit SA_Score module.

        The SA_Score module is included in RDKit's contrib directory.
        We try multiple import paths for different RDKit versions.

        Returns:
            module | None: SA_Score module, or None if not loadable.
        """
        # Try standard RDKit contrib import paths
        try:
            from rdkit.Chem.rdMolDescriptors import CalcTPSA  # noqa: F401
            # Try the SA score from sascorer module
            try:
                from rdkit.Contrib.SA_Score import sascorer
                return sascorer
            except ImportError:
                pass

            # Try alternative import path
            try:
                import sys
                import os
                from rdkit import RDConfig
                sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
                if sa_path not in sys.path:
                    sys.path.insert(0, sa_path)
                import sascorer  # type: ignore
                return sascorer
            except (ImportError, AttributeError):
                pass

        except ImportError:
            pass

        logger.debug(
            "[MoleculePropertyCalculator] SA_Score module not found. "
            "SA scores will use a simple approximation."
        )
        return None

    # ── Main calculation method ───────────────────────────────────────────────
    def calculate(self, smiles: str) -> Optional[dict]:
        """
        Calculate all physicochemical properties for a SMILES string.

        Returns None if the SMILES is invalid (cannot be parsed by RDKit).

        Args:
            smiles: SMILES string to calculate properties for.

        Returns:
            dict | None: Property dict with all calculated values, or None
                         if the SMILES is invalid.

        Example:
            >>> calc = MoleculePropertyCalculator()
            >>> props = calc.calculate("CCN(CC)CCNC(=O)c1ccc2ncncc2c1")
            >>> props["molecular_weight"]
            256.34
            >>> props["qed_score"]
            0.71
        """
        mol = self._Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        try:
            mw = round(self._Descriptors.MolWt(mol), 2)
            logp = round(self._Descriptors.MolLogP(mol), 3)
            hbd = self._Descriptors.NumHDonors(mol)
            hba = self._Descriptors.NumHAcceptors(mol)
            rot_bonds = self._Descriptors.NumRotatableBonds(mol)
            tpsa = round(self._Descriptors.TPSA(mol), 2)
            heavy_atoms = mol.GetNumHeavyAtoms()
            rings = self._rdMolDescriptors.CalcNumRings(mol)
            arom_rings = self._rdMolDescriptors.CalcNumAromaticRings(mol)

            # QED score
            qed = round(self._QED.qed(mol), 4)

            # SA score
            sa = self._calculate_sa_score(mol)

            # Lipinski Rule-of-Five violations
            violations = self._count_lipinski_violations(mw, logp, hbd, hba)

            return {
                "molecular_weight": mw,
                "logP": logp,
                "hbd": hbd,
                "hba": hba,
                "rotatable_bonds": rot_bonds,
                "tpsa": tpsa,
                "heavy_atom_count": heavy_atoms,
                "ring_count": rings,
                "aromatic_ring_count": arom_rings,
                "qed_score": qed,
                "sa_score": sa,
                "lipinski_violations": violations,
            }

        except Exception as exc:  # noqa: BLE001
            logger.debug(
                f"[MoleculePropertyCalculator] Property calculation error "
                f"for '{smiles[:50]}': {exc}"
            )
            return None

    # ── SA score ──────────────────────────────────────────────────────────────
    def _calculate_sa_score(self, mol: Any) -> float:
        """
        Calculate Synthetic Accessibility score (1.0-10.0) for a molecule.

        Uses the RDKit SA_Score contrib module when available, otherwise
        falls back to a simple approximation based on ring complexity
        and heavy atom count.

        Args:
            mol: RDKit Mol object.

        Returns:
            float: SA score between 1.0 and 10.0 (lower = easier).
        """
        if self._sa_scorer is not None:
            try:
                sa = self._sa_scorer.calculateScore(mol)
                return round(max(1.0, min(10.0, sa)), 3)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[MoleculePropertyCalculator] SA_Score error: {exc}")

        # Fallback approximation:
        # More rings + more heavy atoms + stereocentres → harder to synthesise
        try:
            from rdkit.Chem import rdMolDescriptors
            n_rings = rdMolDescriptors.CalcNumRings(mol)
            n_stereo = len(rdMolDescriptors.CalcCIPCodeForAtom(mol, 0)) \
                if mol.GetNumAtoms() > 0 else 0
            n_heavy = mol.GetNumHeavyAtoms()
            n_stereo = len(
                [
                    a for a in mol.GetAtoms()
                    if a.GetChiralTag() != self._Chem.ChiralType.CHI_UNSPECIFIED
                ]
            )
            # Simple heuristic SA score approximation
            approx_sa = 1.0 + (n_rings * 0.5) + (n_heavy * 0.05) + (n_stereo * 0.8)
            return round(max(1.0, min(10.0, approx_sa)), 3)
        except Exception:  # noqa: BLE001
            return 5.0   # neutral default

    # ── Lipinski violations ───────────────────────────────────────────────────
    @staticmethod
    def _count_lipinski_violations(
        mw: float,
        logp: float,
        hbd: int,
        hba: int,
    ) -> int:
        """
        Count Lipinski Rule-of-Five violations.

        Rules:
          - MW ≤ 500 Da
          - LogP ≤ 5
          - H-bond donors ≤ 5
          - H-bond acceptors ≤ 10

        Args:
            mw: Molecular weight in Daltons.
            logp: Calculated LogP.
            hbd: Number of H-bond donors.
            hba: Number of H-bond acceptors.

        Returns:
            int: Number of violations (0-4).
        """
        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1
        return violations

    # ── SMILES normalisation ──────────────────────────────────────────────────
    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """
        Return the canonical RDKit SMILES for a given SMILES string.

        Args:
            smiles: Input SMILES string.

        Returns:
            str | None: Canonical SMILES, or None if input is invalid.

        Example:
            >>> calc.canonicalize_smiles("c1ccccc1")
            'c1ccccc1'
        """
        mol = self._Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self._Chem.MolToSmiles(mol)

    # ── Scaffold extraction ───────────────────────────────────────────────────
    def generate_scaffold(self, smiles: str) -> Optional[str]:
        """
        Extract the Murcko scaffold SMILES for a molecule.

        The Murcko scaffold is the ring system with all side chains removed.
        Useful for grouping molecules by core structure.

        Args:
            smiles: Input SMILES string.

        Returns:
            str | None: Murcko scaffold SMILES, or None if molecule is invalid
                        or has no ring system.

        Example:
            >>> calc.generate_scaffold("CCc1ccc(NC(=O)c2ccccc2)cc1")
            'c1ccc(-c2ccccc2)cc1'
        """
        mol = self._Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
                return None
            return self._Chem.MolToSmiles(scaffold_mol)
        except Exception:  # noqa: BLE001
            return None

    # ── Structural fingerprints ───────────────────────────────────────────────
    def tanimoto_similarity(self, smiles1: str, smiles2: str) -> float:
        """
        Calculate Tanimoto similarity between two molecules using
        Morgan fingerprints (radius=2, 2048 bits).

        Args:
            smiles1: First SMILES string.
            smiles2: Second SMILES string.

        Returns:
            float: Tanimoto coefficient (0.0-1.0). Returns 0.0 if either
                   SMILES is invalid.
        """
        try:
            from rdkit.Chem import AllChem, DataStructs

            mol1 = self._Chem.MolFromSmiles(smiles1)
            mol2 = self._Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                return 0.0

            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
            return round(DataStructs.TanimotoSimilarity(fp1, fp2), 4)

        except Exception:  # noqa: BLE001
            return 0.0

    def filter_diverse_set(
        self,
        smiles_list: list[str],
        min_tanimoto: float = 0.35,
    ) -> list[str]:
        """
        Filter a list of SMILES to a diverse subset using Tanimoto distance.

        Greedily selects molecules such that no two in the selected set
        have Tanimoto similarity > (1 - min_tanimoto). Used to ensure
        chemical diversity in the shortlisted molecules.

        Args:
            smiles_list: List of SMILES strings to filter.
            min_tanimoto: Maximum allowed similarity between any two
                          selected molecules (default: 0.35).

        Returns:
            list[str]: Diverse subset of SMILES strings.
        """
        if not smiles_list:
            return []

        selected: list[str] = [smiles_list[0]]

        for candidate in smiles_list[1:]:
            is_diverse = all(
                self.tanimoto_similarity(candidate, sel) < (1.0 - min_tanimoto)
                for sel in selected
            )
            if is_diverse:
                selected.append(candidate)

        return selected

    # ── Molecule image generation ─────────────────────────────────────────────
    def draw_molecule(
        self,
        smiles: str,
        output_path: str,
        width: int = 400,
        height: int = 300,
    ) -> bool:
        """
        Generate a 2D structure image (PNG) for a molecule.

        Args:
            smiles: SMILES string to draw.
            output_path: Full path for the output PNG file.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            bool: True if the image was successfully saved.
        """
        try:
            from rdkit.Chem import Draw

            mol = self._Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            from rdkit.Chem import rdDepictor
            rdDepictor.Compute2DCoords(mol)

            img = Draw.MolToImage(mol, size=(width, height))
            img.save(output_path)
            logger.debug(f"[MoleculePropertyCalculator] Image saved: {output_path}")
            return True

        except Exception as exc:  # noqa: BLE001
            logger.debug(
                f"[MoleculePropertyCalculator] Image generation failed: {exc}"
            )
            return False