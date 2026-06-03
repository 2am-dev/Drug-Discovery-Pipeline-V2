"""
tools/synthesis_checker.py — RDKit SA score calculator (OPTIONAL module).

This module is only imported when FeatureFlags.ENABLE_CHEMICAL_SYNTHESIS=True.
It provides SA (Synthetic Accessibility) score calculation and basic
retrosynthesis complexity metrics for use by the SynthesisEvaluatorAgent.

SA Score reference:
  Ertl, P.; Schuffenhauer, A. J. Cheminformatics 2009, 1, 8.
  DOI: 10.1186/1758-2946-1-8

SA Score interpretation:
  1.0–2.0: Very easy (readily available or 1–2 steps)
  2.0–3.0: Easy (standard chemistry, 3–5 steps)
  3.0–4.0: Moderate (5–8 steps, some challenging transformations)
  4.0–6.0: Difficult (>8 steps, specialist chemistry)
  6.0–10.0: Very difficult (not recommended for drug candidates)
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger


class SynthesisChecker:
    """
    Calculates synthetic accessibility metrics for drug candidates.

    Wraps the RDKit SA_Score contrib module and provides additional
    structural complexity metrics used by the Synthesis Evaluator agent.

    Usage:
        checker = SynthesisChecker()
        sa_score = checker.calculate_sa_score("CCO")   # 1.0–10.0
        complexity = checker.assess_complexity("CCO")  # dict of metrics
    """

    def __init__(self) -> None:
        """
        Initialise SynthesisChecker and load the RDKit SA scorer.

        Raises:
            ImportError: If RDKit is not installed.
            RuntimeError: If the SA_Score module cannot be loaded.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
            self._Chem = Chem
            self._rdMolDescriptors = rdMolDescriptors
        except ImportError as exc:
            raise ImportError(
                "RDKit is required for synthesis checking. "
                "Install: pip install rdkit"
            ) from exc

        self._sa_scorer = self._load_sa_scorer()

    @staticmethod
    def _load_sa_scorer() -> Any:
        """
        Load the RDKit SA_Score module with multiple fallback paths.

        Returns:
            module: Loaded sascorer module.

        Raises:
            RuntimeError: If the SA_Score module cannot be found.
        """
        # Path 1: RDKit contrib package import
        try:
            from rdkit.Contrib.SA_Score import sascorer
            logger.debug("[SynthesisChecker] SA_Score loaded from rdkit.Contrib.")
            return sascorer
        except ImportError:
            pass

        # Path 2: RDConfig-based path injection
        try:
            import sys
            import os
            from rdkit import RDConfig
            sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
            if os.path.isdir(sa_dir) and sa_dir not in sys.path:
                sys.path.insert(0, sa_dir)
            import sascorer  # type: ignore
            logger.debug("[SynthesisChecker] SA_Score loaded via RDConfig path.")
            return sascorer
        except (ImportError, AttributeError):
            pass

        # Path 3: Try pip-installed sa-score package
        try:
            import sascorer  # type: ignore
            logger.debug("[SynthesisChecker] SA_Score loaded from standalone package.")
            return sascorer
        except ImportError:
            pass

        raise RuntimeError(
            "SA_Score module not found. "
            "Ensure RDKit contrib modules are available. "
            "See: https://www.rdkit.org/docs/Cookbook.html"
        )

    # ── SA score calculation ──────────────────────────────────────────────────
    def calculate_sa_score(self, smiles: str) -> Optional[float]:
        """
        Calculate the Ertl-Schuffenhauer Synthetic Accessibility score.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            float | None: SA score between 1.0 and 10.0, or None if the
                          SMILES is invalid.

        Example:
            >>> checker = SynthesisChecker()
            >>> checker.calculate_sa_score("CCO")
            1.18
            >>> checker.calculate_sa_score("CC(NC(=O)c1ccc(Cl)cc1Cl)C(=O)O")
            2.34
        """
        mol = self._Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug(
                f"[SynthesisChecker] Invalid SMILES: {smiles[:50]}"
            )
            return None

        try:
            sa = self._sa_scorer.calculateScore(mol)
            return round(max(1.0, min(10.0, sa)), 3)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[SynthesisChecker] SA score calculation failed: {exc}")
            return None

    # ── Complexity assessment ─────────────────────────────────────────────────
    def assess_complexity(self, smiles: str) -> Optional[dict]:
        """
        Calculate structural complexity metrics beyond the SA score.

        Provides supplementary information for the LLM to reason about
        synthetic feasibility: stereocentres, ring complexity, reactive groups.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            dict | None: Complexity metrics dict, or None if SMILES invalid.

        Keys returned:
            sa_score:             float (1.0–10.0)
            stereocentres:        int   (number of chiral centres)
            ring_count:           int   (total rings)
            aromatic_rings:       int   (aromatic rings)
            spiro_atoms:          int   (spiro fusion atoms)
            bridgehead_atoms:     int   (bridgehead atoms)
            reactive_groups:      list  (detected reactive functional groups)
            complexity_tier:      str   ("easy"|"moderate"|"difficult"|"very_difficult")
            estimated_steps:      int   (heuristic step estimate)
        """
        mol = self._Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        try:
            # SA score
            sa_score = self.calculate_sa_score(smiles) or 5.0

            # Stereocentres
            stereo = len(
                self._rdMolDescriptors.CalcChiralCenters(mol, includeUnassigned=True)
            )

            # Ring metrics
            ring_info = mol.GetRingInfo()
            ring_count = ring_info.NumRings()
            arom_rings = self._rdMolDescriptors.CalcNumAromaticRings(mol)
            spiro = self._rdMolDescriptors.CalcNumSpiroAtoms(mol)
            bridgehead = self._rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

            # Reactive groups (heuristic SMARTS screening)
            reactive_groups = self._detect_reactive_groups(mol)

            # Complexity tier
            tier, est_steps = self._classify_complexity(
                sa_score=sa_score,
                stereo=stereo,
                ring_count=ring_count,
                spiro=spiro,
                bridgehead=bridgehead,
            )

            return {
                "sa_score": sa_score,
                "stereocentres": stereo,
                "ring_count": ring_count,
                "aromatic_rings": arom_rings,
                "spiro_atoms": spiro,
                "bridgehead_atoms": bridgehead,
                "reactive_groups": reactive_groups,
                "complexity_tier": tier,
                "estimated_steps": est_steps,
            }

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[SynthesisChecker] Complexity assessment failed: {exc}"
            )
            return None

    # ── Reactive group detection ──────────────────────────────────────────────
    def _detect_reactive_groups(self, mol: Any) -> list[str]:
        """
        Screen for reactive functional groups using SMARTS patterns.

        Flags groups that may require specialist handling or protection
        during synthesis (e.g. acyl chlorides, epoxides, isocyanates).

        Args:
            mol: RDKit Mol object.

        Returns:
            list[str]: Names of detected reactive groups (may be empty).
        """
        reactive_smarts = {
            "acyl_chloride":  "C(=O)Cl",
            "acid_anhydride": "C(=O)OC(=O)",
            "epoxide":        "C1OC1",
            "aziridine":      "C1NC1",
            "isocyanate":     "N=C=O",
            "isothiocyanate": "N=C=S",
            "diazo":          "[N]=[N+]=[CH-]",
            "peroxide":       "OO",
            "michael_acceptor": "C=CC=O",
            "aldehyde":       "[CH1](=O)",
        }

        detected: list[str] = []
        for group_name, smarts in reactive_smarts.items():
            try:
                pattern = self._Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    detected.append(group_name)
            except Exception:  # noqa: BLE001
                continue

        return detected

    # ── Complexity classification ─────────────────────────────────────────────
    @staticmethod
    def _classify_complexity(
        sa_score: float,
        stereo: int,
        ring_count: int,
        spiro: int,
        bridgehead: int,
    ) -> tuple[str, int]:
        """
        Classify synthesis complexity and estimate the number of steps.

        Uses a rule-based heuristic combining SA score and structural features.

        Args:
            sa_score: SA score (1.0–10.0).
            stereo: Number of stereocentres.
            ring_count: Total ring count.
            spiro: Number of spiro atoms.
            bridgehead: Number of bridgehead atoms.

        Returns:
            tuple[str, int]: (complexity_tier, estimated_steps)
        """
        # Base steps from SA score
        if sa_score <= 2.0:
            base_steps = 2
            tier = "easy"
        elif sa_score <= 3.0:
            base_steps = 4
            tier = "easy"
        elif sa_score <= 4.0:
            base_steps = 6
            tier = "moderate"
        elif sa_score <= 6.0:
            base_steps = 9
            tier = "difficult"
        else:
            base_steps = 14
            tier = "very_difficult"

        # Adjustments for structural complexity
        step_adjustment = 0
        if stereo > 2:
            step_adjustment += stereo - 2   # each additional stereocentre adds a step
        if spiro > 0:
            step_adjustment += spiro * 2    # spiro centres are challenging
        if bridgehead > 0:
            step_adjustment += bridgehead * 2
        if ring_count > 4:
            step_adjustment += (ring_count - 4)

        estimated_steps = base_steps + step_adjustment

        # Re-classify tier after adjustment
        if estimated_steps <= 4:
            tier = "easy"
        elif estimated_steps <= 7:
            tier = "moderate"
        elif estimated_steps <= 12:
            tier = "difficult"
        else:
            tier = "very_difficult"

        return tier, min(estimated_steps, 25)   # cap at 25 steps

    # ── Batch processing ──────────────────────────────────────────────────────
    def batch_sa_scores(self, smiles_list: list[str]) -> dict[str, Optional[float]]:
        """
        Calculate SA scores for a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            dict[str, float | None]: Mapping of SMILES → SA score.
                                     None for invalid SMILES.

        Example:
            >>> checker.batch_sa_scores(["CCO", "invalid", "c1ccccc1"])
            {'CCO': 1.18, 'invalid': None, 'c1ccccc1': 1.0}
        """
        return {smiles: self.calculate_sa_score(smiles) for smiles in smiles_list}

    def rank_by_synthesisability(
        self,
        smiles_list: list[str],
    ) -> list[tuple[str, float]]:
        """
        Rank a list of SMILES by ascending SA score (easiest first).

        Invalid SMILES (None SA score) are placed at the end.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            list[tuple[str, float]]: (smiles, sa_score) pairs, sorted
                                     easiest-to-synthesise first.
                                     Invalid SMILES get sa_score=10.0.
        """
        scored: list[tuple[str, float]] = []
        for smiles in smiles_list:
            sa = self.calculate_sa_score(smiles)
            scored.append((smiles, sa if sa is not None else 10.0))
        return sorted(scored, key=lambda x: x[1])