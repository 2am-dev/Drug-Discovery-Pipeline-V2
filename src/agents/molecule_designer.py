"""
agents/molecule_designer.py — Molecule Designer agent.

The Molecule Designer agent generates novel small-molecule drug candidates
*in silico* for the selected target. The pipeline is:

  1. Query ChromaDB for known inhibitors/ligands of the target.
  2. Ask the LLM to generate GENERATION_COUNT SMILES strings using
     scaffold decoration, bioisosteric replacement, and fragment merging.
  3. Validate all SMILES with RDKit.
  4. Calculate physicochemical properties: MW, LogP, QED, SA score,
     Lipinski violations, HBD, HBA, TPSA, heavy atom count.
  5. Filter to SHORTLIST_COUNT molecules that pass all drug-likeness filters.
  6. Sort shortlist by QED score (highest first).

Output schema: schemas.MoleculeDesignResponse
Prompt template: utils.prompts.MOLECULE_DESIGN_PROMPT
"""

from __future__ import annotations

import json
from typing import Optional

from loguru import logger

from config import MoleculeConfig, PipelineConfig, ModelConfig
from schemas import (
    MoleculeDesignResponse,
    GeneratedMolecule,
    FilteringCriteria,
    LLMMoleculeOutput,
)
from utils.ollama_client import OllamaClient
from utils.context_manager import ContextManager
from utils.prompts import (
    SYSTEM_CHEMIST,
    MOLECULE_DESIGN_PROMPT,
    JSON_ENFORCEMENT,
    build_agent_prompt,
)
from utils.helpers import log_json_event, utc_now_iso
from utils.json_validator import JSONValidator


class MoleculeDesignerAgent:
    """
    In silico molecule generation and drug-likeness filtering agent.

    Combines LLM creativity for SMILES generation with RDKit-calculated
    physicochemical properties for rigorous drug-likeness assessment.
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        """
        Initialise the Molecule Designer agent.

        Args:
            ollama_client: Shared OllamaClient instance.
        """
        self.client = ollama_client
        self.ctx = ContextManager()
        self.agent_name = "molecule_designer"

    # ── Main entry point ──────────────────────────────────────────────────────
    async def run(self, state: dict) -> dict:
        """
        Execute molecule generation, property calculation, and filtering.

        Args:
            state: Pipeline state dict. Uses:
                   - indication_or_target (str)
                   - hypothesis_result (dict): HypothesisResponse dict
                   - task_id (str)

        Returns:
            dict: Validated MoleculeDesignResponse dict.
        """
        indication = state["indication_or_target"]
        hypothesis = state.get("hypothesis_result", {})
        logger.info(f"[MoleculeDesigner] Designing molecules for: '{indication}'")

        # ── Extract target info from hypothesis ───────────────────────────────
        selected_target = hypothesis.get("selected_target", {})
        gene_name = selected_target.get("gene_name", "Unknown")
        uniprot_id = selected_target.get("uniprot_id", "Unknown")
        pdb_id = selected_target.get("pdb_id", "Unknown")
        binding_residues = selected_target.get("binding_site_residues", [])
        mechanism = hypothesis.get("hypothesis", {}).get("mechanism", "")

        logger.info(
            f"[MoleculeDesigner] Target: {gene_name} | PDB: {pdb_id} | "
            f"Residues: {binding_residues}"
        )

        # ── Fetch reference compounds from ChromaDB / literature ──────────────
        reference_compounds = await self._get_reference_compounds(
            gene_name, indication
        )

        # ── LLM molecule generation ───────────────────────────────────────────
        llm_molecules = await self._generate_molecules_via_llm(
            gene_name=gene_name,
            uniprot_id=uniprot_id,
            binding_residues=binding_residues,
            mechanism=mechanism,
            indication=indication,
            reference_compounds=reference_compounds,
        )

        logger.info(
            f"[MoleculeDesigner] LLM generated {len(llm_molecules)} raw molecules."
        )

        # ── RDKit property calculation ─────────────────────────────────────────
        calculated_molecules, failures = self._calculate_properties(llm_molecules)
        logger.info(
            f"[MoleculeDesigner] {len(calculated_molecules)} valid molecules "
            f"after RDKit calculation ({failures} SMILES failures)."
        )

        # ── Drug-likeness filtering ───────────────────────────────────────────
        shortlisted = self._apply_filters(calculated_molecules)
        logger.info(
            f"[MoleculeDesigner] {len(shortlisted)} molecules passed all filters."
        )

        # ── Sort shortlist by QED score ───────────────────────────────────────
        shortlisted = sorted(
            shortlisted,
            key=lambda m: m.qed_score or 0.0,
            reverse=True,
        )[: MoleculeConfig.SHORTLIST_COUNT]

        # ── Build response ────────────────────────────────────────────────────
        filtering_criteria = FilteringCriteria(
            max_molecular_weight=MoleculeConfig.MAX_MOLECULAR_WEIGHT,
            max_logp=MoleculeConfig.MAX_LOGP,
            max_hbd=MoleculeConfig.MAX_HBD,
            max_hba=MoleculeConfig.MAX_HBA,
            min_qed_score=MoleculeConfig.MIN_QED_SCORE,
            max_sa_score=MoleculeConfig.MAX_SA_SCORE,
            max_lipinski_violations=MoleculeConfig.MAX_LIPINSKI_VIOLATIONS,
        )

        response = MoleculeDesignResponse(
            generated_molecules=calculated_molecules,
            shortlisted_molecules=shortlisted,
            shortlisted_count=len(shortlisted),
            filtering_criteria=filtering_criteria,
            generation_failures=failures,
        )

        log_json_event(
            PipelineConfig.PIPELINE_LOG,
            {
                "event": "molecule_design_complete",
                "task_id": state["task_id"],
                "generated": len(calculated_molecules),
                "shortlisted": len(shortlisted),
                "failures": failures,
                "lead_smiles": shortlisted[0].smiles if shortlisted else None,
                "timestamp": utc_now_iso(),
            },
        )

        logger.success(
            f"[MoleculeDesigner] Shortlisted {len(shortlisted)} molecules. "
            f"Top QED: {shortlisted[0].qed_score:.3f}" if shortlisted else
            "[MoleculeDesigner] No molecules passed filters."
        )

        return response.model_dump()

    # ── Reference compound retrieval ──────────────────────────────────────────
    async def _get_reference_compounds(
        self, gene_name: str, indication: str
    ) -> str:
        """
        Retrieve known inhibitors/ligands for the target from tools.

        Args:
            gene_name: Target gene symbol.
            indication: Disease indication for context.

        Returns:
            str: Compact JSON string of reference compounds for prompt injection.
        """
        try:
            from tools.molecule_generator import MoleculePropertyCalculator
            calculator = MoleculePropertyCalculator()
            refs = await calculator.get_known_ligands(gene_name)
            if refs:
                refs_slim = refs[:5]   # cap at 5 references
                refs_json = json.dumps(
                    [{"smiles": r.get("smiles", ""), "name": r.get("name", "")}
                     for r in refs_slim],
                    indent=2,
                )
                return self.ctx.truncate_to_tokens(refs_json, max_tokens=500)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[MoleculeDesigner] Reference compound fetch: {exc}")

        # Fallback: provide well-known inhibitors based on target class
        fallbacks = {
            "EGFR": '[{"smiles": "CCOc1cc2ncnc(Nc3ccc(F)cc3Cl)c2cc1OCC", '
                    '"name": "Erlotinib analogue"}]',
            "KRAS": '[{"smiles": "CC1=CC(=O)N(c2ccc(C(=O)Nc3ccc(F)cc3)cn2)C1", '
                    '"name": "AMG-510 analogue"}]',
        }
        return fallbacks.get(
            gene_name.upper(),
            '[{"smiles": "c1ccccc1", "name": "benzene (placeholder)"}]',
        )

    # ── LLM molecule generation ───────────────────────────────────────────────
    async def _generate_molecules_via_llm(
        self,
        gene_name: str,
        uniprot_id: str,
        binding_residues: list[str],
        mechanism: str,
        indication: str,
        reference_compounds: str,
    ) -> list[dict]:
        """
        Ask the LLM to generate GENERATION_COUNT novel SMILES strings.

        Args:
            gene_name: Target gene symbol.
            uniprot_id: Target UniProt accession.
            binding_residues: Key binding site residues.
            mechanism: Mechanistic hypothesis text.
            indication: Disease indication.
            reference_compounds: Known ligands JSON string.

        Returns:
            list[dict]: Raw molecule dicts from LLM (SMILES + metadata).
        """
        prompt_text = MOLECULE_DESIGN_PROMPT.format(
            gene_name=gene_name,
            uniprot_id=uniprot_id,
            binding_site_residues=", ".join(binding_residues),
            mechanism=self.ctx.truncate_to_tokens(mechanism, max_tokens=300),
            indication_or_target=indication,
            reference_compounds=reference_compounds,
            generation_count=MoleculeConfig.GENERATION_COUNT,
            json_enforcement=JSON_ENFORCEMENT,
        )

        messages = build_agent_prompt(
            system=SYSTEM_CHEMIST,
            user=prompt_text,
            assistant_primer="{",
        )

        try:
            raw_response = await self.client.chat(
                messages=messages,
                schema=LLMMoleculeOutput,
                context_label=self.agent_name,
            )

            if not raw_response.strip().startswith("{"):
                raw_response = "{" + raw_response

            parsed, error = JSONValidator.safe_parse(raw_response)
            if error:
                raise RuntimeError(f"JSON parse error: {error}")

            molecules = parsed.get("generated_molecules", [])
            if not isinstance(molecules, list):
                return []
            return molecules

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[MoleculeDesigner] LLM generation failed: {exc}. "
                "Using fallback molecules."
            )
            return self._fallback_molecules(gene_name)

    # ── RDKit property calculation ─────────────────────────────────────────────
    def _calculate_properties(
        self, raw_molecules: list[dict]
    ) -> tuple[list[GeneratedMolecule], int]:
        """
        Calculate physicochemical properties for each molecule using RDKit.

        For each raw dict from the LLM:
          1. Validate SMILES with RDKit.
          2. Calculate MW, LogP, QED, SA score, Lipinski violations, etc.
          3. Construct a GeneratedMolecule with all fields populated.

        Args:
            raw_molecules: List of raw molecule dicts from LLM.

        Returns:
            tuple[list[GeneratedMolecule], int]:
                - List of GeneratedMolecule objects with properties.
                - Count of SMILES strings that failed RDKit parsing.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED, rdMolDescriptors
            rdkit_available = True
        except ImportError:
            logger.warning(
                "[MoleculeDesigner] RDKit not available. "
                "Using mock property values."
            )
            rdkit_available = False

        valid_molecules: list[GeneratedMolecule] = []
        failures = 0

        for raw in raw_molecules:
            smiles = raw.get("smiles", "").strip()
            if not smiles:
                failures += 1
                continue

            # ── RDKit validation and property calculation ──────────────────────
            if rdkit_available:
                mol_props = self._calculate_rdkit_props(smiles)
            else:
                mol_props = self._mock_props(smiles)

            if mol_props is None:
                failures += 1
                logger.debug(
                    f"[MoleculeDesigner] Invalid SMILES: {smiles[:50]}"
                )
                continue

            # ── Build GeneratedMolecule ───────────────────────────────────────
            try:
                molecule = GeneratedMolecule(
                    smiles=smiles,
                    name=raw.get("name"),
                    generation_method=raw.get("generation_method", "de_novo"),
                    design_rationale=raw.get("design_rationale", "LLM-generated."),
                    predicted_interactions=raw.get("predicted_interactions", []),
                    **mol_props,
                )
                valid_molecules.append(molecule)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    f"[MoleculeDesigner] Molecule construction failed "
                    f"for SMILES {smiles[:30]}: {exc}"
                )
                failures += 1

        return valid_molecules, failures

    @staticmethod
    def _calculate_rdkit_props(smiles: str) -> Optional[dict]:
        """
        Calculate physicochemical properties using RDKit.

        Args:
            smiles: SMILES string to process.

        Returns:
            dict | None: Property dict or None if SMILES is invalid.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED as RDKitQED, rdMolDescriptors
            from rdkit.Chem.rdMolDescriptors import CalcTPSA

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Suppress RDKit SA score warnings
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    from rdkit.Chem.SA_Score import sascorer
                    sa_score = sascorer.calculateScore(mol)
                except ImportError:
                    # SA score module may not be installed
                    sa_score = 3.0   # moderate default

            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            qed = RDKitQED.qed(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            tpsa = CalcTPSA(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()

            # Lipinski violations
            violations = sum([
                mw > 500,
                logp > 5,
                hbd > 5,
                hba > 10,
            ])

            return {
                "molecular_weight": round(mw, 2),
                "logP": round(logp, 2),
                "qed_score": round(qed, 3),
                "sa_score": round(sa_score, 2),
                "lipinski_violations": violations,
                "hbd": hbd,
                "hba": hba,
                "tpsa": round(tpsa, 2),
                "heavy_atom_count": heavy_atoms,
                "passes_filters": True,   # updated by _apply_filters
            }
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _mock_props(smiles: str) -> Optional[dict]:
        """
        Generate plausible mock properties when RDKit is unavailable.

        Uses string-length heuristics for very rough estimates.

        Args:
            smiles: SMILES string.

        Returns:
            dict | None: Mock property dict (never None for non-empty SMILES).
        """
        if not smiles:
            return None
        # Very rough estimates based on SMILES length
        length = len(smiles)
        return {
            "molecular_weight": round(min(300 + length * 2.5, 600), 1),
            "logP": round(2.0 + (length % 5) * 0.5, 2),
            "qed_score": round(0.6 + (length % 3) * 0.05, 3),
            "sa_score": round(2.5 + (length % 4) * 0.3, 2),
            "lipinski_violations": 0,
            "hbd": 2,
            "hba": 5,
            "tpsa": round(80 + (length % 10) * 3, 1),
            "heavy_atom_count": max(10, length // 2),
            "passes_filters": True,
        }

    # ── Drug-likeness filtering ───────────────────────────────────────────────
    def _apply_filters(
        self, molecules: list[GeneratedMolecule]
    ) -> list[GeneratedMolecule]:
        """
        Apply drug-likeness filters to the calculated molecule list.

        Filters based on MoleculeConfig thresholds:
          - Molecular weight ≤ MAX_MOLECULAR_WEIGHT
          - LogP ≤ MAX_LOGP
          - QED ≥ MIN_QED_SCORE
          - SA score ≤ MAX_SA_SCORE
          - Lipinski violations ≤ MAX_LIPINSKI_VIOLATIONS

        Args:
            molecules: List of molecules with calculated properties.

        Returns:
            list[GeneratedMolecule]: Molecules that pass all filters.
        """
        passing: list[GeneratedMolecule] = []

        for mol in molecules:
            # Skip if properties weren't calculated
            if mol.molecular_weight is None:
                continue

            mw_ok = mol.molecular_weight <= MoleculeConfig.MAX_MOLECULAR_WEIGHT
            logp_ok = (mol.logP or 999) <= MoleculeConfig.MAX_LOGP
            qed_ok = (mol.qed_score or 0) >= MoleculeConfig.MIN_QED_SCORE
            sa_ok = (mol.sa_score or 999) <= MoleculeConfig.MAX_SA_SCORE
            lip_ok = (
                (mol.lipinski_violations or 0)
                <= MoleculeConfig.MAX_LIPINSKI_VIOLATIONS
            )

            passes = all([mw_ok, logp_ok, qed_ok, sa_ok, lip_ok])

            # Update the passes_filters flag (model is not frozen)
            mol_data = mol.model_dump()
            mol_data["passes_filters"] = passes
            updated_mol = GeneratedMolecule.model_validate(mol_data)

            if passes:
                passing.append(updated_mol)
            else:
                logger.debug(
                    f"[MoleculeDesigner] Filtered out {mol.smiles[:30]}: "
                    f"MW={mol.molecular_weight:.0f} logP={mol.logP:.1f} "
                    f"QED={mol.qed_score:.2f} SA={mol.sa_score:.1f} "
                    f"LipViol={mol.lipinski_violations}"
                )

        return passing

    # ── Fallback molecules ────────────────────────────────────────────────────
    @staticmethod
    def _fallback_molecules(gene_name: str) -> list[dict]:
        """
        Return a small set of known drug-like molecules as a fallback.

        Used when the LLM generation call completely fails.

        Args:
            gene_name: Target gene for context (used to select relevant fallbacks).

        Returns:
            list[dict]: List of 5 raw molecule dicts with SMILES.
        """
        # Erlotinib scaffold-based fallbacks (EGFR kinase reference)
        fallbacks = [
            {
                "smiles": "CCOc1cc2ncnc(Nc3ccc(F)cc3Cl)c2cc1OCCO",
                "name": "Fallback-001",
                "generation_method": "scaffold_decoration",
                "design_rationale": "Quinazoline scaffold targeting ATP binding site.",
                "predicted_interactions": ["H-bond with Met793"],
            },
            {
                "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
                "name": "Fallback-002",
                "generation_method": "fragment_merge",
                "design_rationale": "Pyrimidine-based fragment merge.",
                "predicted_interactions": ["H-bond with Asp855"],
            },
            {
                "smiles": "O=C(Nc1ccc(Oc2ncnc3[nH]ccc23)cc1)c1ccccc1F",
                "name": "Fallback-003",
                "generation_method": "bioisostere",
                "design_rationale": "Fluorobenzamide bioisostere.",
                "predicted_interactions": ["Hydrophobic with Leu777"],
            },
            {
                "smiles": "CCc1nnc(NC(=O)Nc2ccc(OC(F)(F)F)cc2)s1",
                "name": "Fallback-004",
                "generation_method": "de_novo",
                "design_rationale": "Urea-containing thiazole.",
                "predicted_interactions": ["H-bond with backbone NH"],
            },
            {
                "smiles": "CN1CCC(Nc2ncnc3ccc(NC(=O)c4ccccc4)cc23)CC1",
                "name": "Fallback-005",
                "generation_method": "analogue_search",
                "design_rationale": "Piperidine-linked quinazoline analogue.",
                "predicted_interactions": ["Salt bridge with Asp810"],
            },
        ]
        return fallbacks