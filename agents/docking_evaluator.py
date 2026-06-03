"""
agents/docking_evaluator.py — Docking Evaluator agent.

Evaluates binding affinity of shortlisted molecules against the selected
target using AutoDock Vina (subprocess) or mock scoring (graceful fallback).

Pipeline:
  1. Download target PDB structure from RCSB (if not cached locally).
  2. Prepare receptor and ligand files (PDBQT format).
  3. Run AutoDock Vina via subprocess (parallel, max 4 workers).
  4. Parse Vina output logs for binding affinity scores.
  5. If Vina unavailable: generate property-based mock scores (with warning).
  6. Ask LLM to interpret docking poses and identify key interactions.
  7. Rank results and select lead compound.

Output schema: schemas.DockingResponse
Prompt template: utils.prompts.DOCKING_ANALYSIS_PROMPT
"""

from __future__ import annotations

import asyncio
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from loguru import logger

from config import DockingConfig, FeatureFlags, PipelineConfig
from schemas import DockingResponse, DockingResult, MockDockingResult
from utils.ollama_client import OllamaClient
from utils.context_manager import ContextManager
from utils.prompts import (
    SYSTEM_DRUG_EXPERT,
    DOCKING_ANALYSIS_PROMPT,
    JSON_ENFORCEMENT,
    build_agent_prompt,
)
from utils.helpers import log_json_event, utc_now_iso
from utils.json_validator import JSONValidator


class DockingEvaluatorAgent:
    """
    Molecular docking evaluation agent with AutoDock Vina integration.

    Runs docking in parallel (ThreadPoolExecutor) and falls back gracefully
    to property-based mock scores when Vina is not installed.
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        """
        Initialise the Docking Evaluator agent.

        Args:
            ollama_client: Shared OllamaClient instance.
        """
        self.client = ollama_client
        self.ctx = ContextManager()
        self.agent_name = "docking_evaluator"
        self._vina_available: Optional[bool] = None   # cached after first check

    # ── Main entry point ──────────────────────────────────────────────────────
    async def run(self, state: dict) -> dict:
        """
        Execute molecular docking for all shortlisted molecules.

        Args:
            state: Pipeline state dict. Uses:
                   - hypothesis_result (dict): HypothesisResponse dict
                   - molecule_design_result (dict): MoleculeDesignResponse dict
                   - task_id (str)

        Returns:
            dict: Validated DockingResponse dict.
        """
        hypothesis = state.get("hypothesis_result", {})
        mol_design = state.get("molecule_design_result", {})

        selected_target = hypothesis.get("selected_target", {})
        gene_name = selected_target.get("gene_name", "Unknown")
        pdb_id = selected_target.get("pdb_id", "1M17")
        binding_residues = selected_target.get("binding_site_residues", [])

        shortlisted = mol_design.get("shortlisted_molecules", [])

        logger.info(
            f"[Docking] Evaluating {len(shortlisted)} molecules against "
            f"{gene_name} (PDB: {pdb_id})"
        )

        if not shortlisted:
            logger.warning(
                "[Docking] No shortlisted molecules to dock. "
                "Using generated molecules directly."
            )
            shortlisted = mol_design.get("generated_molecules", [])[:5]

        # ── Check Vina availability ───────────────────────────────────────────
        vina_available = await self._check_vina()
        docking_software = "AutoDock Vina 1.2.3" if vina_available else (
            "MockDocking (property-based estimate)"
        )

        # ── Download / locate receptor structure ──────────────────────────────
        receptor_path: Optional[Path] = None
        if vina_available:
            receptor_path = await self._prepare_receptor(pdb_id)
            if receptor_path is None:
                logger.warning(
                    "[Docking] Receptor preparation failed. Using mock scoring."
                )
                vina_available = False

        # ── Run docking (real or mock) ─────────────────────────────────────────
        if vina_available and receptor_path:
            raw_results = await self._run_vina_parallel(
                shortlisted, receptor_path, pdb_id
            )
        else:
            raw_results = self._generate_mock_scores(shortlisted)

        # ── LLM interaction analysis ──────────────────────────────────────────
        docking_results = await self._analyse_interactions(
            raw_results=raw_results,
            gene_name=gene_name,
            pdb_id=pdb_id,
            binding_residues=binding_residues,
            docking_software=docking_software,
        )

        # ── Select lead compound ──────────────────────────────────────────────
        if not docking_results:
            raise RuntimeError(
                "[Docking] No docking results produced. "
                "Check molecule shortlist and receptor structure."
            )

        lead = docking_results[0]   # sorted by affinity in _analyse_interactions

        try:
            response = DockingResponse(
                docking_results=docking_results,
                lead_compound_smiles=lead.smiles,
                lead_compound_rationale=(
                    f"Highest predicted binding affinity "
                    f"({lead.binding_affinity_kcal_mol:.2f} kcal/mol) "
                    f"with ligand efficiency {lead.ligand_efficiency:.2f}."
                ),
                receptor_pdb=pdb_id,
                docking_software=docking_software,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[Docking] DockingResponse construction failed: {exc}")
            raise

        log_json_event(
            PipelineConfig.PIPELINE_LOG,
            {
                "event": "docking_complete",
                "task_id": state["task_id"],
                "compounds_docked": len(docking_results),
                "lead_smiles": lead.smiles[:50],
                "lead_affinity": lead.binding_affinity_kcal_mol,
                "mock": response.contains_mock_results,
                "timestamp": utc_now_iso(),
            },
        )

        logger.success(
            f"[Docking] Lead compound selected | "
            f"Affinity: {lead.binding_affinity_kcal_mol:.2f} kcal/mol | "
            f"LE: {lead.ligand_efficiency:.2f} | "
            f"Mock: {response.contains_mock_results}"
        )

        return response.model_dump()

    # ── Vina availability check ───────────────────────────────────────────────
    async def _check_vina(self) -> bool:
        """
        Check whether the AutoDock Vina binary is available on PATH.

        Caches the result to avoid repeated subprocess calls.

        Returns:
            bool: True if Vina is installed and callable.
        """
        if self._vina_available is not None:
            return self._vina_available

        try:
            from tools.docking import DockingTool
            tool = DockingTool()
            self._vina_available = tool.check_vina_available()
        except Exception:  # noqa: BLE001
            self._vina_available = False

        if self._vina_available:
            logger.info("[Docking] AutoDock Vina is available.")
        else:
            logger.warning(
                "[Docking] AutoDock Vina NOT found. "
                "Using property-based mock scores. "
                f"(Binary expected at: '{DockingConfig.VINA_BINARY}')"
            )

        return self._vina_available  # type: ignore[return-value]

    # ── Receptor preparation ──────────────────────────────────────────────────
    async def _prepare_receptor(self, pdb_id: str) -> Optional[Path]:
        """
        Download PDB structure and convert to PDBQT for Vina.

        Args:
            pdb_id: 4-character PDB identifier.

        Returns:
            Path | None: Path to the receptor PDBQT file, or None on failure.
        """
        try:
            from tools.docking import DockingTool
            tool = DockingTool()
            receptor_path = await tool.prepare_receptor(pdb_id)
            logger.info(f"[Docking] Receptor ready: {receptor_path}")
            return receptor_path
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[Docking] Receptor preparation failed: {exc}")
            return None

    # ── Parallel Vina docking ─────────────────────────────────────────────────
    async def _run_vina_parallel(
        self,
        molecules: list[dict],
        receptor_path: Path,
        pdb_id: str,
    ) -> list[dict]:
        """
        Run AutoDock Vina for each molecule in parallel using a thread pool.

        Args:
            molecules: List of GeneratedMolecule dicts.
            receptor_path: Path to prepared receptor PDBQT.
            pdb_id: PDB ID for naming output files.

        Returns:
            list[dict]: Raw docking result dicts with smiles and affinity.
        """
        from tools.docking import DockingTool
        tool = DockingTool()

        loop = asyncio.get_event_loop()
        results: list[dict] = []

        with ThreadPoolExecutor(max_workers=DockingConfig.MAX_WORKERS) as executor:
            # Submit all docking jobs
            futures = []
            for i, mol in enumerate(molecules):
                smiles = mol.get("smiles", "")
                if not smiles:
                    continue
                future = loop.run_in_executor(
                    executor,
                    tool.dock_smiles,
                    smiles,
                    str(receptor_path),
                    f"pose_{i+1:03d}",
                )
                futures.append((smiles, mol, future))

            # Collect results as they complete
            for smiles, mol, future in futures:
                try:
                    result = await asyncio.wait_for(future, timeout=300)
                    if result:
                        result["smiles"] = smiles
                        result["heavy_atom_count"] = mol.get("heavy_atom_count", 20)
                        results.append(result)
                    else:
                        logger.warning(
                            f"[Docking] Vina returned no result for "
                            f"{smiles[:30]}"
                        )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[Docking] Vina timeout for {smiles[:30]}"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"[Docking] Vina error for {smiles[:30]}: {exc}"
                    )

        logger.info(f"[Docking] Vina completed {len(results)}/{len(molecules)} jobs.")
        return results

    # ── Mock score generation ─────────────────────────────────────────────────
    def _generate_mock_scores(self, molecules: list[dict]) -> list[dict]:
        """
        Generate property-based mock docking scores when Vina is unavailable.

        Score estimation formula:
            affinity ≈ -5.0 - (QED × 3.0) - (logP × 0.2) + noise
        This is a rough heuristic — not a substitute for real docking.

        Args:
            molecules: List of GeneratedMolecule dicts.

        Returns:
            list[dict]: Mock result dicts with is_mock=True.
        """
        results: list[dict] = []
        for mol in molecules:
            smiles = mol.get("smiles", "")
            if not smiles:
                continue

            qed = mol.get("qed_score") or 0.6
            logp = mol.get("logP") or 3.0
            heavy_atoms = mol.get("heavy_atom_count") or 20

            # Property-based affinity estimate
            base_affinity = -5.0 - (qed * 3.0) - (logp * 0.2)
            # Add small Gaussian noise for realism (seed from SMILES for reproducibility)
            random.seed(hash(smiles) % (2**31))
            noise = random.gauss(0, 0.3)
            affinity = round(base_affinity + noise, 2)
            affinity = max(-12.0, min(-4.0, affinity))  # clamp to realistic range

            # Ligand efficiency
            le = round(abs(affinity) / heavy_atoms, 3) if heavy_atoms > 0 else 0.0

            results.append({
                "smiles": smiles,
                "binding_affinity_kcal_mol": affinity,
                "ligand_efficiency": le,
                "pose_file": None,
                "is_mock": True,
                "heavy_atom_count": heavy_atoms,
            })

        # Sort by affinity (most negative first)
        results.sort(key=lambda r: r["binding_affinity_kcal_mol"])
        return results

    # ── LLM interaction analysis ──────────────────────────────────────────────
    async def _analyse_interactions(
        self,
        raw_results: list[dict],
        gene_name: str,
        pdb_id: str,
        binding_residues: list[str],
        docking_software: str,
    ) -> list[DockingResult]:
        """
        Use the LLM to predict key molecular interactions for each docked pose.

        For real Vina results, the LLM interprets the binding mode.
        For mock results, the LLM predicts interactions from structure alone.

        Args:
            raw_results: Raw docking result dicts from Vina or mock.
            gene_name: Target gene name.
            pdb_id: PDB structure ID.
            binding_residues: Known binding site residues.
            docking_software: Software version string.

        Returns:
            list[DockingResult]: Fully populated DockingResult objects.
        """
        if not raw_results:
            return []

        # Prepare compact docking data for prompt
        docking_data_slim = [
            {
                "smiles": r["smiles"],
                "affinity_kcal_mol": r.get("binding_affinity_kcal_mol", -6.0),
                "is_mock": r.get("is_mock", False),
            }
            for r in raw_results[:10]   # cap at 10 for token budget
        ]
        docking_data_json = json.dumps(docking_data_slim, indent=2)
        docking_data_json = self.ctx.truncate_to_tokens(docking_data_json, max_tokens=2000)

        prompt_text = DOCKING_ANALYSIS_PROMPT.format(
            gene_name=gene_name,
            pdb_id=pdb_id,
            binding_site_residues=", ".join(binding_residues),
            docking_data_json=docking_data_json,
            docking_software=docking_software,
            json_enforcement=JSON_ENFORCEMENT,
        )

        messages = build_agent_prompt(
            system=SYSTEM_DRUG_EXPERT,
            user=prompt_text,
            assistant_primer="{",
        )

        try:
            raw_response = await self.client.chat(
                messages=messages,
                schema=DockingResponse,
                context_label=self.agent_name,
            )

            if not raw_response.strip().startswith("{"):
                raw_response = "{" + raw_response

            parsed, error = JSONValidator.safe_parse(raw_response)
            if error:
                raise RuntimeError(f"JSON parse error: {error}")

            # Build DockingResult objects from LLM response
            llm_results = parsed.get("docking_results", [])
            return self._merge_llm_and_raw(llm_results, raw_results)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[Docking] LLM interaction analysis failed: {exc}. "
                "Using raw scores without interaction annotation."
            )
            return self._build_results_from_raw(raw_results)

    # ── Result merging helpers ────────────────────────────────────────────────
    @staticmethod
    def _merge_llm_and_raw(
        llm_results: list[dict],
        raw_results: list[dict],
    ) -> list[DockingResult]:
        """
        Merge LLM interaction annotations with raw Vina/mock affinity scores.

        The LLM may modify affinity values — we trust the raw scores over LLM
        values, but use the LLM for interaction descriptions.

        Args:
            llm_results: Docking results from LLM (with interactions).
            raw_results: Raw docking data (ground truth affinity scores).

        Returns:
            list[DockingResult]: Merged results sorted by affinity.
        """
        # Build lookup: smiles → raw affinity
        raw_lookup: dict[str, dict] = {
            r["smiles"]: r for r in raw_results
        }

        merged: list[DockingResult] = []
        for i, llm_r in enumerate(llm_results):
            smiles = llm_r.get("smiles", "")
            raw = raw_lookup.get(smiles, {})

            # Trust raw affinity; use LLM for interaction info
            affinity = raw.get(
                "binding_affinity_kcal_mol",
                llm_r.get("binding_affinity_kcal_mol", -6.0),
            )
            heavy_atoms = raw.get("heavy_atom_count", 20)
            le = round(abs(affinity) / heavy_atoms, 3) if heavy_atoms > 0 else 0.0

            try:
                result = DockingResult(
                    smiles=smiles,
                    binding_affinity_kcal_mol=affinity,
                    ligand_efficiency=llm_r.get("ligand_efficiency", le),
                    pose_file=raw.get("pose_file"),
                    key_interactions=llm_r.get("key_interactions", []),
                    binding_mode_summary=llm_r.get("binding_mode_summary"),
                    rank=i + 1,
                    is_mock=raw.get("is_mock", False),
                )
                merged.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[Docking] Result merge failed for {smiles[:30]}: {exc}")

        # Sort by affinity (most negative = best)
        merged.sort(key=lambda r: r.binding_affinity_kcal_mol)
        # Re-assign ranks after sorting
        for i, r in enumerate(merged):
            object.__setattr__(r, "rank", i + 1)

        return merged

    @staticmethod
    def _build_results_from_raw(raw_results: list[dict]) -> list[DockingResult]:
        """
        Build DockingResult objects directly from raw scores (no LLM enrichment).

        Used as fallback when the LLM interaction analysis call fails.

        Args:
            raw_results: Raw docking result dicts.

        Returns:
            list[DockingResult]: Basic DockingResult objects (no interactions).
        """
        results: list[DockingResult] = []
        sorted_raw = sorted(
            raw_results,
            key=lambda r: r.get("binding_affinity_kcal_mol", -5.0),
        )

        for i, raw in enumerate(sorted_raw):
            smiles = raw.get("smiles", "")
            if not smiles:
                continue
            affinity = raw.get("binding_affinity_kcal_mol", -6.0)
            heavy_atoms = raw.get("heavy_atom_count", 20)
            le = round(abs(affinity) / heavy_atoms, 3)

            try:
                result = DockingResult(
                    smiles=smiles,
                    binding_affinity_kcal_mol=affinity,
                    ligand_efficiency=le,
                    pose_file=raw.get("pose_file"),
                    key_interactions=[],
                    binding_mode_summary=None,
                    rank=i + 1,
                    is_mock=raw.get("is_mock", False),
                )
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[Docking] Raw result build failed: {exc}")

        return results