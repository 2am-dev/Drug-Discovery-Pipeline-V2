"""
agents/synthesis_evaluator.py — Synthesis Evaluator agent (OPTIONAL).

This agent is controlled by FeatureFlags.ENABLE_CHEMICAL_SYNTHESIS.
When disabled, it returns a SynthesisResponse with synthesis_enabled=False
and no routes. When enabled, it:

  1. Calculates RDKit SA scores for all shortlisted molecules.
  2. Filters to molecules with SA score ≤ MAX_SA_SCORE.
  3. Asks the LLM to propose step-by-step synthetic routes for each.
  4. Returns a SynthesisResponse with feasibility ratings.

Output schema: schemas.SynthesisResponse
Prompt template: utils.prompts.SYNTHESIS_PROMPT
"""

from __future__ import annotations

import json
from typing import Optional

from loguru import logger

from config import FeatureFlags, MoleculeConfig, PipelineConfig
from schemas import SynthesisResponse, SynthesisRoute
from utils.ollama_client import OllamaClient
from utils.context_manager import ContextManager
from utils.prompts import (
    SYSTEM_CHEMIST,
    SYNTHESIS_PROMPT,
    JSON_ENFORCEMENT,
    build_agent_prompt,
)
from utils.helpers import log_json_event, utc_now_iso
from utils.json_validator import JSONValidator


class SynthesisEvaluatorAgent:
    """
    Optional chemical synthesis route evaluation agent.

    Combines RDKit SA score calculation with LLM-proposed retrosynthetic
    routes to assess the synthetic feasibility of lead compounds.
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        """
        Initialise the Synthesis Evaluator agent.

        Args:
            ollama_client: Shared OllamaClient instance.
        """
        self.client = ollama_client
        self.ctx = ContextManager()
        self.agent_name = "synthesis_evaluator"

    # ── Main entry point ──────────────────────────────────────────────────────
    async def run(self, state: dict) -> dict:
        """
        Execute synthesis route evaluation for shortlisted molecules.

        Returns a skipped response immediately if ENABLE_CHEMICAL_SYNTHESIS
        is False (belt-and-suspenders check in addition to main.py).

        Args:
            state: Pipeline state dict. Uses:
                   - molecule_design_result (dict): MoleculeDesignResponse dict
                   - docking_result (dict): DockingResponse dict
                   - task_id (str)

        Returns:
            dict: Validated SynthesisResponse dict.
        """
        # ── Guard: check feature flag ─────────────────────────────────────────
        if not FeatureFlags.ENABLE_CHEMICAL_SYNTHESIS:
            logger.info("[Synthesis] ENABLE_CHEMICAL_SYNTHESIS=False. Skipping.")
            return SynthesisResponse(
                synthesis_routes=[],
                synthesis_enabled=False,
                message="Chemical synthesis evaluation skipped per configuration.",
            ).model_dump()

        logger.info("[Synthesis] Starting synthesis route evaluation.")

        mol_design = state.get("molecule_design_result", {})
        docking_result = state.get("docking_result", {})

        # Use shortlisted molecules preferring the docking lead
        shortlisted = mol_design.get("shortlisted_molecules", [])
        lead_smiles = docking_result.get("lead_compound_smiles")

        if not shortlisted:
            logger.warning("[Synthesis] No shortlisted molecules found.")
            return SynthesisResponse(
                synthesis_routes=[],
                synthesis_enabled=True,
                message="No shortlisted molecules available for synthesis evaluation.",
            ).model_dump()

        # ── Step 1: Calculate SA scores ───────────────────────────────────────
        molecules_with_sa = self._calculate_sa_scores(shortlisted)
        logger.info(
            f"[Synthesis] SA scores calculated for {len(molecules_with_sa)} molecules."
        )

        # ── Step 2: Filter by SA score ────────────────────────────────────────
        feasible = [
            m for m in molecules_with_sa
            if m.get("sa_score", 10.0) <= MoleculeConfig.MAX_SA_SCORE
        ]
        logger.info(
            f"[Synthesis] {len(feasible)} molecules have SA score ≤ "
            f"{MoleculeConfig.MAX_SA_SCORE}."
        )

        if not feasible:
            feasible = molecules_with_sa[:3]   # evaluate top 3 anyway
            logger.warning(
                "[Synthesis] No molecules below SA threshold. "
                "Evaluating top 3 by SA score."
            )

        # Prioritise lead compound from docking
        if lead_smiles:
            feasible = sorted(
                feasible,
                key=lambda m: (m.get("smiles") != lead_smiles, m.get("sa_score", 5.0)),
            )

        # ── Step 3: LLM route proposal ────────────────────────────────────────
        routes = await self._propose_routes(feasible[:5])   # max 5 molecules

        # ── Step 4: Build response ────────────────────────────────────────────
        # Find recommended candidate (best SA score among feasible)
        recommended = min(feasible, key=lambda m: m.get("sa_score", 10.0))
        recommended_smiles = recommended.get("smiles")

        response = SynthesisResponse(
            synthesis_routes=routes,
            synthesis_enabled=True,
            recommended_candidate=recommended_smiles,
        )

        log_json_event(
            PipelineConfig.PIPELINE_LOG,
            {
                "event": "synthesis_complete",
                "task_id": state["task_id"],
                "routes_generated": len(routes),
                "recommended_smiles": recommended_smiles,
                "timestamp": utc_now_iso(),
            },
        )

        logger.success(
            f"[Synthesis] Generated {len(routes)} synthesis routes. "
            f"Recommended SMILES: {str(recommended_smiles)[:40]}"
        )

        return response.model_dump()

    # ── SA score calculation ──────────────────────────────────────────────────
    def _calculate_sa_scores(self, molecules: list[dict]) -> list[dict]:
        """
        Calculate RDKit SA scores for a list of molecule dicts.

        If a molecule already has an sa_score from the Molecule Designer,
        that value is used. Otherwise, RDKit SA_Score is computed fresh.

        Args:
            molecules: List of GeneratedMolecule dicts.

        Returns:
            list[dict]: Molecules with sa_score populated, sorted by SA score.
        """
        result: list[dict] = []

        try:
            from rdkit import Chem
            from rdkit.Chem.SA_Score import sascorer
            rdkit_available = True
        except ImportError:
            logger.warning(
                "[Synthesis] RDKit SA_Score unavailable. "
                "Using existing sa_score values."
            )
            rdkit_available = False

        for mol in molecules:
            smiles = mol.get("smiles", "")
            existing_sa = mol.get("sa_score")

            if existing_sa is not None:
                result.append(dict(mol))
                continue

            if rdkit_available and smiles:
                try:
                    from rdkit import Chem
                    from rdkit.Chem.SA_Score import sascorer
                    rdkit_mol = Chem.MolFromSmiles(smiles)
                    if rdkit_mol:
                        sa = round(sascorer.calculateScore(rdkit_mol), 2)
                        mol_copy = dict(mol)
                        mol_copy["sa_score"] = sa
                        result.append(mol_copy)
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"[Synthesis] SA score calc failed: {exc}")

            # Fallback: use 3.0 (moderate)
            mol_copy = dict(mol)
            mol_copy["sa_score"] = mol_copy.get("sa_score", 3.0)
            result.append(mol_copy)

        return sorted(result, key=lambda m: m.get("sa_score", 10.0))

    # ── LLM route proposal ────────────────────────────────────────────────────
    async def _propose_routes(self, molecules: list[dict]) -> list[SynthesisRoute]:
        """
        Ask the LLM to propose synthetic routes for each molecule.

        Args:
            molecules: List of molecule dicts with sa_score populated.

        Returns:
            list[SynthesisRoute]: Validated synthesis route objects.
        """
        molecules_slim = [
            {
                "smiles": m.get("smiles", ""),
                "sa_score": m.get("sa_score", 3.0),
                "molecular_weight": m.get("molecular_weight"),
                "name": m.get("name", "Unknown"),
            }
            for m in molecules
        ]
        molecules_json = json.dumps(molecules_slim, indent=2)
        molecules_json = self.ctx.truncate_to_tokens(molecules_json, max_tokens=1500)

        prompt_text = SYNTHESIS_PROMPT.format(
            molecules_json=molecules_json,
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
                schema=SynthesisResponse,
                context_label=self.agent_name,
            )

            if not raw_response.strip().startswith("{"):
                raw_response = "{" + raw_response

            parsed, error = JSONValidator.safe_parse(raw_response)
            if error:
                raise RuntimeError(f"JSON parse error: {error}")

            raw_routes = parsed.get("synthesis_routes", [])
            routes: list[SynthesisRoute] = []

            for route_dict in raw_routes:
                try:
                    # Inject SA score from our calculation if missing
                    smiles = route_dict.get("smiles", "")
                    matching_mol = next(
                        (m for m in molecules if m.get("smiles") == smiles),
                        None,
                    )
                    if matching_mol and not route_dict.get("sa_score"):
                        route_dict["sa_score"] = matching_mol.get("sa_score", 3.0)

                    route = SynthesisRoute.model_validate(route_dict)
                    routes.append(route)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        f"[Synthesis] Route validation failed: {exc}"
                    )
                    continue

            return routes

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[Synthesis] LLM route proposal failed: {exc}. "
                "Returning empty routes list."
            )
            return []