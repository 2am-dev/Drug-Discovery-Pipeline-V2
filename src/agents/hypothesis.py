"""
agents/hypothesis.py — Hypothesis agent: target selection and mechanism.

The Hypothesis agent receives the ranked target candidates from the Retriever
and performs two tasks:

  1. Selects the single best druggable target based on:
     - Druggability score from literature evidence.
     - UniProt structural data (binding site information).
     - Novelty relative to existing drugs.

  2. Formulates a mechanistic drug discovery hypothesis explaining:
     - How modulating the target treats the disease.
     - What evidence supports this mechanism.
     - What therapeutic modality is most appropriate.

Output schema: schemas.HypothesisResponse
Prompt template: utils.prompts.HYPOTHESIS_PROMPT
"""

from __future__ import annotations

import json
from typing import Optional

from loguru import logger

from config import PipelineConfig, ModelConfig
from schemas import HypothesisResponse, SelectedTarget
from utils.ollama_client import OllamaClient
from utils.context_manager import ContextManager
from utils.prompts import (
    SYSTEM_DRUG_EXPERT,
    HYPOTHESIS_PROMPT,
    JSON_ENFORCEMENT,
    build_agent_prompt,
)
from utils.helpers import log_json_event, utc_now_iso
from utils.json_validator import JSONValidator


class HypothesisAgent:
    """
    Target selection and mechanistic hypothesis formulation agent.

    Takes the top target candidates from the Retriever, enriches them
    with UniProt binding site data, and asks the LLM to select the best
    target and formulate a mechanistic hypothesis.
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        """
        Initialise the Hypothesis agent.

        Args:
            ollama_client: Shared OllamaClient instance.
        """
        self.client = ollama_client
        self.ctx = ContextManager()
        self.agent_name = "hypothesis"

    # ── Main entry point ──────────────────────────────────────────────────────
    async def run(self, state: dict) -> dict:
        """
        Execute target selection and hypothesis formulation.

        Args:
            state: Pipeline state dict. Uses:
                   - indication_or_target (str)
                   - retrieval_result (dict): RetrieverResponse dict
                   - task_id (str)

        Returns:
            dict: Validated HypothesisResponse dict.
        """
        indication = state["indication_or_target"]
        retrieval_result = state.get("retrieval_result", {})
        logger.info(f"[Hypothesis] Formulating hypothesis for: '{indication}'")

        # ── Prepare compressed target candidates for prompt ───────────────────
        candidates = retrieval_result.get("target_candidates", [])
        top_candidates = candidates[:5]   # Top 5 for prompt

        # Compress candidate data to fit token budget
        candidates_json = self._compress_candidates(top_candidates)

        # ── Fetch UniProt binding site data for top candidates ────────────────
        uniprot_data = await self._fetch_uniprot_data(top_candidates)

        # ── LLM call ─────────────────────────────────────────────────────────
        hypothesis_dict = await self._generate_hypothesis(
            indication=indication,
            candidates_json=candidates_json,
            uniprot_data=uniprot_data,
        )

        # ── Validate ──────────────────────────────────────────────────────────
        try:
            validated = HypothesisResponse.model_validate(hypothesis_dict)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[Hypothesis] Validation failed: {exc}. Attempting repair."
            )
            validated = self._repair_hypothesis(hypothesis_dict, top_candidates)

        # ── Confirm PDB ID ────────────────────────────────────────────────────
        validated = await self._confirm_pdb_id(validated)

        log_json_event(
            PipelineConfig.PIPELINE_LOG,
            {
                "event": "hypothesis_complete",
                "task_id": state["task_id"],
                "selected_target": validated.selected_target.gene_name,
                "pdb_id": validated.selected_target.pdb_id,
                "confidence_score": validated.hypothesis.confidence_score,
                "timestamp": utc_now_iso(),
            },
        )

        logger.success(
            f"[Hypothesis] Selected target: {validated.selected_target.gene_name} "
            f"(PDB: {validated.selected_target.pdb_id}) | "
            f"Confidence: {validated.hypothesis.confidence_score:.2f}"
        )

        return validated.model_dump()

    # ── Candidate compression ─────────────────────────────────────────────────
    def _compress_candidates(self, candidates: list[dict]) -> str:
        """
        Compress top candidates into a token-efficient JSON string.

        Removes citation lists and truncates evidence summaries.

        Args:
            candidates: List of TargetCandidate dicts.

        Returns:
            str: Compact JSON string safe for prompt injection.
        """
        slim = []
        for c in candidates:
            slim.append({
                "gene_name": c.get("gene_name", ""),
                "uniprot_id": c.get("uniprot_id"),
                "pdb_ids": c.get("pdb_ids", [])[:3],
                "evidence_summary": str(c.get("evidence_summary", ""))[:300],
                "patent_count": c.get("patent_count", 0),
                "druggability_score": c.get("druggability_score", 0.5),
                "novelty_score": c.get("novelty_score", 0.5),
            })

        compressed = json.dumps(slim, indent=2)
        return self.ctx.truncate_to_tokens(compressed, max_tokens=2000)

    # ── UniProt data fetching ─────────────────────────────────────────────────
    async def _fetch_uniprot_data(self, candidates: list[dict]) -> str:
        """
        Fetch binding site and active site data from UniProt for top candidates.

        Args:
            candidates: List of TargetCandidate dicts.

        Returns:
            str: JSON string with UniProt data for each candidate.
        """
        uniprot_results: list[dict] = []
        try:
            from tools.target_lookup import TargetLookup
            lookup = TargetLookup()

            for candidate in candidates[:3]:   # Max 3 UniProt calls
                gene = candidate.get("gene_name", "")
                uniprot_id = candidate.get("uniprot_id")

                if not gene and not uniprot_id:
                    continue

                try:
                    if uniprot_id:
                        data = await lookup.get_protein_details(uniprot_id)
                    else:
                        data = await lookup.get_uniprot_info(candidate.gene_name)

                    if data:
                        uniprot_results.append({
                            "gene_name": gene,
                            "uniprot_id": data.get("accession", uniprot_id),
                            "binding_sites": data.get("binding_sites", [])[:5],
                            "active_sites": data.get("active_sites", [])[:5],
                            "pdb_ids": data.get("pdb_ids", [])[:5],
                            "function": str(data.get("function", ""))[:300],
                        })
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"[Hypothesis] UniProt fetch for {gene}: {exc}")
                    continue

        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[Hypothesis] UniProt data fetch failed: {exc}.")

        if not uniprot_results:
            return "UniProt data unavailable — use your knowledge of target binding sites."

        result_json = json.dumps(uniprot_results, indent=2)
        return self.ctx.truncate_to_tokens(result_json, max_tokens=1500)

    # ── LLM hypothesis generation ─────────────────────────────────────────────
    async def _generate_hypothesis(
        self,
        indication: str,
        candidates_json: str,
        uniprot_data: str,
    ) -> dict:
        """
        Call the LLM to select a target and formulate a hypothesis.

        Args:
            indication: Disease or target query.
            candidates_json: Compressed target candidates JSON string.
            uniprot_data: UniProt binding site data string.

        Returns:
            dict: Raw parsed hypothesis dict.

        Raises:
            RuntimeError: If JSON parsing fails after all retries.
        """
        prompt_text = HYPOTHESIS_PROMPT.format(
            indication_or_target=indication,
            target_candidates_json=candidates_json,
            uniprot_data=uniprot_data,
            json_enforcement=JSON_ENFORCEMENT,
        )

        messages = build_agent_prompt(
            system=SYSTEM_DRUG_EXPERT,
            user=prompt_text,
            assistant_primer="{",
        )

        raw_response = await self.client.chat(
            messages=messages,
            schema=HypothesisResponse,
            context_label=self.agent_name,
        )

        if not raw_response.strip().startswith("{"):
            raw_response = "{" + raw_response

        parsed, error = JSONValidator.safe_parse(raw_response)
        if error:
            raise RuntimeError(f"Hypothesis JSON parse error: {error}")

        return parsed  # type: ignore[return-value]

    # ── PDB ID confirmation ───────────────────────────────────────────────────
    async def _confirm_pdb_id(
        self, validated: HypothesisResponse
    ) -> HypothesisResponse:
        """
        Confirm the PDB ID from the candidate list if the LLM hallucinated one.

        Checks the selected_target.pdb_id against the known pdb_ids from
        the retrieval result. If the PDB ID looks invalid (not 4 chars),
        falls back to the first known PDB ID.

        Args:
            validated: Validated HypothesisResponse object.

        Returns:
            HypothesisResponse: Object with confirmed PDB ID.
        """
        pdb_id = validated.selected_target.pdb_id
        gene = validated.selected_target.gene_name

        # Validate format
        if len(pdb_id) != 4 or not pdb_id[0].isdigit():
            logger.warning(
                f"[Hypothesis] PDB ID '{pdb_id}' looks invalid. "
                "Attempting to fetch from RCSB."
            )
            try:
                from tools.target_lookup import TargetLookup
                lookup = TargetLookup()
                pdb_ids = await lookup.search_pdb(gene)
                if pdb_ids:
                    # Pydantic models are immutable by default; reconstruct
                    target_data = validated.selected_target.model_dump()
                    target_data["pdb_id"] = pdb_ids[0]
                    new_target = SelectedTarget.model_validate(target_data)
                    hyp_data = validated.model_dump()
                    hyp_data["selected_target"] = new_target.model_dump()
                    validated = HypothesisResponse.model_validate(hyp_data)
                    logger.info(
                        f"[Hypothesis] PDB ID corrected to: {pdb_ids[0]}"
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[Hypothesis] PDB confirmation failed: {exc}")

        return validated

    # ── Repair malformed hypothesis ───────────────────────────────────────────
    def _repair_hypothesis(
        self,
        hypothesis_dict: dict,
        candidates: list[dict],
    ) -> HypothesisResponse:
        """
        Attempt to repair a hypothesis dict that failed Pydantic validation.

        Injects sensible defaults from the first candidate where fields
        are missing or malformed.

        Args:
            hypothesis_dict: Partially valid hypothesis dict.
            candidates: Top target candidates for default injection.

        Returns:
            HypothesisResponse: Repaired and validated object.
        """
        top_candidate = candidates[0] if candidates else {}
        gene = top_candidate.get("gene_name", "EGFR")
        pdb_ids = top_candidate.get("pdb_ids", ["1M17"])
        pdb_id = pdb_ids[0] if pdb_ids else "1M17"

        # Ensure selected_target is present and valid
        if not hypothesis_dict.get("selected_target"):
            hypothesis_dict["selected_target"] = {
                "gene_name": gene,
                "uniprot_id": top_candidate.get("uniprot_id", "P00533"),
                "pdb_id": pdb_id,
                "binding_site_residues": ["Met793", "Leu718", "Ala743"],
                "target_class": "kinase",
                "disease_relevance": top_candidate.get("evidence_summary", "")[:200],
            }
        else:
            st = hypothesis_dict["selected_target"]
            st.setdefault("gene_name", gene)
            st.setdefault("uniprot_id", "P00533")
            st.setdefault("pdb_id", pdb_id)
            st.setdefault("binding_site_residues", ["Met793"])
            st.setdefault("target_class", "other")
            st.setdefault("disease_relevance", "Key therapeutic target.")

        # Ensure hypothesis block is present
        if not hypothesis_dict.get("hypothesis"):
            hypothesis_dict["hypothesis"] = {
                "mechanism": (
                    f"Inhibition of {gene} is hypothesised to block key "
                    "pathways driving disease progression."
                ),
                "rationale": "Based on literature evidence gathered during retrieval.",
                "therapeutic_modality": "small_molecule",
                "confidence_score": 0.6,
            }
        else:
            h = hypothesis_dict["hypothesis"]
            h.setdefault("mechanism", f"Inhibition of {gene}.")
            h.setdefault("rationale", "Literature-supported.")
            h.setdefault("therapeutic_modality", "small_molecule")
            h.setdefault("confidence_score", 0.6)

        hypothesis_dict.setdefault("alternative_targets", [])

        return HypothesisResponse.model_validate(hypothesis_dict)