"""
agents/planner.py — Planner agent: pipeline task graph builder.

The Planner is the first agent invoked in every pipeline run. Its sole
responsibility is to inspect the pipeline configuration (enabled features,
input type) and produce a structured JSON task graph that describes:

  - Which phases will run (status = "pending").
  - Which phases are skipped (status = "skipped").
  - The dependency ordering between phases.
  - An estimated total runtime in minutes.

The Planner does NOT perform any external API calls or LLM inference beyond
a single lightweight call to build the plan. If the LLM call fails, the
Planner constructs a sensible default plan from the current FeatureFlags.

Output schema: schemas.PlannerResponse
Prompt template: utils.prompts.PLANNER_PROMPT
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from config import FeatureFlags, PipelineConfig
from schemas import PlannerResponse, PipelinePhase
from utils.ollama_client import OllamaClient
from utils.prompts import (
    SYSTEM_ANALYST,
    PLANNER_PROMPT,
    JSON_ENFORCEMENT,
    build_agent_prompt,
)
from utils.helpers import log_json_event, utc_now_iso


class PlannerAgent:
    """
    Orchestrator agent that builds the pipeline execution task graph.

    The Planner produces a PlannerResponse JSON object that main.py uses
    to decide which subsequent agents to invoke and in what order.

    Typical usage (called by main.py):
        planner = PlannerAgent(ollama_client=client)
        result = await planner.run(state)
        # result is a PlannerResponse dict stored in state["plan"]
    """

    # Per-phase estimated duration in minutes (used when LLM estimation fails)
    _PHASE_DURATIONS: dict[str, int] = {
        "retrieval": 10,
        "hypothesis": 5,
        "molecule_design": 8,
        "docking": 15,
        "synthesis": 7,
        "report": 5,
    }

    def __init__(self, ollama_client: OllamaClient) -> None:
        """
        Initialise the Planner agent.

        Args:
            ollama_client: Shared OllamaClient instance (remote/local switcher).
        """
        self.client = ollama_client
        self.agent_name = "planner"

    # ── Main entry point ──────────────────────────────────────────────────────
    async def run(self, state: dict) -> dict:
        """
        Build the pipeline execution plan.

        Steps:
          1. Attempt to generate a plan via LLM (with JSON validation).
          2. On failure, fall back to a programmatically built default plan.
          3. Log the plan to pipeline_log.jsonl.
          4. Return the plan as a dict (PlannerResponse-compatible).

        Args:
            state: Current pipeline state dict containing:
                   - indication_or_target (str)
                   - input_type (str: "indication" | "target")
                   - task_id (str)
                   - enable_synthesis (bool)
                   - enable_docking (bool)
                   - enable_patents (bool)

        Returns:
            dict: Validated PlannerResponse dict.
        """
        logger.info(
            f"[Planner] Building task graph for: "
            f"'{state['indication_or_target']}'"
        )

        # ── Try LLM-generated plan ────────────────────────────────────────────
        plan_dict: Optional[dict] = None
        try:
            plan_dict = await self._generate_plan_via_llm(state)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[Planner] LLM plan generation failed: {exc}. "
                "Using programmatic fallback plan."
            )

        # ── Fall back to programmatic plan ────────────────────────────────────
        if plan_dict is None:
            plan_dict = self._build_default_plan(state)

        # ── Validate with Pydantic ────────────────────────────────────────────
        try:
            validated = PlannerResponse.model_validate(plan_dict)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[Planner] Plan validation failed ({exc}). "
                "Rebuilding with default plan."
            )
            validated = PlannerResponse.model_validate(
                self._build_default_plan(state)
            )

        # ── Log the plan ──────────────────────────────────────────────────────
        log_json_event(
            PipelineConfig.PIPELINE_LOG,
            {
                "event": "plan_created",
                "task_id": state["task_id"],
                "plan": validated.model_dump(),
                "timestamp": utc_now_iso(),
            },
        )

        logger.info(
            f"[Planner] Plan created: "
            f"{[p.phase for p in validated.phases if p.status == 'pending']} "
            f"will run | est. {validated.estimated_duration_minutes} min"
        )

        return validated.model_dump()

    # ── LLM plan generation ───────────────────────────────────────────────────
    async def _generate_plan_via_llm(self, state: dict) -> dict:
        """
        Ask the LLM to produce a plan JSON object.

        Args:
            state: Pipeline state dict.

        Returns:
            dict: Raw parsed plan dict (before Pydantic validation).

        Raises:
            RuntimeError: If the LLM call fails or JSON validation fails.
        """
        prompt_text = PLANNER_PROMPT.format(
            indication_or_target=state["indication_or_target"],
            input_type=state["input_type"],
            enable_synthesis=str(state.get("enable_synthesis", False)).lower(),
            enable_docking=str(state.get("enable_docking", True)).lower(),
            enable_patents=str(state.get("enable_patents", True)).lower(),
            task_id=state["task_id"],
            json_enforcement=JSON_ENFORCEMENT,
        )

        messages = build_agent_prompt(
            system=SYSTEM_ANALYST,
            user=prompt_text,
            assistant_primer="{",
        )

        raw_response = await self.client.chat(
            messages=messages,
            schema=PlannerResponse,
            context_label=self.agent_name,
        )

        from utils.json_validator import JSONValidator
        parsed, error = JSONValidator.safe_parse("{" + raw_response
                                                if not raw_response.strip().startswith("{")
                                                else raw_response)
        if error:
            raise RuntimeError(f"JSON parse error: {error}")

        return parsed  # type: ignore[return-value]

    # ── Programmatic fallback plan ────────────────────────────────────────────
    def _build_default_plan(self, state: dict) -> dict:
        """
        Build a safe default plan from current FeatureFlags without LLM.

        Uses the feature flag values from the pipeline state dict, which
        have already been set by main.py from CLI args / env vars.

        Args:
            state: Pipeline state dict.

        Returns:
            dict: PlannerResponse-compatible dict.
        """
        enable_synthesis = state.get("enable_synthesis", False)
        enable_docking = state.get("enable_docking", True)

        phases = [
            {
                "phase": "retrieval",
                "status": "pending",
                "dependencies": [],
            },
            {
                "phase": "hypothesis",
                "status": "pending",
                "dependencies": ["retrieval"],
            },
            {
                "phase": "molecule_design",
                "status": "pending",
                "dependencies": ["hypothesis"],
            },
            {
                "phase": "docking",
                "status": "pending" if enable_docking else "skipped",
                "dependencies": ["molecule_design"],
            },
            {
                "phase": "synthesis",
                "status": "pending" if enable_synthesis else "skipped",
                "dependencies": ["molecule_design"],
            },
            {
                "phase": "report",
                "status": "pending",
                "dependencies": (
                    ["docking"] if enable_docking else ["molecule_design"]
                ),
            },
        ]

        # Calculate estimated duration from enabled phases
        estimated = sum(
            self._PHASE_DURATIONS.get(p["phase"], 5)
            for p in phases
            if p["status"] == "pending"
        )

        return {
            "task_id": state["task_id"],
            "disease_or_target": state["indication_or_target"],
            "phases": phases,
            "estimated_duration_minutes": estimated,
            "pipeline_notes": (
                f"Programmatic plan. "
                f"Docking: {'enabled' if enable_docking else 'disabled'}. "
                f"Synthesis: {'enabled' if enable_synthesis else 'disabled'}."
            ),
        }