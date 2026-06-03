"""
main.py — Entry point for the End-to-End Drug Discovery Hypothesis-to-Report Pipeline.

This module orchestrates the full pipeline by:
  1. Parsing CLI arguments (disease/target, optional flags).
  2. Verifying Ollama connectivity (remote → local fallback).
  3. Invoking the Planner agent to build a task graph.
  4. Running each pipeline phase in dependency order.
  5. Writing the final Markdown/PDF report to outputs/.

Usage:
    python main.py --indication "non-small cell lung cancer" --model gemma4:31b-it-q8_0
    python main.py --target EGFR --enable-synthesis --no-docking
    python main.py --indication "Alzheimer's disease" --local-only

Environment variables (override config.py defaults):
    OLLAMA_REMOTE_URL   — e.g. http://192.168.1.100:11434/v1
    OLLAMA_LOCAL_URL    — e.g. http://localhost:11434/v1
    LLM_MODEL           — e.g. gemma2:27b
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

# ── Internal imports ──────────────────────────────────────────────────────────
from config import FeatureFlags, ModelConfig, OllamaConfig, PipelineConfig
from utils.ollama_client import OllamaClient
from utils.helpers import ensure_output_dirs, load_env, setup_logging


# ── Constants ─────────────────────────────────────────────────────────────────
PIPELINE_VERSION = "1.0.0"
PIPELINE_LOG_PATH = Path("outputs/pipeline_log.jsonl")
ERROR_LOG_PATH = Path("outputs/error_log.jsonl")


# ── Lazy agent imports (avoids circular deps at module level) ─────────────────
def _import_agents():
    """
    Import all agent modules lazily so that import errors in optional
    modules (e.g. synthesis_evaluator) do not crash the whole pipeline.
    """
    from agents.planner import PlannerAgent
    from agents.retriever import RetrieverAgent
    from agents.hypothesis import HypothesisAgent
    from agents.molecule_designer import MoleculeDesignerAgent
    from agents.docking_evaluator import DockingEvaluatorAgent
    from agents.report_compiler import ReportCompilerAgent

    agents = {
        "planner": PlannerAgent,
        "retriever": RetrieverAgent,
        "hypothesis": HypothesisAgent,
        "molecule_designer": MoleculeDesignerAgent,
        "docking_evaluator": DockingEvaluatorAgent,
        "report_compiler": ReportCompilerAgent,
    }

    if FeatureFlags.ENABLE_CHEMICAL_SYNTHESIS:
        try:
            from agents.synthesis_evaluator import SynthesisEvaluatorAgent
            agents["synthesis_evaluator"] = SynthesisEvaluatorAgent
            logger.info("Synthesis evaluator loaded (ENABLE_CHEMICAL_SYNTHESIS=True).")
        except ImportError as exc:
            logger.warning(
                f"synthesis_evaluator could not be imported: {exc}. "
                "Proceeding without synthesis evaluation."
            )
    else:
        logger.info("Synthesis evaluation DISABLED (ENABLE_CHEMICAL_SYNTHESIS=False).")

    return agents


# ── CLI argument parser ───────────────────────────────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser with all pipeline flags.
    """
    parser = argparse.ArgumentParser(
        prog="drug_discovery_pipeline",
        description=(
            "End-to-End Drug Discovery Hypothesis-to-Report Pipeline. "
            "Provide either a disease indication or a specific biological target."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --indication "non-small cell lung cancer"
  python main.py --target EGFR --model gemma2:27b --enable-synthesis
  python main.py --indication "Alzheimer's disease" --local-only --no-docking
        """,
    )

    # ── Input (mutually exclusive) ────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--indication",
        type=str,
        metavar="DISEASE",
        help='Disease indication, e.g. "non-small cell lung cancer"',
    )
    input_group.add_argument(
        "--target",
        type=str,
        metavar="GENE",
        help='Biological target gene symbol, e.g. "EGFR"',
    )

    # ── Model selection ───────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default=ModelConfig.LLM_MODEL,
        choices=["gemma4:31b-it-q8_0", "gemma4:26b-a4b-it-q8_0", "gpt-oss:20b", "llama3.2:latest"],
        help=f"Ollama LLM model to use (default: {ModelConfig.LLM_MODEL})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=ModelConfig.EMBEDDING_MODEL,
        help=f"Ollama embedding model (default: {ModelConfig.EMBEDDING_MODEL})",
    )

    # ── Connectivity ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--remote-url",
        type=str,
        default=None,
        help="Override remote Ollama URL (env: OLLAMA_REMOTE_URL)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        default=False,
        help="Skip remote Ollama; use localhost only.",
    )

    # ── Feature flags ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--enable-synthesis",
        action="store_true",
        default=FeatureFlags.ENABLE_CHEMICAL_SYNTHESIS,
        help="Enable optional chemical synthesis route evaluation.",
    )
    parser.add_argument(
        "--no-docking",
        action="store_true",
        default=False,
        help="Disable AutoDock Vina docking (use mock scores).",
    )
    parser.add_argument(
        "--no-patents",
        action="store_true",
        default=False,
        help="Disable patent search (faster runs, less coverage).",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for all pipeline outputs (default: outputs/).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )

    return parser


# ── Pipeline state helpers ────────────────────────────────────────────────────
def _log_json_event(path: Path, event: dict) -> None:
    """
    Append a JSON-serialisable dict as a single line to a JSONL file.

    Args:
        path: Target JSONL file path.
        event: Dictionary to serialise and append.
    """
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error(f"Failed to write to {path}: {exc}")


def _build_pipeline_state(
    args: argparse.Namespace,
    task_id: str,
    started_at: str,
) -> dict:
    """
    Construct the initial pipeline state dictionary passed between agents.

    Args:
        args: Parsed CLI arguments.
        task_id: Unique UUID for this pipeline run.
        started_at: ISO-8601 timestamp string.

    Returns:
        dict: Initial pipeline state.
    """
    indication_or_target = args.indication if args.indication else args.target
    input_type = "indication" if args.indication else "target"

    return {
        "task_id": task_id,
        "pipeline_version": PIPELINE_VERSION,
        "started_at": started_at,
        "input_type": input_type,
        "indication_or_target": indication_or_target,
        "llm_model": args.model,
        "embedding_model": args.embedding_model,
        # Feature flags (may be overridden by CLI)
        "enable_synthesis": args.enable_synthesis,
        "enable_docking": not args.no_docking,
        "enable_patents": not args.no_patents,
        # Populated by agents as pipeline progresses
        "plan": None,
        "retrieval_result": None,
        "hypothesis_result": None,
        "molecule_design_result": None,
        "docking_result": None,
        "synthesis_result": None,
        "report_result": None,
        # Tracking
        "errors": [],
        "phase_timings": {},
    }


# ── Phase runner ──────────────────────────────────────────────────────────────
async def run_phase(
    phase_name: str,
    agent_instance,
    state: dict,
    pipeline_log: Path,
) -> dict:
    """
    Execute a single pipeline phase, update state, and log the exchange.

    Args:
        phase_name: Human-readable name for logging (e.g. "retrieval").
        agent_instance: Instantiated agent object with an async `.run(state)` method.
        state: Current pipeline state dict (mutated in-place).
        pipeline_log: Path to the JSONL log file.

    Returns:
        dict: Updated pipeline state.

    Raises:
        RuntimeError: If the agent raises an unrecoverable error.
    """
    logger.info(f"▶  Starting phase: {phase_name.upper()}")
    phase_start = time.monotonic()

    try:
        result = await agent_instance.run(state)
        elapsed = round(time.monotonic() - phase_start, 2)
        state["phase_timings"][phase_name] = elapsed

        # Log successful exchange
        _log_json_event(
            pipeline_log,
            {
                "event": "phase_complete",
                "phase": phase_name,
                "task_id": state["task_id"],
                "elapsed_seconds": elapsed,
                "output_keys": list(result.keys()) if isinstance(result, dict) else [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        logger.success(f"✔  Phase {phase_name.upper()} completed in {elapsed}s.")
        return result

    except Exception as exc:  # noqa: BLE001
        elapsed = round(time.monotonic() - phase_start, 2)
        err_payload = {
            "event": "phase_error",
            "phase": phase_name,
            "task_id": state["task_id"],
            "error": str(exc),
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _log_json_event(ERROR_LOG_PATH, err_payload)
        logger.error(f"✘  Phase {phase_name.upper()} FAILED: {exc}")
        raise RuntimeError(f"Phase '{phase_name}' failed: {exc}") from exc


# ── Main pipeline orchestrator ────────────────────────────────────────────────
async def run_pipeline(args: argparse.Namespace) -> int:
    """
    Main async pipeline orchestrator.

    Execution order (respects dependency graph from Planner):
        retrieval → hypothesis → molecule_design → docking
        → synthesis (optional) → report

    Args:
        args: Parsed CLI arguments.

    Returns:
        int: Exit code (0 = success, 1 = failure).
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    task_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    ensure_output_dirs(output_dir)

    setup_logging(level=args.log_level, output_dir=output_dir)
    load_env()

    logger.info("=" * 60)
    logger.info(f"Drug Discovery Pipeline v{PIPELINE_VERSION}")
    logger.info(f"Task ID  : {task_id}")
    logger.info(f"Input    : {args.indication or args.target}")
    logger.info(f"Model    : {args.model}")
    logger.info(f"Synthesis: {'ON' if args.enable_synthesis else 'OFF'}")
    logger.info(f"Docking  : {'ON' if not args.no_docking else 'OFF (mock)'}")
    logger.info("=" * 60)

    # ── Apply CLI overrides to config ─────────────────────────────────────────
    if args.remote_url:
        OllamaConfig.REMOTE_URL = args.remote_url
    if args.local_only:
        OllamaConfig.SKIP_REMOTE = True  # read by OllamaClient
    FeatureFlags.ENABLE_CHEMICAL_SYNTHESIS = args.enable_synthesis
    FeatureFlags.ENABLE_DOCKING = not args.no_docking
    FeatureFlags.ENABLE_PATENT_SEARCH = not args.no_patents
    ModelConfig.LLM_MODEL = args.model
    ModelConfig.EMBEDDING_MODEL = args.embedding_model

    # ── Ollama connectivity check ──────────────────────────────────────────────
    logger.info("Checking Ollama connectivity …")
    ollama_client = OllamaClient()
    endpoint = await ollama_client.get_active_endpoint()
    logger.info(f"Active Ollama endpoint: {endpoint}")

    # ── Build initial pipeline state ──────────────────────────────────────────
    state = _build_pipeline_state(args, task_id, started_at)

    # ── Import agents ─────────────────────────────────────────────────────────
    agent_classes = _import_agents()

    # Instantiate all agents (they share the ollama_client)
    agents = {
        name: cls(ollama_client=ollama_client)
        for name, cls in agent_classes.items()
    }

    pipeline_start = time.monotonic()

    try:
        # ── Phase 0: Planning ─────────────────────────────────────────────────
        plan_result = await run_phase(
            "planning", agents["planner"], state, PIPELINE_LOG_PATH
        )
        state["plan"] = plan_result

        # ── Phase 1: Literature + Patent Retrieval ────────────────────────────
        retrieval_result = await run_phase(
            "retrieval", agents["retriever"], state, PIPELINE_LOG_PATH
        )
        state["retrieval_result"] = retrieval_result

        # ── Phase 2: Hypothesis Formation ─────────────────────────────────────
        hypothesis_result = await run_phase(
            "hypothesis", agents["hypothesis"], state, PIPELINE_LOG_PATH
        )
        state["hypothesis_result"] = hypothesis_result

        # ── Phase 3: Molecule Design ──────────────────────────────────────────
        molecule_result = await run_phase(
            "molecule_design", agents["molecule_designer"], state, PIPELINE_LOG_PATH
        )
        state["molecule_design_result"] = molecule_result

        # ── Phase 4: Docking (optional) ───────────────────────────────────────
        if FeatureFlags.ENABLE_DOCKING:
            docking_result = await run_phase(
                "docking", agents["docking_evaluator"], state, PIPELINE_LOG_PATH
            )
            state["docking_result"] = docking_result
        else:
            logger.warning("Docking DISABLED — skipping docking phase.")
            state["docking_result"] = {
                "docking_results": [],
                "skipped": True,
                "reason": "ENABLE_DOCKING=False",
            }

        # ── Phase 5: Synthesis (optional) ─────────────────────────────────────
        if FeatureFlags.ENABLE_CHEMICAL_SYNTHESIS and "synthesis_evaluator" in agents:
            synthesis_result = await run_phase(
                "synthesis",
                agents["synthesis_evaluator"],
                state,
                PIPELINE_LOG_PATH,
            )
            state["synthesis_result"] = synthesis_result
        else:
            state["synthesis_result"] = {
                "synthesis_routes": [],
                "synthesis_enabled": False,
                "message": "Chemical synthesis evaluation skipped per configuration.",
            }
            logger.info("Synthesis phase skipped.")

        # ── Phase 6: Report Compilation ───────────────────────────────────────
        report_result = await run_phase(
            "report", agents["report_compiler"], state, PIPELINE_LOG_PATH
        )
        state["report_result"] = report_result

    except RuntimeError as exc:
        logger.critical(f"Pipeline aborted: {exc}")
        _log_json_event(
            ERROR_LOG_PATH,
            {
                "event": "pipeline_aborted",
                "task_id": task_id,
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 1

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = round(time.monotonic() - pipeline_start, 2)
    report_path = state.get("report_result", {}).get("markdown_report_path", "N/A")

    summary = {
        "event": "pipeline_complete",
        "task_id": task_id,
        "total_elapsed_seconds": total_elapsed,
        "phase_timings": state["phase_timings"],
        "report_path": report_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _log_json_event(PIPELINE_LOG_PATH, summary)

    logger.info("=" * 60)
    logger.success(f"Pipeline COMPLETE in {total_elapsed}s")
    logger.info(f"Report saved to: {report_path}")
    logger.info(f"Full log: {PIPELINE_LOG_PATH}")
    logger.info("=" * 60)

    return 0


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    """Parse CLI arguments and launch the async pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    exit_code = asyncio.run(run_pipeline(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()