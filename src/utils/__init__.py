"""
utils/__init__.py — Public re-exports for the utils package.

Importing from here gives other modules a single, stable import path:

    from utils import OllamaClient, JSONValidator, ContextManager
    from utils import JSON_ENFORCEMENT, build_agent_prompt
    from utils import setup_logging, ensure_output_dirs
"""

from utils.ollama_client import OllamaClient
from utils.json_validator import JSONValidator
from utils.context_manager import ContextManager
from utils.prompts import (
    JSON_ENFORCEMENT,
    build_agent_prompt,
    PLANNER_PROMPT,
    RETRIEVER_SUMMARISE_PROMPT,
    HYPOTHESIS_PROMPT,
    MOLECULE_DESIGN_PROMPT,
    DOCKING_ANALYSIS_PROMPT,
    SYNTHESIS_PROMPT,
    REPORT_SECTION_PROMPT,
)
from utils.helpers import (
    setup_logging,
    ensure_output_dirs,
    load_env,
    log_json_event,
    safe_json_loads,
    truncate_string,
    flatten_dict,
    utc_now_iso,
)

__all__ = [
    # Clients
    "OllamaClient",
    # Validation
    "JSONValidator",
    # Context management
    "ContextManager",
    # Prompts
    "JSON_ENFORCEMENT",
    "build_agent_prompt",
    "PLANNER_PROMPT",
    "RETRIEVER_SUMMARISE_PROMPT",
    "HYPOTHESIS_PROMPT",
    "MOLECULE_DESIGN_PROMPT",
    "DOCKING_ANALYSIS_PROMPT",
    "SYNTHESIS_PROMPT",
    "REPORT_SECTION_PROMPT",
    # Helpers
    "setup_logging",
    "ensure_output_dirs",
    "load_env",
    "log_json_event",
    "safe_json_loads",
    "truncate_string",
    "flatten_dict",
    "utc_now_iso",
]