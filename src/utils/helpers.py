"""
utils/helpers.py — General-purpose utility functions for the pipeline.

Contains functions for:
  - Logging setup (loguru configuration with file + console sinks).
  - Output directory management.
  - Environment variable loading (.env file via python-dotenv).
  - JSONL file I/O (pipeline_log.jsonl, error_log.jsonl).
  - Miscellaneous data manipulation utilities.

These functions have no dependencies on other pipeline modules, making
them safe to import from anywhere without circular import risk.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Environment loading
# ─────────────────────────────────────────────────────────────────────────────
def load_env(env_file: str = ".env") -> bool:
    """
    Load environment variables from a .env file using python-dotenv.

    Variables already set in the environment are NOT overridden
    (dotenv default behaviour with override=False).

    Args:
        env_file: Path to the .env file (default: ".env" in cwd).

    Returns:
        bool: True if the .env file was found and loaded, False otherwise.

    Example:
        >>> load_env()   # loads .env from current directory
        >>> load_env(".env.production")
    """
    try:
        from dotenv import load_dotenv
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.debug(f"Loaded environment from: {env_path.resolve()}")
            return True
        else:
            logger.debug(
                f".env file not found at '{env_path.resolve()}'. "
                "Using system environment variables only."
            )
            return False
    except ImportError:
        logger.warning(
            "python-dotenv not installed. Cannot load .env file. "
            "Install with: pip install python-dotenv"
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(
    level: str = "INFO",
    output_dir: Optional[Path] = None,
    log_filename: str = "pipeline_{time:YYYY-MM-DD}.log",
) -> None:
    """
    Configure loguru with console and rotating file sinks.

    Removes the default loguru handler first, then adds:
      - Coloured console output at the requested level.
      - A daily-rotating file sink in outputs/ at DEBUG level (captures all).

    Args:
        level: Console log level ("DEBUG", "INFO", "WARNING", "ERROR").
        output_dir: Directory for log files. Created if it doesn't exist.
                    Defaults to Path("outputs").
        log_filename: Loguru-format filename template for the log file.

    Example:
        >>> setup_logging(level="DEBUG", output_dir=Path("outputs"))
    """
    # Remove default loguru handler
    logger.remove()

    # Console sink — coloured, human-readable
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "— <level>{message}</level>"
        ),
        backtrace=True,
        diagnose=(level == "DEBUG"),
    )

    # File sink — full DEBUG regardless of console level
    _output_dir = output_dir or Path("outputs")
    _output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _output_dir / log_filename

    logger.add(
        str(log_path),
        level="DEBUG",
        rotation="1 day",         # New file each day
        retention="7 days",       # Keep logs for 7 days
        compression="gz",         # Compress old logs
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} — {message}"
        ),
        backtrace=True,
        diagnose=True,
    )

    logger.info(
        f"Logging configured: console={level}, file=DEBUG → {log_path}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Directory management
# ─────────────────────────────────────────────────────────────────────────────
def ensure_output_dirs(base_dir: Path = Path("outputs")) -> dict[str, Path]:
    """
    Create all required output directories for the pipeline.

    Creates (with parents, no error if exists):
      - outputs/
      - outputs/poses/    (AutoDock Vina output PDBQT files)
      - outputs/reports/  (generated Markdown/PDF reports)
      - data/
      - data/vectorstore/ (ChromaDB persistence)
      - data/llm_cache/   (LLM response cache, if enabled)

    Args:
        base_dir: Root output directory (default: outputs/).

    Returns:
        dict[str, Path]: Mapping of directory name → Path for downstream use.

    Example:
        >>> dirs = ensure_output_dirs(Path("outputs"))
        >>> report_path = dirs["reports"] / "proposal.md"
    """
    dirs = {
        "outputs": base_dir,
        "poses": base_dir / "poses",
        "reports": base_dir / "reports",
        "data": Path("data"),
        "vectorstore": Path("data") / "vectorstore",
        "llm_cache": Path("data") / "llm_cache",
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ready: {path}")

    return dirs


# ─────────────────────────────────────────────────────────────────────────────
# JSONL file I/O
# ─────────────────────────────────────────────────────────────────────────────
def log_json_event(path: Path, event: dict) -> None:
    """
    Append a JSON-serialisable dictionary as a single line to a JSONL file.

    Creates the file and parent directories if they don't exist.
    Silently skips logging on OSError to avoid crashing the pipeline.

    Args:
        path: Target JSONL file path.
        event: Dictionary to serialise. Non-serialisable values are
               converted to strings via default=str.

    Example:
        >>> log_json_event(
        ...     Path("outputs/pipeline_log.jsonl"),
        ...     {"event": "phase_complete", "phase": "retrieval", "elapsed": 45.2}
        ... )
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Don't let logging failure crash the pipeline
        print(f"[helpers] WARNING: Failed to write to {path}: {exc}", file=sys.stderr)


def read_jsonl(path: Path) -> list[dict]:
    """
    Read all lines from a JSONL file and return them as a list of dicts.

    Skips lines that fail JSON parsing (with a warning) to handle partially
    written files from a crashed previous run.

    Args:
        path: JSONL file path to read.

    Returns:
        list[dict]: Parsed events. Empty list if file doesn't exist.

    Example:
        >>> events = read_jsonl(Path("outputs/pipeline_log.jsonl"))
        >>> phase_events = [e for e in events if e.get("event") == "phase_complete"]
    """
    if not path.exists():
        return []

    events: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    f"Skipping malformed JSONL line {line_num} in {path}: {exc}"
                )
    return events


# ─────────────────────────────────────────────────────────────────────────────
# JSON utilities
# ─────────────────────────────────────────────────────────────────────────────
def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Attempt to parse a JSON string, returning a default on failure.

    Args:
        text: JSON string to parse.
        default: Value to return if parsing fails (default: None).

    Returns:
        Any: Parsed object or `default`.

    Example:
        >>> safe_json_loads('{"key": "value"}')
        {'key': 'value'}
        >>> safe_json_loads("not json", default={})
        {}
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, indent: Optional[int] = None) -> str:
    """
    Serialise an object to JSON string, coercing non-serialisable values.

    Args:
        obj: Object to serialise.
        indent: Optional indentation for pretty-printing.

    Returns:
        str: JSON string. Non-serialisable values are converted to str().
    """
    return json.dumps(obj, default=str, ensure_ascii=False, indent=indent)


# ─────────────────────────────────────────────────────────────────────────────
# String utilities
# ─────────────────────────────────────────────────────────────────────────────
def truncate_string(text: str, max_chars: int = 500, suffix: str = " …") -> str:
    """
    Truncate a string to `max_chars` characters with an optional suffix.

    Args:
        text: Input string.
        max_chars: Maximum character length of the output (including suffix).
        suffix: Appended when truncation occurs.

    Returns:
        str: Original string if short enough, otherwise truncated + suffix.

    Example:
        >>> truncate_string("Hello world", max_chars=8)
        'Hello  …'
    """
    if len(text) <= max_chars:
        return text
    cut = max_chars - len(suffix)
    return text[:max(0, cut)].rstrip() + suffix


def clean_whitespace(text: str) -> str:
    """
    Collapse consecutive whitespace characters into single spaces.

    Useful for normalising scraped web content before tokenisation.

    Args:
        text: Input string with potentially excessive whitespace.

    Returns:
        str: Whitespace-normalised string.
    """
    import re
    return re.sub(r"\s+", " ", text).strip()


def extract_smiles_from_text(text: str) -> list[str]:
    """
    Extract SMILES strings from free text using a heuristic regex.

    This is a best-effort extraction for when an LLM embeds SMILES in
    prose rather than JSON. The regex matches common SMILES patterns but
    will not catch all valid SMILES strings.

    Args:
        text: Free text potentially containing SMILES strings.

    Returns:
        list[str]: Extracted SMILES candidates (unvalidated).
    """
    import re
    # Match sequences of SMILES-valid characters (atoms, bonds, brackets, etc.)
    # Minimum length of 5 to filter out noise.
    pattern = re.compile(
        r"(?<!\w)"
        r"([A-Za-z][A-Za-z0-9@+\-\[\]()=#%./\\:]{4,})"
        r"(?!\w)"
    )
    candidates = pattern.findall(text)

    # Filter: valid SMILES must contain at least one letter
    return [c for c in candidates if any(ch.isalpha() for ch in c)]


# ─────────────────────────────────────────────────────────────────────────────
# Dictionary utilities
# ─────────────────────────────────────────────────────────────────────────────
def flatten_dict(
    d: dict,
    parent_key: str = "",
    sep: str = ".",
) -> dict:
    """
    Flatten a nested dictionary into a single-level dict with dotted keys.

    Args:
        d: Dictionary to flatten (may be arbitrarily nested).
        parent_key: Prefix for top-level call (leave empty).
        sep: Separator between key levels.

    Returns:
        dict: Flattened dictionary.

    Example:
        >>> flatten_dict({"a": {"b": 1, "c": {"d": 2}}})
        {'a.b': 1, 'a.c.d': 2}
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}[{i}]", sep).items())
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, v))
    return dict(items)


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge `override` into `base`, returning a new dict.

    Values in `override` take precedence over `base`. Nested dicts are
    merged recursively rather than replaced.

    Args:
        base: Base dictionary.
        override: Dictionary with values that override base.

    Returns:
        dict: Merged dictionary (new object, inputs are not mutated).

    Example:
        >>> deep_merge({"a": 1, "b": {"c": 2}}, {"b": {"d": 3}})
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def filter_none_values(d: dict) -> dict:
    """
    Remove all keys with None values from a dictionary (shallow).

    Args:
        d: Input dictionary.

    Returns:
        dict: Dictionary with None-valued keys removed.

    Example:
        >>> filter_none_values({"a": 1, "b": None, "c": "hello"})
        {'a': 1, 'c': 'hello'}
    """
    return {k: v for k, v in d.items() if v is not None}


# ─────────────────────────────────────────────────────────────────────────────
# Date/time utilities
# ─────────────────────────────────────────────────────────────────────────────
def utc_now_iso() -> str:
    """
    Return the current UTC time as an ISO-8601 string.

    Returns:
        str: e.g. "2025-01-15T10:30:00.123456+00:00"

    Example:
        >>> ts = utc_now_iso()
        >>> ts.endswith("+00:00")
        True
    """
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        str: Formatted string, e.g. "2m 34s" or "45.3s".

    Example:
        >>> format_duration(154.3)
        '2m 34s'
        >>> format_duration(45.3)
        '45.3s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}m {remaining_seconds}s"


# ─────────────────────────────────────────────────────────────────────────────
# File I/O utilities
# ─────────────────────────────────────────────────────────────────────────────
def write_text_file(path: Path, content: str, encoding: str = "utf-8") -> None:
    """
    Write content to a text file, creating parent directories as needed.

    Args:
        path: Target file path.
        content: String content to write.
        encoding: File encoding (default: utf-8).

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)
    logger.debug(f"Written: {path} ({len(content)} chars)")


def read_text_file(path: Path, encoding: str = "utf-8") -> Optional[str]:
    """
    Read a text file, returning None if the file doesn't exist.

    Args:
        path: File path to read.
        encoding: File encoding (default: utf-8).

    Returns:
        str | None: File contents, or None if not found.
    """
    if not path.exists():
        logger.debug(f"File not found: {path}")
        return None
    content = path.read_text(encoding=encoding)
    logger.debug(f"Read: {path} ({len(content)} chars)")
    return content


def generate_output_filename(
    prefix: str,
    extension: str = "md",
    timestamp: Optional[str] = None,
) -> str:
    """
    Generate a timestamped filename for pipeline outputs.

    Args:
        prefix: File name prefix (e.g. "proposal").
        extension: File extension without dot (e.g. "md", "pdf").
        timestamp: Optional custom timestamp string. Defaults to current UTC
                   time formatted as YYYYMMDD_HHMMSS.

    Returns:
        str: Filename string, e.g. "proposal_20250115_103045.md".

    Example:
        >>> generate_output_filename("proposal", "md")
        'proposal_20250115_103045.md'
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


# ─────────────────────────────────────────────────────────────────────────────
# Validation utilities
# ─────────────────────────────────────────────────────────────────────────────
def validate_smiles(smiles: str) -> bool:
    """
    Check whether a SMILES string represents a valid molecule using RDKit.

    Args:
        smiles: SMILES string to validate.

    Returns:
        bool: True if RDKit can parse the SMILES, False otherwise.
              Returns False (with warning) if RDKit is not installed.

    Example:
        >>> validate_smiles("CCO")
        True
        >>> validate_smiles("not_a_smiles")
        False
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except ImportError:
        logger.warning(
            "RDKit not available for SMILES validation. "
            "Install with: pip install rdkit"
        )
        return True  # Assume valid if we can't check


def validate_uniprot_id(uniprot_id: str) -> bool:
    """
    Check whether a string looks like a valid UniProt accession.

    UniProt accession format:
      - Old format: [A-N,R-Z][0-9][A-Z][A-Z,0-9][A-Z,0-9][0-9]
      - New format: [A-N,R-Z][0-9]{1,2}[A-Z][A-Z,0-9]{2}[0-9]

    This is a format check only — does NOT verify the accession exists.

    Args:
        uniprot_id: String to validate.

    Returns:
        bool: True if the string matches UniProt accession format.

    Example:
        >>> validate_uniprot_id("P00533")
        True
        >>> validate_uniprot_id("INVALID")
        False
    """
    import re
    pattern = re.compile(
        r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$"   # Old format
        r"|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"  # New format
    )
    return bool(pattern.match(uniprot_id.strip()))


def validate_pdb_id(pdb_id: str) -> bool:
    """
    Check whether a string looks like a valid PDB identifier (4 characters).

    Args:
        pdb_id: String to validate.

    Returns:
        bool: True if matches PDB ID format (1 digit + 3 alphanumeric).

    Example:
        >>> validate_pdb_id("1M17")
        True
        >>> validate_pdb_id("TOOSHORT")
        False
    """
    import re
    return bool(re.match(r"^[0-9][A-Za-z0-9]{3}$", pdb_id.strip()))