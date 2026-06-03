"""
config.py — Centralised configuration for the Drug Discovery Pipeline.

All tuneable parameters live here. Environment variables take precedence
over the class-level defaults, enabling Docker/Kubernetes deployments
without code changes.

Load order:
    1. Class defaults (below)
    2. .env file (loaded by utils/helpers.py via python-dotenv)
    3. Environment variables (os.environ)
    4. CLI flags (applied in main.py after argparse)

Example .env file:
    OLLAMA_REMOTE_URL=http://192.168.1.50:11434/v1
    LLM_MODEL=gemma2:27b
    ENABLE_CHEMICAL_SYNTHESIS=true
    CHROMA_PERSIST_DIR=data/vectorstore
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ── Helper: read env with type coercion ───────────────────────────────────────
def _env_str(key: str, default: str) -> str:
    """Return environment variable as string, falling back to default."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """Return environment variable coerced to int, falling back to default."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Return environment variable coerced to float, falling back to default."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    """
    Return environment variable coerced to bool.

    Truthy strings: "1", "true", "yes", "on"  (case-insensitive)
    All others → False.
    """
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# ─────────────────────────────────────────────────────────────────────────────
# Ollama connectivity
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OllamaConfig:
    """
    Configuration for Ollama server connectivity.

    The pipeline tries REMOTE_URL first. If health check fails within
    HEALTH_CHECK_TIMEOUT seconds, it falls back to LOCAL_URL.
    Set SKIP_REMOTE=True (via --local-only CLI flag) to bypass remote entirely.
    """

    # Primary remote endpoint (edit or set OLLAMA_REMOTE_URL env var)
    REMOTE_URL: str = field(
        default_factory=lambda: _env_str(
            "OLLAMA_REMOTE_URL", "http://10.10.27.37:11434/v1"
        )
    )

    # Local fallback endpoint
    LOCAL_URL: str = field(
        default_factory=lambda: _env_str(
            "OLLAMA_LOCAL_URL", "http://localhost:11434/v1"
        )
    )

    # Seconds to wait for health-check response before switching to fallback
    HEALTH_CHECK_TIMEOUT: int = field(
        default_factory=lambda: _env_int("OLLAMA_HEALTH_TIMEOUT", 300)
    )

    # Maximum number of retries per LLM call before giving up
    MAX_RETRIES: int = field(
        default_factory=lambda: _env_int("OLLAMA_MAX_RETRIES", 3)
    )

    # Base delay (seconds) for exponential backoff; actual delay = BASE * 2^attempt
    RETRY_BASE_DELAY: float = field(
        default_factory=lambda: _env_float("OLLAMA_RETRY_BASE_DELAY", 1.0)
    )

    # Maximum backoff delay cap (seconds)
    RETRY_MAX_DELAY: float = field(
        default_factory=lambda: _env_float("OLLAMA_RETRY_MAX_DELAY", 30.0)
    )

    # Internal flag: set to True by --local-only CLI flag
    SKIP_REMOTE: bool = field(
        default_factory=lambda: _env_bool("OLLAMA_SKIP_REMOTE", False)
    )

    # API key placeholder (Ollama doesn't require one, but openai client does)
    API_KEY: str = field(
        default_factory=lambda: _env_str("OLLAMA_API_KEY", "ollama")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model selection and inference parameters
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """
    Model names and inference hyper-parameters.

    Context budget strategy
    ───────────────────────
    Gemma4:31b-it-q8_0 and gemma4:26b-a4b-it-q8_0 both support 32K context windows via Ollama.
    We allocate MAX_CONTEXT_TOKENS (8 000) for the *prompt* and leave the
    remaining ~24K for the model's completion reasoning. This keeps responses
    fast and avoids KV-cache thrashing on long contexts.
    """

    # LLM for reasoning agents
    LLM_MODEL: str = field(
        default_factory=lambda: _env_str("LLM_MODEL", "gemma4:31b-it-q8_0")
    )

    # Fallback model used automatically when remote is unreachable
    # or when the primary model returns 404 on the local endpoint
    LLM_FALLBACK_MODEL: str = field(
        default_factory=lambda: _env_str("LLM_FALLBACK_MODEL", "mistral:7b")
    )

    # Embedding model for ChromaDB ingestion
    EMBEDDING_MODEL: str = field(
        default_factory=lambda: _env_str("EMBEDDING_MODEL", "nomic-embed-text")
    )

    # Hard limit on tokens sent to LLM per call (prompt only, not completion)
    MAX_CONTEXT_TOKENS: int = field(
        default_factory=lambda: _env_int("MAX_CONTEXT_TOKENS", 8000)
    )

    # Safety margin: trim input if within this many tokens of MAX_CONTEXT_TOKENS
    CONTEXT_SAFETY_MARGIN: int = field(
        default_factory=lambda: _env_int("CONTEXT_SAFETY_MARGIN", 500)
    )

    # Inference temperature (0.0 = deterministic, 1.0 = creative)
    TEMPERATURE: float = field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7)
    )

    # Lower temperature for JSON-strict calls to reduce hallucination
    JSON_TEMPERATURE: float = field(
        default_factory=lambda: _env_float("LLM_JSON_TEMPERATURE", 0.2)
    )

    # Max tokens the model is allowed to generate per response
    MAX_COMPLETION_TOKENS: int = field(
        default_factory=lambda: _env_int("MAX_COMPLETION_TOKENS", 4096)
    )

    # Supported models (used for validation in arg parser)
    SUPPORTED_MODELS: List[str] = field(
        default_factory=lambda: [
            # ── Remote models (primary) ───────────────────────────────────────
            "gemma4:31b-it-q8_0",        # best reasoning, use as primary
            "gemma4:26b-a4b-it-q8_0",    # MoE variant, faster
            "gemma4:e4b-it-bf16",         # full precision, highest quality
            "gpt-oss:20b",               # tiktoken fallback for tokenizer
            # ── Local models (fallback when remote unreachable) ───────────────
            "mistral:7b",                # best local option for instruction following
            "llama3:8b",
            "llama3.2:latest",
            "deepseek-coder:6.7b",
            "medgemma1.5:latest",
            "phi3:mini",
        ]
    )

    # Tokenizer mapping: model prefix → HuggingFace tokenizer name
    # Used by context_manager.py for accurate token counting.
    TOKENIZER_MAP: dict = field(
        default_factory=lambda: {
            # gemma4 uses gemma2 tokenizer — same vocabulary, verified on HF Hub
            "gemma4":  "google/gemma-2-9b-it",
            # mistral — verified public HF repo
            "mistral": "mistralai/Mistral-7B-v0.1",
            # llama3 variants — use smallest publicly accessible checkpoint
            "llama3":  "unsloth/Llama-3.2-1B-Instruct",
            # gpt-oss: no public HF tokenizer → intentionally omitted
            # so _resolve_hf_tokenizer returns None and tiktoken takes over
        }
    )

    # Tiktoken fallback model for when HF tokenizer isn't available
    TIKTOKEN_FALLBACK_ENCODING: str = field(
        default_factory=lambda: _env_str("TIKTOKEN_ENCODING", "cl100k_base")
    )

    # 10% safety buffer applied when using tiktoken (less accurate than HF)
    TIKTOKEN_SAFETY_BUFFER: float = field(
        default_factory=lambda: _env_float("TIKTOKEN_SAFETY_BUFFER", 0.10)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature flags
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FeatureFlags:
    """
    Boolean toggles for optional pipeline modules.

    Set to False to skip computationally expensive or externally-dependent
    phases. All flags can be overridden by environment variables or CLI args.

    Flag priority (highest → lowest):
        CLI args > Environment variables > Defaults below
    """

    # Chemical synthesis route evaluation via RDKit SA score + LLM
    # Requires synthesis_evaluator.py agent and synthesis_checker.py tool.
    ENABLE_CHEMICAL_SYNTHESIS: bool = field(
        default_factory=lambda: _env_bool("ENABLE_CHEMICAL_SYNTHESIS", False)
    )

    # Molecular docking via AutoDock Vina subprocess
    # If Vina binary not found, gracefully falls back to mock scores.
    ENABLE_DOCKING: bool = field(
        default_factory=lambda: _env_bool("ENABLE_DOCKING", True)
    )

    # Patent search via PatentsView API
    # Disable for air-gapped environments or to reduce API call latency.
    ENABLE_PATENT_SEARCH: bool = field(
        default_factory=lambda: _env_bool("ENABLE_PATENT_SEARCH", True)
    )

    # Force every LLM response through JSON validation + retry logic.
    # Set False only for debugging; never disable in production.
    STRICT_JSON_MODE: bool = field(
        default_factory=lambda: _env_bool("STRICT_JSON_MODE", True)
    )

    # Cache LLM responses to disk (speeds up re-runs during development).
    ENABLE_LLM_CACHE: bool = field(
        default_factory=lambda: _env_bool("ENABLE_LLM_CACHE", False)
    )

    # Generate PDF report in addition to Markdown (requires weasyprint or pandoc).
    ENABLE_PDF_REPORT: bool = field(
        default_factory=lambda: _env_bool("ENABLE_PDF_REPORT", False)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vector store (ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VectorStoreConfig:
    """ChromaDB configuration for literature and patent embeddings."""

    # Local persistence directory for ChromaDB
    PERSIST_DIR: str = field(
        default_factory=lambda: _env_str("CHROMA_PERSIST_DIR", "data/vectorstore")
    )

    # Collection names
    LITERATURE_COLLECTION: str = "drug_discovery_literature"
    PATENT_COLLECTION: str = "drug_discovery_patents"

    # Number of top results to retrieve per query
    TOP_K: int = field(
        default_factory=lambda: _env_int("CHROMA_TOP_K", 20)
    )

    # Distance metric for similarity search
    DISTANCE_METRIC: str = field(
        default_factory=lambda: _env_str("CHROMA_METRIC", "cosine")
    )

    # Batch size for embedding ingestion (avoid OOM on large corpora)
    EMBEDDING_BATCH_SIZE: int = field(
        default_factory=lambda: _env_int("CHROMA_EMBED_BATCH", 50)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Literature and patent retrieval
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RetrievalConfig:
    """Parameters controlling literature mining and patent search."""

    # PubMed E-utilities base URL
    PUBMED_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Maximum number of PubMed abstracts to fetch per query
    PUBMED_MAX_RESULTS: int = field(
        default_factory=lambda: _env_int("PUBMED_MAX_RESULTS", 150)
    )

    # Number of abstracts processed per LLM summarisation batch
    LITERATURE_BATCH_SIZE: int = field(
        default_factory=lambda: _env_int("LITERATURE_BATCH_SIZE", 50)
    )

    # ArXiv feed URL for preprint retrieval
    ARXIV_BASE_URL: str = "https://export.arxiv.org/api/query"

    # Maximum arXiv results per query
    ARXIV_MAX_RESULTS: int = field(
        default_factory=lambda: _env_int("ARXIV_MAX_RESULTS", 30)
    )

    # PatentsView API endpoint
    PATENTSVIEW_URL: str = "https://api.patentsview.org/patents/query"

    # Maximum patents to fetch per query
    PATENTS_MAX_RESULTS: int = field(
        default_factory=lambda: _env_int("PATENTS_MAX_RESULTS", 50)
    )

    # UniProt REST API base
    UNIPROT_BASE_URL: str = "https://rest.uniprot.org/uniprotkb"

    # Request timeout for all external HTTP calls (seconds)
    HTTP_TIMEOUT: int = field(
        default_factory=lambda: _env_int("HTTP_TIMEOUT", 30)
    )

    # User-Agent header for scraping (PubMed requires contact email)
    USER_AGENT: str = field(
        default_factory=lambda: _env_str(
            "HTTP_USER_AGENT",
            "DrugDiscoveryPipeline/1.0 (research; contact@example.com)",
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Molecule design
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MoleculeConfig:
    """Parameters for *in silico* molecule generation and filtering."""

    # Total candidates to generate before filtering
    GENERATION_COUNT: int = field(
        default_factory=lambda: _env_int("MOL_GENERATION_COUNT", 20)
    )

    # Top N molecules passed to docking after filtering
    SHORTLIST_COUNT: int = field(
        default_factory=lambda: _env_int("MOL_SHORTLIST_COUNT", 5)
    )

    # Lipinski Rule-of-Five thresholds
    MAX_MOLECULAR_WEIGHT: float = 500.0
    MAX_LOGP: float = 5.0
    MAX_HBD: int = 5    # H-bond donors
    MAX_HBA: int = 10   # H-bond acceptors
    MAX_LIPINSKI_VIOLATIONS: int = 1

    # QED (Quantitative Estimate of Drug-likeness) minimum threshold
    MIN_QED_SCORE: float = field(
        default_factory=lambda: _env_float("MIN_QED_SCORE", 0.5)
    )

    # SA (Synthetic Accessibility) score threshold (lower = easier to synthesise)
    MAX_SA_SCORE: float = field(
        default_factory=lambda: _env_float("MAX_SA_SCORE", 4.0)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Docking
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DockingConfig:
    """AutoDock Vina configuration."""

    # Path to Vina binary (must be on PATH or provide absolute path)
    VINA_BINARY: str = field(
        default_factory=lambda: _env_str("VINA_BINARY", "vina")
    )

    # Docking search box size (Ångströms)
    SEARCH_BOX_SIZE: tuple = (20, 20, 20)

    # Exhaustiveness of global search (higher = more accurate, slower)
    EXHAUSTIVENESS: int = field(
        default_factory=lambda: _env_int("VINA_EXHAUSTIVENESS", 8)
    )

    # Number of docking poses to generate per ligand
    NUM_POSES: int = field(
        default_factory=lambda: _env_int("VINA_NUM_POSES", 5)
    )

    # Parallel docking workers
    MAX_WORKERS: int = field(
        default_factory=lambda: _env_int("VINA_MAX_WORKERS", 4)
    )

    # RCSB PDB download URL template
    PDB_DOWNLOAD_URL: str = "https://files.rcsb.org/download/{pdb_id}.pdb"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-level settings
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Top-level pipeline meta-configuration."""

    VERSION: str ="1.0.0"

    # Output directory (relative to project root)
    OUTPUT_DIR: Path = field(
        default_factory=lambda: Path(_env_str("OUTPUT_DIR", "outputs"))
    )

    # LLM response cache directory (used when ENABLE_LLM_CACHE=True)
    CACHE_DIR: Path = field(
        default_factory=lambda: Path(_env_str("CACHE_DIR", "data/llm_cache"))
    )

    # JSONL log paths
    PIPELINE_LOG: Path = field(
        default_factory=lambda: Path("outputs/pipeline_log.jsonl")
    )
    ERROR_LOG: Path = field(
        default_factory=lambda: Path("outputs/error_log.jsonl")
    )

    # Maximum JSON retry attempts per agent call
    JSON_MAX_RETRIES: int = field(
        default_factory=lambda: _env_int("JSON_MAX_RETRIES", 3)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton-style accessors (instantiated once at import time)
# ─────────────────────────────────────────────────────────────────────────────
# Other modules import these directly:
#   from config import OllamaConfig, ModelConfig, FeatureFlags, ...
#
# Because @dataclass fields use default_factory, each instantiation reads
# the current os.environ, so values updated by main.py are respected.

OllamaConfig = OllamaConfig()        # type: ignore[assignment]
ModelConfig = ModelConfig()          # type: ignore[assignment]
FeatureFlags = FeatureFlags()        # type: ignore[assignment]
VectorStoreConfig = VectorStoreConfig()  # type: ignore[assignment]
RetrievalConfig = RetrievalConfig()  # type: ignore[assignment]
MoleculeConfig = MoleculeConfig()    # type: ignore[assignment]
DockingConfig = DockingConfig()      # type: ignore[assignment]
PipelineConfig = PipelineConfig()    # type: ignore[assignment]