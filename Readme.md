# Drug Discovery Hypothesis-to-Report Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Remote%20%7C%20Local-green)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An autonomous, multi-agent drug discovery pipeline that accepts a **disease
indication** or **biological target** as input and produces a comprehensive
**Project Proposal Report** without human intervention.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Quick Start](#quick-start)
3. [Ollama Setup (Remote & Local)](#ollama-setup-remote--local)
4. [Configuration Reference](#configuration-reference)
5. [Feature Flags](#feature-flags)
6. [JSON Output Examples](#json-output-examples)
7. [Token Budget Explanation](#token-budget-explanation)
8. [Troubleshooting JSON Parsing Errors](#troubleshooting-json-parsing-errors)
9. [Project Structure](#project-structure)
10. [Contributing](#contributing)

---

## Architecture Overview

```
User Input (disease / target)
        │
        ▼
┌───────────────┐    JSON     ┌─────────────────┐    JSON    ┌──────────────────┐
│    Planner    │──────────▶│    Retriever     │──────────▶│   Hypothesis     │
│  (task graph) │            │ (lit + patents)  │            │  (target select) │
└───────────────┘            └─────────────────┘            └──────────────────┘
                                                                      │
                              ┌───────────────────────────────────────┘
                              ▼
                    ┌──────────────────┐    JSON    ┌──────────────────┐
                    │ Molecule Designer│──────────▶│ Docking Evaluator│
                    │  (SMILES + props)│            │  (Vina / mock)   │
                    └──────────────────┘            └──────────────────┘
                                                             │
                              ┌──────────────────────────────┘
                              ▼
                    ┌──────────────────┐    JSON    ┌──────────────────┐
                    │ Synthesis Eval.  │──────────▶│ Report Compiler  │
                    │  (OPTIONAL)      │            │  (MD / PDF)      │
                    └──────────────────┘            └──────────────────┘
```

Every inter-agent communication uses **strict JSON** validated against Pydantic
schemas. No free-text is passed between stages.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-org/drug_discovery_pipeline.git
cd drug_discovery_pipeline
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Ollama endpoints

```bash
cp .env.example .env
# Edit .env and set OLLAMA_REMOTE_URL to your server
```

### 3. Run the pipeline

```bash
# Basic run — disease indication
python main.py --indication "non-small cell lung cancer"

# Specific target, alternative model
python main.py --target EGFR --model gemma2:27b

# Full run with synthesis enabled
python main.py --indication "Alzheimer's disease" --enable-synthesis

# Local Ollama only (no remote server)
python main.py --indication "KRAS-driven pancreatic cancer" --local-only

# Fast debug run (no docking, no patents)
python main.py --indication "Type 2 diabetes" --no-docking --no-patents
```

---

## Ollama Setup (Remote & Local)

### Remote Server Setup

Your remote Ollama server must be running and accessible on port `11434`.

```bash
# On the remote server (Linux / NVIDIA GPU)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull gemma4:31b-it-q8_0        # Primary LLM (~20 GB)
ollama pull gemma4:26b-a4b-it-q8_0    # Alternative LLM (~16 GB)
ollama pull nomic-embed-text          # Embeddings (~274 MB)

# Start server with external access
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

> **Security Note:** Bind Ollama to `0.0.0.0` only on a trusted network or
> behind a VPN/firewall. Ollama has no built-in authentication.

Set your `.env`:

```env
OLLAMA_REMOTE_URL=http://192.168.1.100:11434/v1
```

### Local Setup (RTX 5070 Ti Fallback)

```bash
# On your local machine with 5070 Ti
ollama pull gemma4:31b-it-q8_0
ollama pull nomic-embed-text
ollama serve    # Listens on localhost:11434 by default
```

The pipeline automatically falls back to `localhost:11434` if the remote
server is unreachable (health check timeout: 5 seconds, configurable).

### Connectivity Priority

```
1. Remote Ollama (OLLAMA_REMOTE_URL)  ← tried first
        │  timeout or connection error
        ▼
2. Local Ollama (localhost:11434)     ← automatic fallback
        │  also fails
        ▼
3. RuntimeError raised               ← pipeline aborted
```

---

## Configuration Reference

All settings can be controlled via environment variables or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_REMOTE_URL` | `http://remote-server:11434/v1` | Remote Ollama endpoint |
| `OLLAMA_LOCAL_URL` | `http://localhost:11434/v1` | Local Ollama endpoint |
| `OLLAMA_HEALTH_TIMEOUT` | `5` | Health check timeout (seconds) |
| `OLLAMA_MAX_RETRIES` | `3` | Max LLM call retries |
| `LLM_MODEL` | `gemma4:31b-it-q8_0` | Primary LLM model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `MAX_CONTEXT_TOKENS` | `8000` | Max tokens per LLM prompt |
| `LLM_TEMPERATURE` | `0.7` | Inference temperature |
| `LLM_JSON_TEMPERATURE` | `0.2` | Temperature for JSON-strict calls |
| `CHROMA_PERSIST_DIR` | `data/vectorstore` | ChromaDB storage path |
| `PUBMED_MAX_RESULTS` | `150` | Max PubMed abstracts |
| `MOL_GENERATION_COUNT` | `20` | Molecules generated before filtering |
| `MOL_SHORTLIST_COUNT` | `5` | Top molecules passed to docking |
| `VINA_BINARY` | `vina` | Path to AutoDock Vina binary |
| `VINA_EXHAUSTIVENESS` | `8` | Vina search exhaustiveness |

---

## Feature Flags

Toggle optional modules via environment variables or CLI flags:

| Flag | Default | CLI Override | Description |
|---|---|---|---|
| `ENABLE_CHEMICAL_SYNTHESIS` | `false` | `--enable-synthesis` | RDKit SA score + LLM synthesis routes |
| `ENABLE_DOCKING` | `true` | `--no-docking` | AutoDock Vina molecular docking |
| `ENABLE_PATENT_SEARCH` | `true` | `--no-patents` | PatentsView API search |
| `STRICT_JSON_MODE` | `true` | — | JSON validation with retry |
| `ENABLE_LLM_CACHE` | `false` | — | Cache LLM responses to disk |
| `ENABLE_PDF_REPORT` | `false` | — | Generate PDF in addition to Markdown |

### Enabling Chemical Synthesis

```bash
# Via CLI
python main.py --indication "NSCLC" --enable-synthesis

# Via .env
ENABLE_CHEMICAL_SYNTHESIS=true

# Via environment variable
export ENABLE_CHEMICAL_SYNTHESIS=true
python main.py --indication "NSCLC"
```

When enabled, the pipeline adds a synthesis route evaluation phase using
RDKit's Synthetic Accessibility (SA) score and an LLM to propose step-by-step
synthetic routes for the top-ranked molecules.

---

## JSON Output Examples

### Planner Output

```json
{
  "task_id": "a3f8c2d1-4e56-7890-abcd-ef1234567890",
  "disease_or_target": "non-small cell lung cancer",
  "phases": [
    {"phase": "retrieval", "status": "pending", "dependencies": []},
    {"phase": "hypothesis", "status": "pending", "dependencies": ["retrieval"]},
    {"phase": "molecule_design", "status": "pending", "dependencies": ["hypothesis"]},
    {"phase": "docking", "status": "pending", "dependencies": ["molecule_design"]},
    {"phase": "synthesis", "status": "skipped", "dependencies": ["molecule_design"]},
    {"phase": "report", "status": "pending", "dependencies": ["docking"]}
  ],
  "estimated_duration_minutes": 45
}
```

### Hypothesis Agent Output

```json
{
  "selected_target": {
    "gene_name": "EGFR",
    "uniprot_id": "P00533",
    "pdb_id": "1M17",
    "binding_site_residues": ["L718", "V726", "A743", "M793"]
  },
  "hypothesis": {
    "mechanism": "Selective inhibition of the EGFR ATP-binding site prevents autophosphorylation of Y1068 and Y1173, blocking downstream RAS-MAPK and PI3K-AKT signaling cascades responsible for NSCLC proliferation.",
    "rationale": "EGFR T790M resistance mutation presents in 60% of patients after first-line TKI therapy. Third-generation covalent inhibitors targeting C797 show promise in overcoming this resistance.",
    "confidence_score": 0.87
  },
  "alternative_targets": [
    {"gene_name": "ALK", "uniprot_id": "Q9UM73", "rationale": "Present in 5% of NSCLC cases"}
  ]
}
```

### Molecule Designer Output

```json
{
  "generated_molecules": [
    {
      "smiles": "CCN(CC)CCNC(=O)c1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1",
      "molecular_weight": 445.9,
      "logP": 4.2,
      "qed_score": 0.72,
      "sa_score": 2.8,
      "lipinski_violations": 0,
      "generation_method": "scaffold_decoration"
    }
  ],
  "shortlisted_count": 5,
  "filtering_criteria": {
    "max_mw": 500,
    "max_logp": 5.0,
    "min_qed": 0.5,
    "max_sa": 4.0,
    "max_lipinski_violations": 1
  }
}
```

### Docking Output

```json
{
  "docking_results": [
    {
      "smiles": "CCN(CC)CCNC(=O)c1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1",
      "binding_affinity_kcal_mol": -8.4,
      "ligand_efficiency": 0.35,
      "pose_file": "outputs/pose_001.pdbqt",
      "key_interactions": ["H-bond with Met793", "Pi-stacking with Phe723"]
    }
  ],
  "receptor_pdb": "1M17",
  "docking_software": "AutoDock Vina 1.2.3"
}
```

---

## Token Budget Explanation

The pipeline operates within a strict **8,000-token prompt budget** per LLM call,
leaving ~24,000 tokens of headroom for model reasoning within Gemma4:31b's
32K context window.

```
Total context window:  32,000 tokens
Prompt budget:          8,000 tokens  ← our input
Completion headroom:   24,000 tokens  ← model's reasoning space
Safety margin:            500 tokens  ← never used, prevents edge cases
```

### Token Counting Strategy

```
1. Try HuggingFace AutoTokenizer (accurate, model-specific)
   └── Gemma4 → google/gemma4:31b-it-q8_0 tokenizer
   └── Gemma4  → google/gemma4:26b-a4b-it-q8_0 tokenizer

2. Fallback: tiktoken (cl100k_base encoding)
   └── Apply 10% safety buffer to compensate for encoding differences
   └── Effective limit = 8000 × (1 - 0.10) = 7,200 tokens
```

### Automatic Chunking Behaviour

| Input size | Action |
|---|---|
| < 7,500 tokens | Pass directly to LLM |
| 7,500–8,000 tokens | Trim least-critical fields |
| > 8,000 tokens | Summarise + compress before LLM call |

Literature reviews are processed in **batches of 50 abstracts**, each batch
summarised before the next batch is loaded. Only the compressed summaries
(~500 tokens each) are passed to downstream agents.

---

## Troubleshooting JSON Parsing Errors

### Symptom: `JSONDecodeError` or `ValidationError` in logs

The pipeline automatically retries up to 3 times with progressively stricter
prompts. If all 3 attempts fail, check:

**1. Model temperature too high**
```env
LLM_JSON_TEMPERATURE=0.1   # Reduce from default 0.2
```

**2. Model generating markdown code blocks**

The prompt explicitly forbids ` ```json ` blocks, but some models ignore this.
The `JSONValidator` strips them automatically. If stripping fails, check
`outputs/error_log.jsonl` for the raw response.

**3. Truncated JSON (response cut off)**

```env
MAX_COMPLETION_TOKENS=8192   # Increase from default 4096
```

**4. Model not following JSON structure**

Try a more instruction-following model:
```bash
python main.py --indication "NSCLC" --model gemma4:31b-it-q8_0  # Best JSON adherence
```

**5. Enable debug logging to see raw LLM responses**
```bash
python main.py --indication "NSCLC" --log-level DEBUG
```

Raw responses are written to `outputs/error_log.jsonl` on validation failure.

### Symptom: `ConnectionError` — cannot reach Ollama

```bash
# Check remote server
curl http://your-remote-server:11434/api/tags

# Check local server
curl http://localhost:11434/api/tags

# Force local-only mode
python main.py --indication "NSCLC" --local-only
```

### Symptom: `RuntimeError: No Ollama endpoint available`

Both remote and local Ollama are unreachable. Start at least one:
```bash
ollama serve   # On local machine
```

---

## Project Structure

```
drug_discovery_pipeline/
├── main.py                    # Entry point, CLI, pipeline orchestrator
├── config.py                  # All configuration (env-aware dataclasses)
├── .env.example               # Environment variable template
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── agents/                    # Multi-agent pipeline stages
│   ├── __init__.py
│   ├── planner.py             # Task graph builder (Phase 0)
│   ├── retriever.py           # Literature + patent mining (Phase 1)
│   ├── hypothesis.py          # Target selection + mechanism (Phase 2)
│   ├── molecule_designer.py   # In silico molecule generation (Phase 3)
│   ├── docking_evaluator.py   # AutoDock Vina binding affinity (Phase 4)
│   ├── synthesis_evaluator.py # [OPTIONAL] Synthetic route proposal (Phase 5)
│   └── report_compiler.py     # Markdown/PDF report generation (Phase 6)
│
├── tools/                     # Standalone utility tools used by agents
│   ├── __init__.py
│   ├── literature_search.py   # PubMed + arXiv scraping
│   ├── patent_search.py       # PatentsView API client
│   ├── target_lookup.py       # UniProt REST API client
│   ├── docking.py             # AutoDock Vina subprocess wrapper
│   ├── molecule_generator.py  # RDKit molecule generation + filtering
│   └── synthesis_checker.py   # [OPTIONAL] RDKit SA score calculator
│
├── utils/                     # Shared infrastructure utilities
│   ├── __init__.py
│   ├── ollama_client.py       # Smart remote/local Ollama switching
│   ├── json_validator.py      # JSON parsing, validation, retry logic
│   ├── context_manager.py     # Token counting, chunking, summarisation
│   ├── prompts.py             # JSON-enforced prompt templates
│   └── helpers.py             # Logging setup, file I/O, env loading
│
├── schemas/                   # Pydantic v2 response schemas
│   ├── __init__.py
│   ├── hypothesis.py          # HypothesisResponse, TargetSchema
│   ├── molecule.py            # MoleculeSchema, MoleculeDesignResponse
│   ├── docking.py             # DockingResultSchema, DockingResponse
│   └── report.py             # ReportMetadata, ReportResponse
│
├── outputs/                   # Generated files (git-ignored)
│   ├── pipeline_log.jsonl     # Full pipeline event log
│   ├── error_log.jsonl        # Error events with raw LLM responses
│   └── proposal_*.md          # Generated Markdown reports
│
└── data/
    └── vectorstore/           # ChromaDB persistent embeddings (git-ignored)
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Run `black .` and `ruff check .` before committing.
3. Add tests to `tests/` for any new agent or tool.
4. Ensure `mypy --strict` passes on all modified files.
5. Submit a pull request with a clear description of changes.

---

*Built with Ollama · Gemma4 · RDKit · ChromaDB · Pydantic v2*