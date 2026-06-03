# ─────────────────────────────────────────────────────────────────────────────
# Makefile — Drug Discovery Pipeline convenience commands
# Place at: drug_discovery_pipeline/Makefile   (project root)
#
# Usage:
#   make install         Install all dependencies
#   make run             Run pipeline with default settings
#   make test            Run test suite
#   make lint            Run ruff + black check
#   make format          Auto-format with black + ruff --fix
#   make clean           Remove generated outputs and cache
#   make check-ollama    Test Ollama connectivity
#   make docker-up       Start Ollama in Docker
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: all install install-dev install-rdkit run run-local run-synthesis \
        test test-fast test-integration lint format type-check \
        clean clean-outputs clean-cache clean-vectorstore \
        check-ollama pull-models docker-up docker-down \
        help

# ── Default target ────────────────────────────────────────────────────────────
all: help

# ── Python interpreter ────────────────────────────────────────────────────────
PYTHON  := python3
PIP     := $(PYTHON) -m pip
PYTEST  := $(PYTHON) -m pytest
BLACK   := $(PYTHON) -m black
RUFF    := $(PYTHON) -m ruff
MYPY    := $(PYTHON) -m mypy

# ── Default run settings (override from CLI: make run INDICATION="NSCLC") ─────
INDICATION  ?= non-small cell lung cancer
MODEL       ?= gemma4:31b-it-q8_0
LOG_LEVEL   ?= INFO

# ─────────────────────────────────────────────────────────────────────────────
# Installation
# ─────────────────────────────────────────────────────────────────────────────

install:
	@echo "Installing production dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Installation complete."

install-dev: install
	@echo "Installing development dependencies..."
	$(PIP) install \
		pytest>=7.4.0 \
		pytest-asyncio>=0.23.0 \
		pytest-mock>=3.12.0 \
		pytest-cov>=4.1.0 \
		black>=23.12.0 \
		ruff>=0.1.9 \
		mypy>=1.8.0
	@echo "✅ Dev dependencies installed."

install-rdkit:
	@echo "Installing RDKit (this may take a while)..."
	$(PIP) install rdkit
	@echo "✅ RDKit installed."

# ─────────────────────────────────────────────────────────────────────────────
# Running the pipeline
# ─────────────────────────────────────────────────────────────────────────────

run:
	@echo "Running pipeline: indication='$(INDICATION)' model=$(MODEL)"
	$(PYTHON) main.py \
		--indication "$(INDICATION)" \
		--model $(MODEL) \
		--log-level $(LOG_LEVEL)

run-local:
	@echo "Running pipeline (local Ollama only)..."
	$(PYTHON) main.py \
		--indication "$(INDICATION)" \
		--model $(MODEL) \
		--local-only \
		--log-level $(LOG_LEVEL)

run-synthesis:
	@echo "Running pipeline with synthesis evaluation enabled..."
	$(PYTHON) main.py \
		--indication "$(INDICATION)" \
		--model $(MODEL) \
		--enable-synthesis \
		--log-level $(LOG_LEVEL)

run-fast:
	@echo "Running pipeline (no docking, no patents — fast debug mode)..."
	$(PYTHON) main.py \
		--indication "$(INDICATION)" \
		--no-docking \
		--no-patents \
		--log-level DEBUG

run-target:
	@echo "Running pipeline with specific target: $(TARGET)"
	$(PYTHON) main.py \
		--target "$(TARGET)" \
		--model $(MODEL) \
		--log-level $(LOG_LEVEL)

# ─────────────────────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────────────────────

test:
	@echo "Running full test suite..."
	$(PYTEST) tests/ -v

test-fast:
	@echo "Running fast tests only (excluding slow/integration)..."
	$(PYTEST) tests/ -v -m "not slow and not integration"

test-integration:
	@echo "Running integration tests (requires live Ollama)..."
	$(PYTEST) tests/ -v -m "integration"

test-coverage:
	@echo "Running tests with coverage report..."
	$(PYTEST) tests/ --cov=. --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-schemas:
	@echo "Running schema tests only..."
	$(PYTEST) tests/test_schemas/ -v

test-utils:
	@echo "Running utils tests only..."
	$(PYTEST) tests/test_utils/ -v

# ─────────────────────────────────────────────────────────────────────────────
# Code quality
# ─────────────────────────────────────────────────────────────────────────────

lint:
	@echo "Running ruff linter..."
	$(RUFF) check .
	@echo "Running black format check..."
	$(BLACK) --check .
	@echo "✅ Lint complete."

format:
	@echo "Auto-formatting with black..."
	$(BLACK) .
	@echo "Auto-fixing with ruff..."
	$(RUFF) check --fix .
	@echo "✅ Formatting complete."

type-check:
	@echo "Running mypy type checker..."
	$(MYPY) agents/ utils/ schemas/ tools/ config.py main.py
	@echo "✅ Type check complete."

quality: lint type-check
	@echo "✅ All quality checks passed."

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

clean: clean-outputs clean-cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete."

clean-outputs:
	@echo "Removing pipeline outputs..."
	rm -rf outputs/reports/ outputs/poses/ outputs/*.jsonl outputs/*.log
	mkdir -p outputs/reports outputs/poses
	touch outputs/.gitkeep outputs/reports/.gitkeep outputs/poses/.gitkeep
	@echo "✅ Outputs cleared."

clean-cache:
	@echo "Removing LLM cache..."
	rm -rf data/llm_cache/
	mkdir -p data/llm_cache
	@echo "✅ LLM cache cleared."

clean-vectorstore:
	@echo "⚠️  Removing ChromaDB vectorstore (all embeddings will be lost)..."
	rm -rf data/vectorstore/
	mkdir -p data/vectorstore
	@echo "✅ Vectorstore cleared."

clean-pdb:
	@echo "Removing cached PDB structure files..."
	rm -rf data/pdb/
	mkdir -p data/pdb
	@echo "✅ PDB cache cleared."

clean-all: clean clean-vectorstore clean-pdb
	@echo "✅ Full cleanup complete."

# ─────────────────────────────────────────────────────────────────────────────
# Ollama management
# ─────────────────────────────────────────────────────────────────────────────

check-ollama:
	@echo "Checking Ollama connectivity..."
	$(PYTHON) scripts/check_ollama.py

pull-models:
	@echo "Pulling required Ollama models..."
	ollama pull gemma4:31b-it-q8_0
	ollama pull nomic-embed-text
	@echo "✅ Models pulled."

pull-models-alt:
	@echo "Pulling alternative Ollama models..."
	ollama pull gemma4:26b-a4b-it-q8_0
	ollama pull nomic-embed-text

serve-local:
	@echo "Starting local Ollama server..."
	OLLAMA_HOST=0.0.0.0:11434 ollama serve

# ─────────────────────────────────────────────────────────────────────────────
# Docker
# ─────────────────────────────────────────────────────────────────────────────

docker-up:
	@echo "Starting Docker services..."
	docker compose -f docker/docker-compose.yml up -d
	@echo "✅ Docker services started."

docker-down:
	@echo "Stopping Docker services..."
	docker compose -f docker/docker-compose.yml down
	@echo "✅ Docker services stopped."

docker-logs:
	docker compose -f docker/docker-compose.yml logs -f

docker-build:
	docker compose -f docker/docker-compose.yml build

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Drug Discovery Pipeline — Available Commands"
	@echo "════════════════════════════════════════════"
	@echo ""
	@echo "INSTALLATION:"
	@echo "  make install            Install production dependencies"
	@echo "  make install-dev        Install + dev/test dependencies"
	@echo "  make install-rdkit      Install RDKit separately"
	@echo ""
	@echo "RUNNING:"
	@echo "  make run                Run pipeline (set INDICATION='...')"
	@echo "  make run-local          Force local Ollama only"
	@echo "  make run-synthesis      Enable chemical synthesis evaluation"
	@echo "  make run-fast           Quick run: no docking, no patents"
	@echo "  make run-target TARGET=EGFR  Run with specific gene target"
	@echo ""
	@echo "  Examples:"
	@echo "    make run INDICATION='Alzheimer disease' MODEL=gemma2:27b"
	@echo "    make run-target TARGET=KRAS"
	@echo ""
	@echo "TESTING:"
	@echo "  make test               Full test suite"
	@echo "  make test-fast          Skip slow/integration tests"
	@echo "  make test-coverage      Coverage report → htmlcov/"
	@echo ""
	@echo "CODE QUALITY:"
	@echo "  make lint               Ruff + black check"
	@echo "  make format             Auto-format with black + ruff --fix"
	@echo "  make type-check         Run mypy"
	@echo ""
	@echo "CLEANUP:"
	@echo "  make clean              Remove __pycache__, build artefacts"
	@echo "  make clean-outputs      Remove generated reports and logs"
	@echo "  make clean-vectorstore  Wipe ChromaDB (⚠️  all embeddings lost)"
	@echo "  make clean-all          Full cleanup (everything above)"
	@echo ""
	@echo "OLLAMA:"
	@echo "  make check-ollama       Test remote + local connectivity"
	@echo "  make pull-models        Pull gemma4:31b-it-q8_0 + nomic-embed-text"
	@echo "  make serve-local        Start local Ollama on 0.0.0.0:11434"
	@echo ""
	@echo "DOCKER:"
	@echo "  make docker-up          Start all services"
	@echo "  make docker-down        Stop all services"
	@echo ""