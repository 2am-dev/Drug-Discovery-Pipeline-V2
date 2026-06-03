# ─────────────────────────────────────────────────────────────────────────────
# docker/Dockerfile — Drug Discovery Pipeline container
# Place at: drug_discovery_pipeline/docker/Dockerfile
#
# Build:   docker build -f docker/Dockerfile -t drug-discovery .
# Run:     docker run --env-file .env drug-discovery \
#              --indication "NSCLC" --local-only
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bookworm

# System dependencies for RDKit, WeasyPrint, and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libcairo2-dev \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    libssl-dev \
    # AutoDock Vina dependency
    libboost-filesystem1.74.0 \
    libboost-thread1.74.0 \
    # Pandoc for PDF generation (optional)
    pandoc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash pipeline
WORKDIR /app
RUN chown pipeline:pipeline /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY --chown=pipeline:pipeline . .

# Create necessary directories
RUN mkdir -p outputs/reports outputs/poses data/vectorstore data/pdb data/llm_cache \
    && chown -R pipeline:pipeline outputs data

# Switch to non-root user
USER pipeline

# Pre-download HuggingFace tokenizers (optional — uncomment to bake into image)
# RUN python -c "from transformers import AutoTokenizer; \
#     AutoTokenizer.from_pretrained('gemma4/gemma4:31b-it-q8_0')"

# Health check: verify the pipeline can import successfully
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import main; import config; print('OK')" || exit 1

# Default entry point
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]