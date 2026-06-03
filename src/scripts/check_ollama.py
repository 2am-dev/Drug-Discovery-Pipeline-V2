"""
scripts/check_ollama.py — Pre-flight Ollama connectivity checker.
Place at: drug_discovery_pipeline/scripts/check_ollama.py

Run with:
    python scripts/check_ollama.py
    make check-ollama
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path so we can import pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent))


async def check_endpoint(label: str, url: str) -> bool:
    """
    Check if an Ollama endpoint is reachable and return its available models.

    Args:
        label: Human-readable label ("remote" or "local").
        url: Full Ollama URL including /v1 suffix.

    Returns:
        bool: True if the endpoint is healthy.
    """
    import aiohttp

    root_url = url.rstrip("/").removesuffix("/v1")
    tags_url = f"{root_url}/api/tags"

    print(f"\n{'─' * 50}")
    print(f"Checking {label} Ollama: {tags_url}")

    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(tags_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    print(f"  ✅ {label.upper()} Ollama is ONLINE")
                    print(f"  URL: {url}")
                    if models:
                        print(f"  Available models ({len(models)}):")
                        for model in models:
                            print(f"    - {model}")
                    else:
                        print("  ⚠️  No models pulled yet.")
                        print(f"  Run: ollama pull gemma4:31b-it-q8_0")
                        print(f"  Run: ollama pull nomic-embed-text")
                    return True
                else:
                    print(f"  ❌ {label.upper()} Ollama returned HTTP {resp.status}")
                    return False
    except Exception as exc:
        print(f"  ❌ {label.upper()} Ollama UNREACHABLE: {exc}")
        return False


async def main() -> int:
    """Run connectivity checks for both remote and local Ollama endpoints."""
    from dotenv import load_dotenv
    load_dotenv()

    from config import OllamaConfig, ModelConfig

    print("=" * 50)
    print("Drug Discovery Pipeline — Ollama Connectivity Check")
    print("=" * 50)
    print(f"LLM Model:        {ModelConfig.LLM_MODEL}")
    print(f"Embedding Model:  {ModelConfig.EMBEDDING_MODEL}")

    remote_ok = False
    local_ok  = False

    if not OllamaConfig.SKIP_REMOTE:
        remote_ok = await check_endpoint("remote", OllamaConfig.REMOTE_URL)
    else:
        print("\n⏭  Remote Ollama skipped (OLLAMA_SKIP_REMOTE=true)")

    local_ok = await check_endpoint("local", OllamaConfig.LOCAL_URL)

    print(f"\n{'─' * 50}")
    print("SUMMARY:")

    if remote_ok or local_ok:
        active = "remote" if remote_ok else "local"
        print(f"  ✅ Pipeline will use {active.upper()} Ollama")
        print(f"     Make sure these models are pulled:")
        print(f"       ollama pull {ModelConfig.LLM_MODEL}")
        print(f"       ollama pull {ModelConfig.EMBEDDING_MODEL}")
        return 0
    else:
        print("  ❌ BOTH Ollama endpoints are UNREACHABLE")
        print("     Start Ollama with:  ollama serve")
        print("     Or set OLLAMA_REMOTE_URL in your .env file")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)