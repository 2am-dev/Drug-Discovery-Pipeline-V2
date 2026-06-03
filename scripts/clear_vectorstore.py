"""
scripts/clear_vectorstore.py — Wipe the ChromaDB vector store between runs.
Place at: drug_discovery_pipeline/scripts/clear_vectorstore.py

Run with:
    python scripts/clear_vectorstore.py
    make clean-vectorstore
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    """Delete and recreate the ChromaDB persistence directory."""
    from dotenv import load_dotenv
    load_dotenv()

    from config import VectorStoreConfig

    persist_dir = Path(VectorStoreConfig.PERSIST_DIR)

    if not persist_dir.exists():
        print(f"Vectorstore directory does not exist: {persist_dir}")
        print("Nothing to clear.")
        return

    # Count items before deletion
    file_count = sum(1 for _ in persist_dir.rglob("*") if _.is_file())
    dir_size_mb = sum(
        f.stat().st_size for f in persist_dir.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    print(f"Vectorstore: {persist_dir}")
    print(f"Files: {file_count} ({dir_size_mb:.1f} MB)")
    print()

    confirm = input(
        "⚠️  This will DELETE all stored embeddings. "
        "Type 'yes' to confirm: "
    )
    if confirm.strip().lower() != "yes":
        print("Cancelled.")
        return

    shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Vectorstore cleared: {persist_dir}")


if __name__ == "__main__":
    main()