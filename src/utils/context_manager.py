"""
utils/context_manager.py — Token counting, text chunking, and context compression.

This module ensures that NO LLM call ever exceeds MAX_CONTEXT_TOKENS tokens
in its prompt. It does this by:

  1. Counting tokens accurately via HuggingFace AutoTokenizer (when available)
     or via tiktoken as a fast fallback with a 10% safety buffer.
  2. Chunking large texts into overlapping windows for batch processing.
  3. Summarising long literature batches so that only compressed findings
     are carried forward to downstream agents.
  4. Compressing pipeline state dictionaries by removing verbose fields
     that are no longer needed downstream.

Token counting strategy
───────────────────────
  Primary  → transformers.AutoTokenizer
             (downloaded from HuggingFace Hub on first use, ~100MB cache)
  Fallback → tiktoken (cl100k_base encoding) × 1.10 safety buffer

The 10% tiktoken buffer accounts for the fact that tiktoken (designed for
OpenAI's tokenizers) may under-count for llama/Gemma vocabularies.

Performance notes
─────────────────
- AutoTokenizer is loaded once and cached as a class attribute (_tokenizer).
- The load is done lazily (first call), not at import time, to avoid slowing
  down startup when the HuggingFace cache is cold.
- All chunking operations are synchronous (CPU-bound) and fast enough to run
  in the async event loop without blocking.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Optional

from loguru import logger

from config import ModelConfig


class ContextManager:
    """
    Manages token budgets and context compression for LLM prompts.

    Usage
    ─────
    ```python
    ctx = ContextManager()

    # Count tokens
    n = ctx.count_tokens("Hello, world!")
    n = ctx.count_messages_tokens([{"role": "user", "content": "Hi"}])

    # Chunk a long document
    chunks = ctx.chunk_text(long_text, max_tokens=2000, overlap=100)

    # Compress pipeline state before passing to next agent
    slim_state = ctx.compress_state(full_state, keep_keys=["hypothesis_result"])
    ```
    """

    # Cached tokenizer instance (shared across all ContextManager instances)
    _tokenizer: Any = None
    _tokenizer_loaded: bool = False
    _use_tiktoken: bool = False
    _tiktoken_enc: Any = None

    def __init__(self) -> None:
        # Attempt to load the appropriate tokenizer on first instantiation
        if not ContextManager._tokenizer_loaded:
            self._load_tokenizer()

    # ── Tokenizer loading ─────────────────────────────────────────────────────
    @classmethod
    def _load_tokenizer(cls) -> None:
        """
        Try to load the HuggingFace tokenizer for the configured model.

        Falls back to tiktoken if:
          - The transformers package is not installed.
          - The model's tokenizer is not in HuggingFace Hub.
          - Any other import/download error occurs.

        Side effects:
            Sets cls._tokenizer, cls._use_tiktoken, cls._tokenizer_loaded.
        """
        cls._tokenizer_loaded = True  # Set first to prevent recursive calls
        model_name = ModelConfig.LLM_MODEL  # e.g. "gemma4:31b-it-q8_0, "llama3.2:latest"

        # Determine the HuggingFace tokenizer name from the model prefix
        hf_model = cls._resolve_hf_tokenizer(model_name)

        if hf_model:
            try:
                from transformers import AutoTokenizer
                logger.info(
                    f"Loading HuggingFace tokenizer for '{hf_model}' "
                    "(first run: may download ~100 MB) …"
                )
                cls._tokenizer = AutoTokenizer.from_pretrained(
                    hf_model,
                    trust_remote_code=True,
                )
                cls._use_tiktoken = False
                logger.success(
                    f"HuggingFace tokenizer loaded: {hf_model}"
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"HuggingFace tokenizer load failed ({exc}). "
                    "Falling back to tiktoken."
                )

        # Tiktoken fallback
        cls._load_tiktoken_fallback()

    @classmethod
    def _resolve_hf_tokenizer(cls, ollama_model_name: str) -> Optional[str]:
        """
        Map an Ollama model name to a HuggingFace tokenizer repository.

        Args:
            ollama_model_name: e.g. "gemma4:31b-it-q8_0", "llama3.2:latest".

        Returns:
            str | None: HuggingFace repo name, or None if not mapped.
        """
        tokenizer_map = ModelConfig.TOKENIZER_MAP
        # The map keys are model *prefixes* (e.g. "Gemma4", "llama3")
        name_lower = ollama_model_name.lower()
        for prefix, hf_name in tokenizer_map.items():
            if name_lower.startswith(prefix):
                return hf_name
        logger.debug(
            f"No HuggingFace tokenizer mapping for '{ollama_model_name}'. "
            "Will use tiktoken."
        )
        return None

    @classmethod
    def _load_tiktoken_fallback(cls) -> None:
        """
        Load tiktoken as the fallback token counter.

        Side effects:
            Sets cls._tiktoken_enc and cls._use_tiktoken.
        """
        try:
            import tiktoken
            encoding_name = ModelConfig.TIKTOKEN_FALLBACK_ENCODING
            cls._tiktoken_enc = tiktoken.get_encoding(encoding_name)
            cls._use_tiktoken = True
            logger.info(
                f"Using tiktoken ({encoding_name}) with "
                f"{int(ModelConfig.TIKTOKEN_SAFETY_BUFFER * 100)}% safety buffer."
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"tiktoken also failed to load: {exc}. "
                "Token counting will use a rough character-based estimate."
            )
            cls._use_tiktoken = False

    # ── Token counting ─────────────────────────────────────────────────────────
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Uses HuggingFace tokenizer if available, otherwise tiktoken with
        a 10% safety buffer, otherwise a rough character/4 estimate.

        Args:
            text: Input string to tokenize.

        Returns:
            int: Token count (conservative estimate when using fallback).
        """
        if not text:
            return 0

        if not self._use_tiktoken and self._tokenizer is not None:
            # HuggingFace tokenizer — accurate
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            return len(ids)

        if self._use_tiktoken and self._tiktoken_enc is not None:
            # tiktoken — fast but slightly inaccurate for non-GPT models
            raw_count = len(self._tiktoken_enc.encode(text))
            # Apply safety buffer (round up)
            buffered = int(raw_count * (1.0 + ModelConfig.TIKTOKEN_SAFETY_BUFFER))
            return buffered

        # Last resort: character-based approximation (~4 chars per token)
        return max(1, len(text) // 4)

    def count_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """
        Count total tokens across a list of chat messages.

        Includes a small overhead per message (role token + separator) to
        match how Ollama/OpenAI counts chat completion tokens.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.

        Returns:
            int: Total token count (conservative).
        """
        total = 0
        for msg in messages:
            # Each message has ~4 overhead tokens for role/separators
            overhead = 4
            content = msg.get("content", "")
            total += self.count_tokens(content) + overhead
        # Add 2 for priming the reply
        return total + 2

    def fits_in_budget(
        self,
        text: str,
        budget: Optional[int] = None,
    ) -> bool:
        """
        Check whether a text fits within the token budget.

        Args:
            text: Input text to check.
            budget: Token budget (defaults to MAX_CONTEXT_TOKENS - safety margin).

        Returns:
            bool: True if text fits.
        """
        _budget = budget or (
            ModelConfig.MAX_CONTEXT_TOKENS - ModelConfig.CONTEXT_SAFETY_MARGIN
        )
        return self.count_tokens(text) <= _budget

    # ── Text chunking ──────────────────────────────────────────────────────────
    def chunk_text(
        self,
        text: str,
        max_tokens: int = 2000,
        overlap_tokens: int = 100,
    ) -> list[str]:
        """
        Split a long text into overlapping chunks, each within `max_tokens`.

        The overlap ensures that sentences split across chunk boundaries are
        still captured in at least one chunk's context.

        Algorithm:
          1. Split text into sentences.
          2. Greedily add sentences to current chunk until token limit reached.
          3. Start new chunk, backtracking `overlap_tokens` worth of sentences.

        Args:
            text: Input text to chunk.
            max_tokens: Maximum tokens per chunk.
            overlap_tokens: Token budget for overlap between consecutive chunks.

        Returns:
            list[str]: List of text chunks.
        """
        if self.count_tokens(text) <= max_tokens:
            return [text]

        # Split on sentence boundaries (period/exclamation/question + space)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sent_tokens = self.count_tokens(sentence)

            # Single sentence larger than max_tokens — hard-wrap it
            if sent_tokens > max_tokens:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                # Wrap long sentence by character
                hard_chunks = textwrap.wrap(sentence, width=max_tokens * 4)
                chunks.extend(hard_chunks)
                i += 1
                continue

            if current_tokens + sent_tokens <= max_tokens:
                current_sentences.append(sentence)
                current_tokens += sent_tokens
                i += 1
            else:
                # Flush current chunk
                if current_sentences:
                    chunks.append(" ".join(current_sentences))

                # Build overlap: walk back until we've accumulated overlap_tokens
                overlap_sentences: list[str] = []
                overlap_total = 0
                for s in reversed(current_sentences):
                    s_tok = self.count_tokens(s)
                    if overlap_total + s_tok > overlap_tokens:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_total += s_tok

                current_sentences = overlap_sentences
                current_tokens = overlap_total
                # Don't increment i — reprocess this sentence in the new chunk

        # Flush remaining sentences
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        logger.debug(
            f"chunk_text: {len(text)} chars → {len(chunks)} chunks "
            f"(max {max_tokens} tokens each)"
        )
        return chunks

    def chunk_documents(
        self,
        documents: list[str],
        max_tokens_per_batch: int = 4000,
    ) -> list[list[str]]:
        """
        Group a list of documents into batches that fit within a token budget.

        Used by the retriever agent to process abstracts in batches before
        passing them to the LLM for summarisation.

        Args:
            documents: List of document strings (e.g. PubMed abstracts).
            max_tokens_per_batch: Maximum total tokens per batch.

        Returns:
            list[list[str]]: List of batches, each a list of document strings.
        """
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for doc in documents:
            doc_tokens = self.count_tokens(doc)
            if doc_tokens > max_tokens_per_batch:
                # Single document exceeds batch limit — chunk it individually
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                sub_chunks = self.chunk_text(doc, max_tokens=max_tokens_per_batch)
                for chunk in sub_chunks:
                    batches.append([chunk])
                continue

            if current_tokens + doc_tokens > max_tokens_per_batch:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [doc]
                current_tokens = doc_tokens
            else:
                current_batch.append(doc)
                current_tokens += doc_tokens

        if current_batch:
            batches.append(current_batch)

        logger.debug(
            f"chunk_documents: {len(documents)} docs → {len(batches)} batches"
        )
        return batches

    # ── State compression ──────────────────────────────────────────────────────
    def compress_state(
        self,
        state: dict,
        keep_keys: Optional[list[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Reduce the pipeline state dictionary to its essential fields.

        Agents should call this before serialising state into a prompt.
        The goal is to pass ONLY what the next agent needs, not the entire
        history of all previous agents.

        Compression strategy:
          1. Keep only the keys listed in `keep_keys` (plus task metadata).
          2. Within each kept key, truncate long string values.
          3. For lists, keep only the first N items if list is too long.

        Args:
            state: Full pipeline state dictionary.
            keep_keys: Keys to preserve (all others are dropped).
                       Metadata keys (task_id, pipeline_version, etc.)
                       are always preserved.
            max_tokens: Token budget for the compressed state.
                        Defaults to MAX_CONTEXT_TOKENS / 2.

        Returns:
            dict: Compressed state dictionary.
        """
        _max = max_tokens or (ModelConfig.MAX_CONTEXT_TOKENS // 2)

        # Always-preserve metadata keys
        metadata_keys = {
            "task_id",
            "pipeline_version",
            "started_at",
            "input_type",
            "indication_or_target",
            "llm_model",
            "enable_synthesis",
            "enable_docking",
            "enable_patents",
        }

        target_keys = set(keep_keys or []) | metadata_keys

        # Filter to relevant keys
        compressed: dict = {
            k: v for k, v in state.items()
            if k in target_keys and v is not None
        }

        # Iteratively trim values until we fit in the budget
        import json as _json
        serialised = _json.dumps(compressed, default=str)
        if self.count_tokens(serialised) <= _max:
            return compressed

        # Trim strategy: shorten string values, truncate lists
        compressed = self._trim_dict_values(compressed, _max)
        return compressed

    def _trim_dict_values(self, data: dict, max_tokens: int) -> dict:
        """
        Recursively trim dictionary values to fit within token budget.

        Args:
            data: Dictionary to trim.
            max_tokens: Target token budget.

        Returns:
            dict: Trimmed dictionary.
        """
        import json as _json

        # Estimate current size
        current = _json.dumps(data, default=str)
        if self.count_tokens(current) <= max_tokens:
            return data

        trimmed = {}
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 500:
                # Truncate long strings to first 500 chars
                trimmed[key] = value[:500] + " … [truncated]"
            elif isinstance(value, list) and len(value) > 5:
                # Keep only first 5 list items
                trimmed[key] = value[:5]
                logger.debug(
                    f"compress_state: truncated list '{key}' "
                    f"from {len(value)} to 5 items."
                )
            elif isinstance(value, dict):
                # Recursively trim nested dicts
                trimmed[key] = self._trim_dict_values(value, max_tokens // 2)
            else:
                trimmed[key] = value

        return trimmed

    # ── Message compression ────────────────────────────────────────────────────
    def compress_messages(
        self,
        messages: list[dict[str, str]],
        budget: int,
    ) -> list[dict[str, str]]:
        """
        Reduce a message list to fit within the token budget.

        Strategy:
          1. Always keep the system message (index 0) intact.
          2. Always keep the last user message intact.
          3. Truncate or remove intermediate messages as needed.

        Args:
            messages: Full chat message list.
            budget: Maximum total tokens allowed.

        Returns:
            list[dict[str, str]]: Compressed message list.
        """
        if not messages:
            return messages

        current_tokens = self.count_messages_tokens(messages)
        if current_tokens <= budget:
            return messages

        logger.debug(
            f"compress_messages: {current_tokens} tokens → targeting {budget}"
        )

        result = list(messages)

        # Pass 1: Truncate long middle messages (not first or last)
        for i in range(1, len(result) - 1):
            if self.count_messages_tokens(result) <= budget:
                break
            content = result[i].get("content", "")
            if len(content) > 200:
                # Truncate to ~200 chars + summary note
                result[i] = dict(result[i])
                result[i]["content"] = (
                    content[:200] + " … [content truncated to fit context window]"
                )

        # Pass 2: Remove middle messages entirely if still over budget
        while len(result) > 2 and self.count_messages_tokens(result) > budget:
            # Remove the second message (keep system + last user)
            removed = result.pop(1)
            logger.debug(
                f"compress_messages: removed message "
                f"role='{removed.get('role')}' "
                f"(content length: {len(removed.get('content', ''))})"
            )

        # Pass 3: Truncate the last user message if still over budget
        if len(result) >= 1 and self.count_messages_tokens(result) > budget:
            last = result[-1]
            content = last.get("content", "")
            # Binary search for the right truncation point
            lo, hi = 0, len(content)
            while lo < hi - 10:
                mid = (lo + hi) // 2
                test = list(result)
                test[-1] = dict(last)
                test[-1]["content"] = content[:mid] + " … [truncated]"
                if self.count_messages_tokens(test) <= budget:
                    lo = mid
                else:
                    hi = mid
            result[-1] = dict(last)
            result[-1]["content"] = content[:lo] + " … [truncated]"

        final_tokens = self.count_messages_tokens(result)
        logger.info(
            f"compress_messages: compressed {current_tokens} → {final_tokens} tokens"
        )
        return result

    # ── Literature summarisation helper ───────────────────────────────────────
    def prepare_literature_batch(
        self,
        abstracts: list[str],
        max_batch_tokens: int = 3500,
    ) -> list[str]:
        """
        Prepare a batch of abstracts for LLM summarisation.

        Joins abstracts with separator, truncating the batch if needed.

        Args:
            abstracts: List of abstract strings.
            max_batch_tokens: Maximum tokens for the joined batch.

        Returns:
            list[str]: List of (potentially truncated) abstract strings
                       that together fit within max_batch_tokens.
        """
        result: list[str] = []
        total_tokens = 0
        separator_tokens = self.count_tokens("\n---\n")

        for abstract in abstracts:
            abstract_tokens = self.count_tokens(abstract)
            if total_tokens + abstract_tokens + separator_tokens > max_batch_tokens:
                logger.debug(
                    f"prepare_literature_batch: stopping at {len(result)} abstracts "
                    f"({total_tokens} tokens) to stay under {max_batch_tokens}."
                )
                break
            result.append(abstract)
            total_tokens += abstract_tokens + separator_tokens

        return result

    # ── Utility: truncate text to token limit ─────────────────────────────────
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text so that it fits within `max_tokens`.

        Uses binary search for efficiency on long texts.

        Args:
            text: Input text.
            max_tokens: Maximum token count.

        Returns:
            str: Truncated text (with " … [truncated]" suffix if truncation
                 was necessary).
        """
        if self.count_tokens(text) <= max_tokens:
            return text

        # Binary search over character positions
        lo, hi = 0, len(text)
        while lo < hi - 10:
            mid = (lo + hi) // 2
            if self.count_tokens(text[:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid

        truncated = text[:lo].rstrip() + " … [truncated]"
        logger.debug(
            f"truncate_to_tokens: {len(text)} chars → {lo} chars "
            f"({max_tokens} token limit)"
        )
        return truncated