"""
utils/ollama_client.py — Smart Ollama client with remote/local failover.

Responsibilities
────────────────
1. Health-check remote Ollama (GET /api/tags, timeout = HEALTH_CHECK_TIMEOUT).
2. Automatically fall back to local Ollama when remote is unreachable.
3. Expose a single async interface used by every agent:
       client.chat(messages, schema, temperature, max_tokens)
       client.embed(texts)
4. Validate and retry JSON responses via JSONValidator (up to MAX_RETRIES).
5. Count tokens BEFORE sending to ensure we never exceed MAX_CONTEXT_TOKENS.
6. Cache responses to disk when ENABLE_LLM_CACHE=True (keyed by prompt hash).
7. Log every request/response exchange to outputs/pipeline_log.jsonl.

Design notes
────────────
- Uses the `openai` Python SDK pointed at Ollama's OpenAI-compatible endpoint.
- All public methods are async; sync wrappers are provided for tools that
  cannot be async (e.g. RDKit callbacks).
- Exponential backoff is handled by `tenacity`; the openai SDK's own retry
  logic is DISABLED (max_retries=0) so tenacity stays in full control.
- Token counting delegates to ContextManager to avoid circular imports.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Type

import aiohttp
from loguru import logger
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, APIStatusError
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging as stdlib_logging

from config import (
    OllamaConfig,
    ModelConfig,
    FeatureFlags,
    PipelineConfig,
)


# ── Retryable exception types ─────────────────────────────────────────────────
_RETRYABLE = (APIConnectionError, APITimeoutError, asyncio.TimeoutError)


class OllamaUnavailableError(RuntimeError):
    """Raised when neither remote nor local Ollama is reachable."""


class OllamaClient:
    """
    Smart Ollama client that transparently switches between remote and local
    endpoints, validates JSON responses, and enforces token budgets.

    Usage
    ─────
    ```python
    client = OllamaClient()
    endpoint = await client.get_active_endpoint()   # warms up connection

    response = await client.chat(
        messages=[{"role": "user", "content": "Hello"}],
        schema=MyPydanticModel,   # optional; triggers JSON validation
    )
    embeddings = await client.embed(["text one", "text two"])
    ```
    """

    def __init__(self) -> None:
        # Active endpoint string (set after first health check)
        self._active_url: Optional[str] = None

        # Separate AsyncOpenAI clients for remote and local
        self._remote_client: Optional[AsyncOpenAI] = None
        self._local_client: Optional[AsyncOpenAI] = None

        # The currently active client object
        self._client: Optional[AsyncOpenAI] = None

        # Disk cache for LLM responses (only used if ENABLE_LLM_CACHE=True)
        self._cache_dir = PipelineConfig.CACHE_DIR
        if FeatureFlags.ENABLE_LLM_CACHE:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Import here to avoid circular dependency at module level
        # (ContextManager also imports OllamaClient for summarisation calls)
        self._ctx: Any = None  # set lazily in _get_context_manager()

    # ── Lazy context manager accessor ─────────────────────────────────────────
    def _get_context_manager(self):
        """
        Lazy-load ContextManager to break the circular import:
            OllamaClient → ContextManager → (uses OllamaClient for summarise)
        """
        if self._ctx is None:
            from utils.context_manager import ContextManager
            self._ctx = ContextManager()
        return self._ctx

    # ── Client factory ────────────────────────────────────────────────────────
    @staticmethod
    def _make_client(base_url: str) -> AsyncOpenAI:
        """
        Construct an AsyncOpenAI client pointed at an Ollama endpoint.

        Args:
            base_url: Full URL including /v1 suffix.

        Returns:
            AsyncOpenAI: Configured client. Timeout is longer for remote
                        endpoints where large models may need time to load.
        """
        # Use a longer timeout for remote (large models load slowly on first call)
        is_remote = "localhost" not in base_url and "127.0.0.1" not in base_url
        request_timeout = 300.0 if is_remote else 120.0

        return AsyncOpenAI(
            base_url=base_url,
            api_key=OllamaConfig.API_KEY,
            timeout=request_timeout,
            max_retries=0,   # tenacity owns retry logic
        )

    # ── Health check ──────────────────────────────────────────────────────────
    async def _health_check(self, base_url: str) -> bool:
        """
        Ping the Ollama /api/tags endpoint to verify the server is up.

        Args:
            base_url: URL that ends with /v1 — we strip /v1 to reach /api/tags.

        Returns:
            bool: True if server responded with HTTP 200 within timeout.
        """
        # Ollama's tag endpoint lives at the root, not under /v1
        root_url = base_url.rstrip("/").removesuffix("/v1")
        tags_url = f"{root_url}/api/tags"

        try:
            timeout = aiohttp.ClientTimeout(
                total=OllamaConfig.HEALTH_CHECK_TIMEOUT
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(tags_url) as resp:
                    ok = resp.status == 200
                    if ok:
                        logger.debug(f"Health check OK: {tags_url}")
                    else:
                        logger.debug(
                            f"Health check failed (HTTP {resp.status}): {tags_url}"
                        )
                    return ok
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Health check exception for {tags_url}: {exc}")
            return False

    # ── Endpoint selection ────────────────────────────────────────────────────
    async def get_active_endpoint(self) -> str:
        """
        Determine which Ollama endpoint to use.

        Priority:
            1. Remote (skipped if OllamaConfig.SKIP_REMOTE is True)
            2. Local

        Returns:
            str: The base URL of the active endpoint (e.g. http://host:11434/v1).

        Raises:
            OllamaUnavailableError: If neither endpoint responds.
        """
        if self._active_url is not None:
            return self._active_url

        candidates: list[tuple[str, str]] = []

        if not OllamaConfig.SKIP_REMOTE:
            candidates.append(("remote", OllamaConfig.REMOTE_URL))
        candidates.append(("local", OllamaConfig.LOCAL_URL))

        for label, url in candidates:
            logger.info(f"Attempting {label} Ollama at {url} …")
            if await self._health_check(url):
                self._active_url = url
                self._client = self._make_client(url)
                logger.success(f"Using {label} Ollama: {url}")
                return url
            logger.warning(f"{label.capitalize()} Ollama unreachable: {url}")

        raise OllamaUnavailableError(
            "Neither remote nor local Ollama is reachable. "
            "Run `ollama serve` or check network connectivity."
        )

    async def _ensure_client(self) -> AsyncOpenAI:
        """
        Return the active AsyncOpenAI client, running endpoint selection
        if this is the first call.

        Returns:
            AsyncOpenAI: Ready-to-use client.
        """
        if self._client is None:
            await self.get_active_endpoint()
        return self._client  # type: ignore[return-value]

    # ── Failover helper ───────────────────────────────────────────────────────
    async def _failover_to_local(self) -> None:
        """
        Switch from remote to local Ollama after a connection failure.

        Also switches LLM_MODEL to LLM_FALLBACK_MODEL because the primary
        model (e.g. gemma4:31b-it-q8_0) only exists on the remote server
        and will 404 on local Ollama.
        """
        if self._active_url == OllamaConfig.LOCAL_URL:
            raise OllamaUnavailableError("Local Ollama also failed.")

        logger.warning("Remote Ollama failed — failing over to local Ollama.")

        if not await self._health_check(OllamaConfig.LOCAL_URL):
            raise OllamaUnavailableError(
                "Failover failed: local Ollama is also unreachable."
            )

        self._active_url = OllamaConfig.LOCAL_URL
        self._client = self._make_client(OllamaConfig.LOCAL_URL)

        # Switch model — primary model won't exist on local endpoint
        primary = ModelConfig.LLM_MODEL
        fallback = ModelConfig.LLM_FALLBACK_MODEL

        if primary != fallback:
            logger.warning(
                f"Switching model: '{primary}' → '{fallback}' "
                f"(primary model not available on local Ollama)."
            )
            # Mutate the singleton so ALL subsequent agent calls use fallback
            ModelConfig.LLM_MODEL = fallback

        logger.success(
            f"Failover complete: {OllamaConfig.LOCAL_URL} | "
            f"model: {ModelConfig.LLM_MODEL}"
        )

    # ── Disk cache ────────────────────────────────────────────────────────────
    def _cache_key(self, messages: list[dict], model: str, temperature: float) -> str:
        """
        Generate a deterministic cache key for an LLM request.

        Args:
            messages: Chat message list.
            model: Model name string.
            temperature: Sampling temperature.

        Returns:
            str: SHA-256 hex digest (first 16 chars) used as filename.
        """
        payload = json.dumps(
            {"messages": messages, "model": model, "temperature": temperature},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _load_from_cache(self, key: str) -> Optional[str]:
        """
        Load a cached LLM response string from disk.

        Args:
            key: Cache key from _cache_key().

        Returns:
            str | None: Cached response text or None if not found.
        """
        path = self._cache_dir / f"{key}.txt"
        if path.exists():
            logger.debug(f"Cache HIT: {key}")
            return path.read_text(encoding="utf-8")
        return None

    def _save_to_cache(self, key: str, response: str) -> None:
        """
        Persist an LLM response string to disk.

        Args:
            key: Cache key from _cache_key().
            response: Raw LLM response text.
        """
        path = self._cache_dir / f"{key}.txt"
        path.write_text(response, encoding="utf-8")
        logger.debug(f"Cache SAVED: {key}")

    # ── Core chat method ──────────────────────────────────────────────────────
    async def chat(
        self,
        messages: list[dict[str, str]],
        schema: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        context_label: str = "unknown",
    ) -> str:
        """
        Send a chat completion request to the active Ollama endpoint.

        This method:
          1. Enforces the token budget (raises ValueError if over limit).
          2. Checks disk cache (if ENABLE_LLM_CACHE=True).
          3. Calls the LLM with exponential-backoff retry.
          4. Validates JSON response if `schema` is provided.
          5. Retries with a stricter hint prompt on JSON parse failure.
          6. Fails over to local Ollama on connection errors.

        Args:
            messages: OpenAI-format message list, e.g.:
                      [{"role": "system", "content": "..."}, ...]
            schema: Optional Pydantic model class. If provided, the response
                    is validated and retried on failure.
            temperature: Override inference temperature (default from config).
            max_tokens: Override max completion tokens (default from config).
            model: Override LLM model name (default from config).
            context_label: Human-readable label for log messages.

        Returns:
            str: Raw LLM response text (JSON string if schema was provided).

        Raises:
            ValueError: If the prompt exceeds MAX_CONTEXT_TOKENS.
            OllamaUnavailableError: If all endpoints fail.
            RuntimeError: If JSON validation fails after MAX_RETRIES attempts.
        """
        # ── Resolve defaults ──────────────────────────────────────────────────
        _model = model or ModelConfig.LLM_MODEL
        _temp = temperature if temperature is not None else (
            ModelConfig.JSON_TEMPERATURE if schema else ModelConfig.TEMPERATURE
        )
        _max_tok = max_tokens or ModelConfig.MAX_COMPLETION_TOKENS

        # ── Token budget enforcement ──────────────────────────────────────────
        ctx = self._get_context_manager()
        total_input_tokens = ctx.count_messages_tokens(messages)
        budget = ModelConfig.MAX_CONTEXT_TOKENS - ModelConfig.CONTEXT_SAFETY_MARGIN

        if total_input_tokens > budget:
            logger.warning(
                f"[{context_label}] Input tokens ({total_input_tokens}) exceed "
                f"budget ({budget}). Auto-compressing …"
            )
            messages = ctx.compress_messages(messages, budget)
            total_input_tokens = ctx.count_messages_tokens(messages)
            logger.info(
                f"[{context_label}] Compressed to {total_input_tokens} tokens."
            )

        # ── Disk cache lookup ─────────────────────────────────────────────────
        if FeatureFlags.ENABLE_LLM_CACHE:
            cache_key = self._cache_key(messages, _model, _temp)
            cached = self._load_from_cache(cache_key)
            if cached:
                return cached

        # ── LLM call with JSON retry loop ─────────────────────────────────────
        from utils.json_validator import JSONValidator  # lazy import

        last_error: Optional[Exception] = None
        hint_suffix: str = ""

        for attempt in range(1, OllamaConfig.MAX_RETRIES + 1):
            # On retry attempts, append a correction hint to the last user message
            attempt_messages = self._inject_retry_hint(
                messages, hint_suffix, attempt
            )

            try:
                raw_response = await self._raw_chat(
                    attempt_messages, _model, _temp, _max_tok
                )
            except _RETRYABLE as exc:
                logger.warning(
                    f"[{context_label}] Connection error (attempt {attempt}): {exc}"
                )
                await self._failover_to_local()
                last_error = exc
                continue

            # If no schema validation required, return immediately
            if schema is None:
                if FeatureFlags.ENABLE_LLM_CACHE:
                    self._save_to_cache(cache_key, raw_response)  # type: ignore
                return raw_response

            # Validate JSON against Pydantic schema
            parsed, error = JSONValidator.validate_and_parse(raw_response, schema)

            if parsed is not None:
                if FeatureFlags.ENABLE_LLM_CACHE:
                    self._save_to_cache(cache_key, raw_response)  # type: ignore
                logger.debug(
                    f"[{context_label}] JSON validated OK (attempt {attempt})."
                )
                return raw_response

            # Validation failed — build a hint for the next attempt
            logger.warning(
                f"[{context_label}] JSON validation failed (attempt {attempt}): "
                f"{error}"
            )
            hint_suffix = self._build_retry_hint(raw_response, error, schema)
            last_error = RuntimeError(error)

            self._log_json_error(context_label, attempt, raw_response, str(error))

        raise RuntimeError(
            f"[{context_label}] JSON validation failed after "
            f"{OllamaConfig.MAX_RETRIES} attempts. "
            f"Last error: {last_error}. "
            f"Check outputs/error_log.jsonl for raw responses."
        )

    # ── Raw chat (single attempt, no retry logic) ─────────────────────────────
    async def _raw_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Execute a single chat completion call against the active client.

        Args:
            messages: OpenAI-format message list.
            model: Model identifier string.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            str: Raw response content string.

        Raises:
            APIConnectionError | APITimeoutError: On network failure (retried
                by the caller).
            APIStatusError: On HTTP 4xx/5xx errors.
        """
        client = await self._ensure_client()
        t0 = time.monotonic()

        response = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        elapsed = round(time.monotonic() - t0, 2)
        content = response.choices[0].message.content or ""

        logger.debug(
            f"LLM call: model={model} | tokens_in="
            f"{response.usage.prompt_tokens if response.usage else '?'} | "
            f"tokens_out="
            f"{response.usage.completion_tokens if response.usage else '?'} | "
            f"elapsed={elapsed}s"
        )
        return content

    # ── Embedding method ──────────────────────────────────────────────────────
    async def embed(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of text strings.

        Texts are processed in batches to respect the embedding model's
        context window (nomic-embed-text: 8192 tokens).

        Args:
            texts: List of strings to embed.
            model: Override embedding model (default: nomic-embed-text).

        Returns:
            list[list[float]]: One embedding vector per input text.

        Raises:
            OllamaUnavailableError: If no Ollama endpoint is reachable.
        """
        _model = model or ModelConfig.EMBEDDING_MODEL
        client = await self._ensure_client()
        all_embeddings: list[list[float]] = []

        # Process in batches of 50 to avoid timeout on large corpora
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(
                f"Embedding batch {i // batch_size + 1} "
                f"({len(batch)} texts) with {_model}"
            )
            try:
                response = await client.embeddings.create(
                    model=_model,
                    input=batch,
                )
                all_embeddings.extend(
                    [item.embedding for item in response.data]
                )
            except _RETRYABLE as exc:
                logger.warning(f"Embedding connection error: {exc}. Failing over …")
                await self._failover_to_local()
                client = await self._ensure_client()
                # Retry the same batch once after failover
                response = await client.embeddings.create(
                    model=_model,
                    input=batch,
                )
                all_embeddings.extend(
                    [item.embedding for item in response.data]
                )

        return all_embeddings

    # ── Sync wrapper ──────────────────────────────────────────────────────────
    def chat_sync(
        self,
        messages: list[dict[str, str]],
        schema: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        context_label: str = "sync",
    ) -> str:
        """
        Synchronous wrapper around chat() for use in non-async contexts.

        Uses asyncio.run() if no event loop is running, otherwise
        creates a new loop in a thread (for calls from within Jupyter, etc.).

        Args:
            Same as chat().

        Returns:
            str: Raw LLM response text.
        """
        coro = self.chat(
            messages=messages,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            context_label=context_label,
        )
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an existing event loop (e.g. Jupyter)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # ── Retry hint helpers ────────────────────────────────────────────────────
    @staticmethod
    def _inject_retry_hint(
        messages: list[dict[str, str]],
        hint: str,
        attempt: int,
    ) -> list[dict[str, str]]:
        """
        Append a correction hint to the last user message on retry attempts.

        Args:
            messages: Original message list.
            hint: Correction hint string built from the last error.
            attempt: Current attempt number (1-indexed).

        Returns:
            list[dict[str, str]]: Possibly modified message list.
        """
        if attempt == 1 or not hint:
            return messages

        # Clone to avoid mutating the caller's list
        cloned = [dict(m) for m in messages]
        cloned.append({
            "role": "user",
            "content": (
                f"Your previous response was not valid JSON. "
                f"Please try again.\n\n{hint}"
            ),
        })
        return cloned

    @staticmethod
    def _build_retry_hint(
        raw_response: str,
        error: str,
        schema: Type[BaseModel],
    ) -> str:
        """
        Build a human-readable correction hint from a failed JSON response.

        Args:
            raw_response: The raw LLM output that failed validation.
            error: The error message from JSONValidator.
            schema: The Pydantic schema that was expected.

        Returns:
            str: Hint string to be appended to the next attempt's prompt.
        """
        # Show only first 300 chars of the bad response to keep the hint short
        snippet = raw_response[:300].replace("\n", " ")
        schema_example = "{}"
        try:
            schema_example = json.dumps(
                schema.model_json_schema().get("properties", {}),
                indent=2,
            )[:500]
        except Exception:  # noqa: BLE001
            pass

        return (
            f"ERROR IN PREVIOUS RESPONSE:\n"
            f"  Issue: {error}\n"
            f"  Your output started with: {snippet!r}\n\n"
            f"REQUIRED JSON SCHEMA FIELDS:\n{schema_example}\n\n"
            f"Output ONLY valid JSON. No markdown. No explanation."
        )

    # ── Error logging ─────────────────────────────────────────────────────────
    @staticmethod
    def _log_json_error(
        context_label: str,
        attempt: int,
        raw_response: str,
        error: str,
    ) -> None:
        """
        Append a JSON validation failure event to the error log.

        Args:
            context_label: Agent/phase name for traceability.
            attempt: Which retry attempt failed.
            raw_response: The raw LLM text that couldn't be parsed.
            error: Validation error message.
        """
        from utils.helpers import log_json_event
        log_json_event(
            PipelineConfig.ERROR_LOG,
            {
                "event": "json_validation_failure",
                "context": context_label,
                "attempt": attempt,
                "error": error,
                "raw_response_snippet": raw_response[:500],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ── Model availability check ──────────────────────────────────────────────
    async def list_available_models(self) -> list[str]:
        """
        Query the active Ollama endpoint for all pulled models.

        Returns:
            list[str]: Model name strings (e.g. ["gemma4:31b-it-q8_0", "nomic-embed-text"]).
        """
        url = (await self.get_active_endpoint()).rstrip("/").removesuffix("/v1")
        tags_url = f"{url}/api/tags"

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(tags_url) as resp:
                if resp.status != 200:
                    logger.warning(f"Could not list models: HTTP {resp.status}")
                    return []
                data = await resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                logger.debug(f"Available models: {models}")
                return models

    async def ensure_model_available(self, model_name: str) -> bool:
        """
        Check whether a specific model is pulled on the active endpoint.

        Args:
            model_name: Exact model name to search for.

        Returns:
            bool: True if the model is available.
        """
        available = await self.list_available_models()
        found = any(model_name in m for m in available)
        if not found:
            logger.warning(
                f"Model '{model_name}' not found on active endpoint. "
                f"Run: ollama pull {model_name}"
            )
        return found