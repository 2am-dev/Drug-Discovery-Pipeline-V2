"""
utils/json_validator.py — JSON response validation and sanitisation.

Every LLM response that is expected to conform to a schema passes through
this module before being used by any agent. The validator:

  1. Strips markdown code fences (```json … ``` or ``` … ```).
  2. Strips leading/trailing non-JSON prose.
  3. Attempts json.loads().
  4. Validates the parsed dict against the provided Pydantic v2 model.
  5. Returns (parsed_dict, None) on success or (None, error_message) on failure.

The caller (OllamaClient.chat) decides how many retries to perform.
This module only validates a single response — it has no network dependencies.

Example
───────
    from utils.json_validator import JSONValidator
    from schemas.hypothesis import HypothesisResponse

    parsed, error = JSONValidator.validate_and_parse(raw_text, HypothesisResponse)
    if error:
        print(f"Bad JSON: {error}")
    else:
        obj = HypothesisResponse(**parsed)
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Type, Tuple

from loguru import logger
from pydantic import BaseModel, ValidationError


class JSONValidator:
    """
    Stateless utility class for parsing and validating LLM JSON responses.

    All methods are static — no instantiation required.
    """

    # ── Regex patterns for stripping markdown artifacts ────────────────────────
    # Matches ```json ... ``` or ``` ... ``` (with optional language tag)
    _CODE_FENCE_RE = re.compile(
        r"```(?:json|JSON|python|text|xml|yaml|)?\s*([\s\S]*?)```",
        re.MULTILINE,
    )

    # Matches <json> ... </json> XML-style wrappers (some models use this)
    _XML_JSON_RE = re.compile(
        r"<json>\s*([\s\S]*?)\s*</json>",
        re.IGNORECASE | re.MULTILINE,
    )

    # Matches the first { or [ character — JSON must start here
    _JSON_START_RE = re.compile(r"[{\[]")

    # Matches the last } or ] character — JSON must end here
    _JSON_END_RE = re.compile(r"[}\]]")

    @classmethod
    def strip_markdown(cls, text: str) -> str:
        """
        Remove markdown formatting and extract the JSON payload.

        Tries multiple extraction strategies in order:
          1. Extract content from code fences (```json … ```).
          2. Extract content from <json> XML tags.
          3. Slice from the first { or [ to the last } or ].
          4. Return stripped text as-is (last resort).

        Args:
            text: Raw LLM response string, potentially containing markdown.

        Returns:
            str: Cleaned string that should be valid JSON.
        """
        text = text.strip()

        # Strategy 1: Code fences
        fence_match = cls._CODE_FENCE_RE.search(text)
        if fence_match:
            extracted = fence_match.group(1).strip()
            logger.debug("JSONValidator: stripped code fence.")
            return extracted

        # Strategy 2: XML-style <json> tags
        xml_match = cls._XML_JSON_RE.search(text)
        if xml_match:
            extracted = xml_match.group(1).strip()
            logger.debug("JSONValidator: stripped <json> tags.")
            return extracted

        # Strategy 3: Slice from first brace/bracket to last
        start_match = cls._JSON_START_RE.search(text)
        end_match = cls._JSON_END_RE.search(text[::-1])  # search from the right

        if start_match and end_match:
            start_idx = start_match.start()
            # end position in original string (reversed position → forward)
            end_idx = len(text) - end_match.start()
            sliced = text[start_idx:end_idx].strip()
            if sliced != text:
                logger.debug("JSONValidator: sliced to first/last brace.")
            return sliced

        # Strategy 4: Return as-is
        return text

    @classmethod
    def safe_parse(cls, text: str) -> Tuple[Optional[Any], Optional[str]]:
        """
        Attempt to parse a string as JSON after cleaning markdown artifacts.

        Args:
            text: Raw or cleaned string to parse.

        Returns:
            Tuple[Any | None, str | None]:
                - (parsed_object, None) on success.
                - (None, error_message) on failure.
        """
        cleaned = cls.strip_markdown(text)

        try:
            parsed = json.loads(cleaned)
            return parsed, None
        except json.JSONDecodeError as exc:
            # Attempt a second pass: replace smart quotes and common escaping issues
            repaired = cls._repair_json(cleaned)
            try:
                parsed = json.loads(repaired)
                logger.debug("JSONValidator: JSON repaired successfully.")
                return parsed, None
            except json.JSONDecodeError:
                return None, (
                    f"JSONDecodeError at line {exc.lineno}, col {exc.colno}: "
                    f"{exc.msg}. "
                    f"Input snippet: {cleaned[:200]!r}"
                )

    @classmethod
    def validate_and_parse(
        cls,
        text: str,
        schema: Type[BaseModel],
    ) -> Tuple[Optional[dict], Optional[str]]:
        """
        Parse `text` as JSON and validate it against a Pydantic v2 schema.

        This is the primary method called by OllamaClient after each LLM call.

        Args:
            text: Raw LLM response string.
            schema: Pydantic BaseModel subclass defining the expected structure.

        Returns:
            Tuple[dict | None, str | None]:
                - (validated_dict, None) on success.
                - (None, error_message) on failure.

        Example:
            >>> parsed, err = JSONValidator.validate_and_parse(raw, HypothesisResponse)
            >>> if err:
            ...     print(f"Validation failed: {err}")
            >>> else:
            ...     obj = HypothesisResponse(**parsed)
        """
        # Step 1: Parse JSON
        parsed, parse_error = cls.safe_parse(text)
        if parse_error:
            return None, f"JSON parse error: {parse_error}"

        # Step 2: Ensure it's a dict (some models return arrays at top level)
        if not isinstance(parsed, dict):
            return None, (
                f"Expected JSON object (dict), got {type(parsed).__name__}. "
                f"The schema requires a top-level JSON object."
            )

        # Step 3: Pydantic validation
        try:
            schema.model_validate(parsed)
            return parsed, None
        except ValidationError as exc:
            # Summarise validation errors concisely
            errors = exc.errors(include_url=False)
            error_summary = "; ".join(
                f"'{'.'.join(str(loc) for loc in e['loc'])}': {e['msg']}"
                for e in errors[:5]  # cap at 5 to avoid prompt bloat
            )
            return None, f"Pydantic validation error(s): {error_summary}"

    @classmethod
    def extract_partial(cls, text: str, keys: list[str]) -> dict:
        """
        Best-effort extraction of specific keys from malformed JSON.

        Used as a last resort when full validation keeps failing. Scans the
        response text with regex to find key-value pairs for the listed keys.

        Args:
            text: Raw LLM response (possibly malformed JSON).
            keys: List of top-level JSON keys to look for.

        Returns:
            dict: Partially extracted key-value pairs (may be empty).
        """
        result: dict = {}
        for key in keys:
            # Match "key": "value" or "key": 123 or "key": true/false/null
            pattern = re.compile(
                rf'"{re.escape(key)}"\s*:\s*('
                r'"(?:[^"\\]|\\.)*"'    # quoted string
                r'|\d+(?:\.\d+)?'       # number
                r'|true|false|null'     # boolean / null
                r'|\[.*?\]'             # simple array (non-greedy)
                r'|\{.*?\}'             # simple object (non-greedy)
                r')',
                re.DOTALL,
            )
            match = pattern.search(text)
            if match:
                raw_val = match.group(1)
                try:
                    result[key] = json.loads(raw_val)
                except json.JSONDecodeError:
                    result[key] = raw_val.strip('"')
        return result

    @staticmethod
    def _repair_json(text: str) -> str:
        """
        Apply heuristic fixes to common LLM JSON formatting mistakes.

        Fixes attempted:
          - Smart/curly quotes → straight quotes.
          - Trailing commas before } or ].
          - Single-quoted strings → double-quoted.
          - Unescaped newlines inside strings (replaced with \\n).

        Args:
            text: Potentially broken JSON string.

        Returns:
            str: Repaired string (may still be invalid).
        """
        # Smart quotes (common in GPT-family outputs)
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")

        # Trailing commas before closing brace/bracket
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # Single-quoted keys/values → double-quoted
        # Only apply to clearly single-quoted strings (risky, apply last)
        # This regex is conservative: only matches 'word' patterns
        text = re.sub(r"(?<![\\])'([^']*)'", r'"\1"', text)

        return text

    @classmethod
    def assert_valid(cls, text: str, schema: Type[BaseModel]) -> dict:
        """
        Parse and validate, raising a clear RuntimeError on failure.

        Convenience wrapper for non-retry contexts (e.g. tests).

        Args:
            text: Raw LLM response string.
            schema: Expected Pydantic schema.

        Returns:
            dict: Validated parsed dictionary.

        Raises:
            RuntimeError: On parse or validation failure.
        """
        parsed, error = cls.validate_and_parse(text, schema)
        if error:
            raise RuntimeError(
                f"JSON validation failed for schema "
                f"'{schema.__name__}': {error}\n"
                f"Raw response:\n{text[:500]}"
            )
        return parsed  # type: ignore[return-value]

    @classmethod
    def is_valid_json(cls, text: str) -> bool:
        """
        Quick boolean check: is this string parseable JSON?

        Args:
            text: String to check.

        Returns:
            bool: True if parseable as JSON (any type).
        """
        _, error = cls.safe_parse(text)
        return error is None

    @classmethod
    def pretty_print(cls, text: str) -> str:
        """
        Parse and re-format a JSON string with indentation for logging.

        Args:
            text: Raw JSON string.

        Returns:
            str: Indented JSON string, or original text on parse failure.
        """
        parsed, error = cls.safe_parse(text)
        if error or parsed is None:
            return text
        return json.dumps(parsed, indent=2, ensure_ascii=False)