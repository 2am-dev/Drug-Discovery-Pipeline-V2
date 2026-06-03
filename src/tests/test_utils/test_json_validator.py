"""
tests/test_utils/test_json_validator.py
Place at: drug_discovery_pipeline/tests/test_utils/test_json_validator.py
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from utils.json_validator import JSONValidator


class SimpleSchema(BaseModel):
    name: str
    value: int
    score: float


class TestStripMarkdown:
    """Tests for JSONValidator.strip_markdown()"""

    def test_strips_json_code_fence(self):
        text = '```json\n{"name": "test", "value": 1, "score": 0.5}\n```'
        result = JSONValidator.strip_markdown(text)
        assert result.startswith("{")

    def test_strips_plain_code_fence(self):
        text = '```\n{"name": "test", "value": 1, "score": 0.5}\n```'
        result = JSONValidator.strip_markdown(text)
        assert result.startswith("{")

    def test_strips_xml_json_tags(self):
        text = '<json>{"name": "test", "value": 1, "score": 0.5}</json>'
        result = JSONValidator.strip_markdown(text)
        assert result.startswith("{")

    def test_slices_prose_before_json(self):
        text = 'Here is the JSON output:\n{"name": "test", "value": 1, "score": 0.5}'
        result = JSONValidator.strip_markdown(text)
        assert result.startswith("{")

    def test_passthrough_clean_json(self):
        text = '{"name": "test", "value": 1, "score": 0.5}'
        result = JSONValidator.strip_markdown(text)
        assert result == text


class TestSafeParse:
    """Tests for JSONValidator.safe_parse()"""

    def test_parses_valid_json(self):
        parsed, error = JSONValidator.safe_parse('{"key": "value"}')
        assert parsed == {"key": "value"}
        assert error is None

    def test_returns_error_for_invalid_json(self):
        parsed, error = JSONValidator.safe_parse("not valid json")
        assert parsed is None
        assert error is not None

    def test_repairs_trailing_comma(self):
        text = '{"name": "test", "value": 1,}'
        parsed, error = JSONValidator.safe_parse(text)
        # Should repair and parse successfully
        assert error is None or parsed is not None

    def test_repairs_smart_quotes(self):
        text = '\u201cname\u201d: \u201ctest\u201d'
        # Not valid JSON even after repair — just shouldn't crash
        JSONValidator.safe_parse(text)


class TestValidateAndParse:
    """Tests for JSONValidator.validate_and_parse()"""

    def test_validates_correct_schema(self):
        text = '{"name": "EGFR", "value": 42, "score": 0.87}'
        parsed, error = JSONValidator.validate_and_parse(text, SimpleSchema)
        assert parsed is not None
        assert error is None
        assert parsed["name"] == "EGFR"

    def test_returns_error_for_missing_field(self):
        text = '{"name": "EGFR", "value": 42}'  # missing score
        parsed, error = JSONValidator.validate_and_parse(text, SimpleSchema)
        assert parsed is None
        assert "score" in error.lower() or "validation" in error.lower()

    def test_returns_error_for_wrong_type(self):
        text = '{"name": "EGFR", "value": "not_an_int", "score": 0.87}'
        parsed, error = JSONValidator.validate_and_parse(text, SimpleSchema)
        assert parsed is None
        assert error is not None

    def test_returns_error_for_array_instead_of_object(self):
        text = '[{"name": "EGFR", "value": 42, "score": 0.87}]'
        parsed, error = JSONValidator.validate_and_parse(text, SimpleSchema)
        assert parsed is None
        assert "dict" in error.lower() or "object" in error.lower()

    def test_strips_code_fence_before_validating(self):
        text = '```json\n{"name": "EGFR", "value": 42, "score": 0.87}\n```'
        parsed, error = JSONValidator.validate_and_parse(text, SimpleSchema)
        assert parsed is not None
        assert error is None


class TestIsValidJson:
    def test_valid_json_returns_true(self):
        assert JSONValidator.is_valid_json('{"key": "value"}') is True

    def test_invalid_json_returns_false(self):
        assert JSONValidator.is_valid_json("not json") is False

    def test_empty_string_returns_false(self):
        assert JSONValidator.is_valid_json("") is False


class TestExtractPartial:
    def test_extracts_string_value(self):
        text = '{"gene_name": "EGFR", "other": "stuff"}'
        result = JSONValidator.extract_partial(text, ["gene_name"])
        assert result.get("gene_name") == "EGFR"

    def test_extracts_numeric_value(self):
        text = '{"confidence_score": 0.87, "text": "hello"}'
        result = JSONValidator.extract_partial(text, ["confidence_score"])
        assert result.get("confidence_score") == pytest.approx(0.87)

    def test_returns_empty_for_missing_key(self):
        text = '{"gene_name": "EGFR"}'
        result = JSONValidator.extract_partial(text, ["missing_key"])
        assert result == {}