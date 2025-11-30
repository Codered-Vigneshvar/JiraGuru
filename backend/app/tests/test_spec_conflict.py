"""Tests for spec conflict helpers."""

from app.ai import spec_conflict


def test_parse_response_handles_wrapped_json():
    text = "Result:\n```json\n{\"has_conflict\": false, \"conflicts\": []}\n```"
    parsed = spec_conflict._parse_response(text)  # type: ignore[attr-defined]
    assert parsed["has_conflict"] is False
    assert parsed["conflicts"] == []
