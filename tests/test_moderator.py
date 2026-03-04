"""
Tests for processing.moderator

Uses respx to mock Ollama HTTP calls — no running LLM required.
"""

from __future__ import annotations

import json
import pytest
import httpx
import respx

from processing.moderator import classify, _parse_response, _fallback, _VALID_LABELS


# ------------------------------------------------------------------
# _parse_response unit tests (pure function, no HTTP)
# ------------------------------------------------------------------

class TestParseResponse:
    def test_valid_json(self):
        raw = '{"label": "hate", "confidence": 0.92, "reason": "contains slur"}'
        result = _parse_response(raw)
        assert result["label"] == "hate"
        assert result["confidence"] == pytest.approx(0.92)
        assert result["flagged"] is True

    def test_safe_label_not_flagged(self):
        raw = '{"label": "safe", "confidence": 0.99, "reason": "normal post"}'
        result = _parse_response(raw)
        assert result["label"] == "safe"
        assert result["flagged"] is False

    def test_unknown_label_falls_back(self):
        raw = '{"label": "unknown_category", "confidence": 0.8, "reason": "?"}'
        result = _parse_response(raw)
        assert result["label"] == "safe"  # fallback

    def test_markdown_fenced_json(self):
        raw = '```json\n{"label": "spam", "confidence": 0.85, "reason": "ad"}\n```'
        result = _parse_response(raw)
        assert result["label"] == "spam"

    def test_confidence_clamped(self):
        raw = '{"label": "nsfw", "confidence": 1.5, "reason": "over limit"}'
        result = _parse_response(raw)
        assert result["confidence"] <= 1.0

    def test_missing_json_returns_fallback(self):
        result = _parse_response("sorry, I cannot help")
        assert result["label"] == "safe"
        assert result["confidence"] == 0.0

    def test_all_valid_labels_accepted(self):
        for label in _VALID_LABELS:
            raw = json.dumps({"label": label, "confidence": 0.9, "reason": "test"})
            result = _parse_response(raw)
            assert result["label"] == label


# ------------------------------------------------------------------
# classify integration tests (mocked HTTP)
# ------------------------------------------------------------------

@pytest.mark.asyncio
class TestClassify:
    @respx.mock
    async def test_classify_hate_speech(self):
        response_body = {
            "model": "llama3.2:3b",
            "response": '{"label": "hate", "confidence": 0.95, "reason": "contains hate speech"}',
            "done": True,
        }
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json=response_body)
        )

        result = await classify("some hateful text here")
        assert result["label"] == "hate"
        assert result["flagged"] is True
        assert result["confidence"] > 0.0

    @respx.mock
    async def test_classify_safe_post(self):
        response_body = {
            "model": "llama3.2:3b",
            "response": '{"label": "safe", "confidence": 0.99, "reason": "normal post"}',
            "done": True,
        }
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json=response_body)
        )

        result = await classify("Just had a great coffee this morning!")
        assert result["label"] == "safe"
        assert result["flagged"] is False

    @respx.mock
    async def test_timeout_returns_fallback(self):
        respx.post("http://localhost:11434/api/generate").mock(
            side_effect=httpx.TimeoutException("timeout")
        )

        result = await classify("any text")
        assert result["label"] == "safe"
        assert result["confidence"] == 0.0
        assert "fallback" in result["reason"].lower()

    @respx.mock
    async def test_http_500_returns_fallback(self):
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(500, text="internal error")
        )

        result = await classify("some text")
        assert result["label"] == "safe"
        assert result["flagged"] is False

    @respx.mock
    async def test_label_parsing_from_noisy_response(self):
        # Model adds extra prose around the JSON
        response_body = {
            "model": "llama3.2:3b",
            "response": (
                'Based on my analysis, here is the classification:\n'
                '{"label": "spam", "confidence": 0.88, "reason": "promotional link"}\n'
                'Let me know if you need more details.'
            ),
            "done": True,
        }
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json=response_body)
        )

        result = await classify("Buy now! Click here!")
        assert result["label"] == "spam"

    @respx.mock
    async def test_long_text_truncated_in_prompt(self):
        """Ensures very long posts don't cause errors (truncated to 1000 chars)."""
        long_text = "x" * 5000
        response_body = {
            "model": "llama3.2:3b",
            "response": '{"label": "safe", "confidence": 0.7, "reason": "ok"}',
            "done": True,
        }
        route = respx.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json=response_body)
        )

        result = await classify(long_text)
        assert result["label"] == "safe"
        # Verify only one request was made (no error)
        assert route.called


# ------------------------------------------------------------------
# _fallback helper
# ------------------------------------------------------------------

def test_fallback_structure():
    fb = _fallback()
    assert fb["label"] == "safe"
    assert fb["flagged"] is False
    assert "fallback" in fb["reason"].lower()
    assert fb["confidence"] == 0.0
