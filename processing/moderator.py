"""
Content Moderator — Ollama async HTTP client

Uses a local Llama 3.2 3B model (via Ollama) to classify each post
into one of five safety labels:

    safe | spam | hate | nsfw | violence

The model is prompted with a strict JSON-output instruction.
On timeout or parse failure the fallback label is "safe" (fail-open
for throughput; adjust to "flagged" for stricter pipelines).
"""

from __future__ import annotations

import json
import os
import re

import httpx
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
_TIMEOUT = 8.0  # seconds — keep processing latency bounded
_FALLBACK_LABEL = "safe"
_VALID_LABELS = {"safe", "spam", "hate", "nsfw", "violence"}

_SYSTEM_PROMPT = """You are a strict content moderation classifier.
Classify the given social media post into EXACTLY ONE of these labels:
  safe | spam | hate | nsfw | violence

Rules:
- "safe"     → normal discussion, news, personal update
- "spam"     → unsolicited ads, repetitive links, bot content
- "hate"     → slurs, discrimination based on identity
- "nsfw"     → sexual content, explicit material
- "violence" → threats, graphic violence, self-harm encouragement

Respond with ONLY valid JSON and nothing else:
{"label": "<label>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}"""


async def classify(text: str, client: httpx.AsyncClient | None = None) -> dict:
    """
    Classify ``text`` using the local Ollama model.

    Returns a dict:
        {
          "label": str,        # one of _VALID_LABELS
          "confidence": float, # 0.0 – 1.0
          "reason": str,
          "flagged": bool,     # True when label != "safe"
        }
    """
    prompt = f"Post to classify:\n\"\"\"{text[:1000]}\"\"\""

    payload = {
        "model": OLLAMA_MODEL,
        "system": _SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic for moderation
            "num_predict": 128,
        },
    }

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(base_url=OLLAMA_URL, timeout=_TIMEOUT)

    try:
        response = await client.post("/api/generate", json=payload)
        response.raise_for_status()
        raw = response.json().get("response", "")
        return _parse_response(raw)
    except httpx.TimeoutException:
        logger.warning(f"Ollama timeout for text snippet: {text[:60]!r}")
        return _fallback()
    except httpx.HTTPStatusError as exc:
        logger.error(f"Ollama HTTP {exc.response.status_code}: {exc}")
        return _fallback()
    except Exception as exc:
        logger.error(f"Ollama unexpected error: {exc}")
        return _fallback()
    finally:
        if own_client:
            await client.aclose()


def _parse_response(raw: str) -> dict:
    """Extract JSON from the model response, with graceful fallback."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Find the first JSON object in the response
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        logger.debug(f"No JSON found in: {raw!r}")
        return _fallback()

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return _fallback()

    label = str(data.get("label", "")).lower().strip()
    if label not in _VALID_LABELS:
        label = _FALLBACK_LABEL

    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "label": label,
        "confidence": confidence,
        "reason": str(data.get("reason", ""))[:200],
        "flagged": label != "safe",
    }


def _fallback() -> dict:
    return {
        "label": _FALLBACK_LABEL,
        "confidence": 0.0,
        "reason": "fallback — model unavailable or timed out",
        "flagged": False,
    }
