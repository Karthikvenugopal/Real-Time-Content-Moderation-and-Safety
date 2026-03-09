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
