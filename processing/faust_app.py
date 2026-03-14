"""
Faust Stream Processor — Real-Time Content Moderation Pipeline

Consumes from `bluesky.raw`, runs each post through:
  1. Embedding   (sentence-transformers all-MiniLM-L6-v2)
  2. Moderation  (Ollama llama3.2:3b — async, 8 s timeout)
  3. Clustering  (online MiniBatchKMeans, n=20)
  4. Storage     (Redis TimeSeries + SortedSet + counters)

Output topic `bluesky.moderated` carries enriched post records.

Run:
  python -m faust -A processing.faust_app worker -l info
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import faust
import httpx
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from redis.asyncio import Redis

from processing import embedder, moderator, redis_client
from processing.topic_clusterer import get_clusterer

load_dotenv()

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
TOPIC_IN = os.getenv("TOPIC_BLUESKY_RAW", "bluesky.raw")
TOPIC_OUT = os.getenv("TOPIC_BLUESKY_MODERATED", "bluesky.moderated")

# ------------------------------------------------------------------
# Faust app + topics
# ------------------------------------------------------------------

app = faust.App(
    "content-moderation",
    broker=f"kafka://{KAFKA_BROKER}",
    value_serializer="raw",
    topic_replication_factor=1,
    broker_credentials=None,
    web_enabled=False,
)

raw_topic = app.topic(TOPIC_IN, value_type=bytes)
moderated_topic = app.topic(TOPIC_OUT, value_type=bytes)


# ------------------------------------------------------------------
# Shared async resources (lazy-initialised on first agent call)
# ------------------------------------------------------------------

_redis: Redis | None = None
_http_client: httpx.AsyncClient | None = None
_initialized = False


async def _ensure_resources() -> None:
    """Lazily create Redis and HTTP client on first use."""
    global _redis, _http_client, _initialized
    if _initialized:
        return
    _redis = Redis.from_url(REDIS_URL, decode_responses=False)
    _http_client = httpx.AsyncClient(
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        timeout=10.0,
    )
    await redis_client.bootstrap(_redis)
    _initialized = True
    logger.info("Redis and HTTP client initialised")


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

@dataclass
class RawPost(faust.Record, serializer="json"):
    did: str = ""
    uri: str = ""
    text: str = ""
    created_at: str = ""
    langs: list[str] = field(default_factory=list)
    ingested_at: str = ""
    has_embed: bool = False
    reply: bool = False


@dataclass
class ModeratedPost(faust.Record, serializer="json"):
    # Original fields
    did: str = ""
    uri: str = ""
    text: str = ""
    created_at: str = ""
    ingested_at: str = ""
    # Moderation
    label: str = "safe"
    confidence: float = 0.0
    reason: str = ""
    flagged: bool = False
    # Clustering
    topic_id: int = -1
    # Timing
    processed_at: float = 0.0
    latency_ms: float = 0.0
