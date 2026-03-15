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


# ------------------------------------------------------------------
# Main agent
# ------------------------------------------------------------------

@app.agent(raw_topic)
async def process_post(stream: faust.StreamT) -> None:  # type: ignore[type-arg]
    await _ensure_resources()
    clusterer = get_clusterer()

    async for raw_bytes in stream:
        t0 = time.perf_counter()

        # 1. Parse
        try:
            post_dict = json.loads(raw_bytes)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(f"JSON parse error: {exc}")
            continue

        text = post_dict.get("text", "")
        if not text:
            continue

        # 2. Embed (CPU, ~1 ms for 1 text)
        try:
            embedding: np.ndarray = embedder.embed_one(text)
        except Exception as exc:
            logger.error(f"Embedding error: {exc}")
            continue

        # 3. Moderate (async Ollama, ~200-800 ms on M3)
        try:
            mod = await moderator.classify(text, client=_http_client)
        except Exception as exc:
            logger.error(f"Moderator error: {exc}")
            mod = {"label": "safe", "confidence": 0.0, "reason": "", "flagged": False}

        # 4. Cluster (online, thread-safe)
        topic_id = clusterer.add(embedding)
        if topic_id is None:
            topic_id = -1  # model not yet initialised

        # 5. Store in Redis
        if _redis:
            ts_ms = int(time.time() * 1000)
            await redis_client.ts_add(_redis, f"moderation:{mod['label']}", 1, ts_ms)
            await redis_client.increment_counter(_redis, "counter:total")

            if topic_id >= 0:
                await redis_client.ts_add(_redis, f"trend:topic:{topic_id}", 1, ts_ms)
                await redis_client.update_trending(_redis, topic_id)
                # Periodically update sample texts for topic labelling
                if int(time.time()) % 10 == 0:
                    await redis_client.append_topic_sample(_redis, topic_id, text)

            if mod["flagged"]:
                await redis_client.push_flagged(
                    _redis, post_dict.get("uri", ""), mod["label"], text
                )
                await redis_client.increment_counter(_redis, "counter:flagged")

        latency_ms = (time.perf_counter() - t0) * 1000

        # 6. Publish enriched record
        enriched = {
            **post_dict,
            "label": mod["label"],
            "confidence": mod["confidence"],
            "reason": mod["reason"],
            "flagged": mod["flagged"],
            "topic_id": topic_id,
            "processed_at": time.time(),
            "latency_ms": round(latency_ms, 2),
        }
        await moderated_topic.send(value=json.dumps(enriched).encode())

        if int(time.time()) % 30 == 0:
            logger.info(
                f"label={mod['label']}  topic={topic_id}  "
                f"latency={latency_ms:.0f}ms  "
                f"clusterer_ready={clusterer.is_ready}  "
                f"total_seen={clusterer.total_seen}"
            )


# ------------------------------------------------------------------
# Periodic task: log throughput every 60 s
# ------------------------------------------------------------------

@app.timer(interval=60.0)
async def log_throughput() -> None:
    if _redis:
        total = await redis_client.get_counter(_redis, "counter:total")
        flagged = await redis_client.get_counter(_redis, "counter:flagged")
        trending = await redis_client.get_trending(_redis, top_n=5)
        rate = flagged / total * 100 if total else 0
        logger.info(
            f"[throughput] total={total:,}  flagged={flagged:,} ({rate:.1f}%)  "
            f"top_topics={[t[0] for t in trending]}"
        )
