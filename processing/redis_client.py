"""
Redis helper layer — TimeSeries, Hash, SortedSet, and counter operations.

All public functions are async and accept a redis.asyncio.Redis instance.

Key schema
----------
trend:topic:{id}          TimeSeries  — post volume per topic (1 h retention)
moderation:{label}        TimeSeries  — post volume per label  (1 h retention)
topic:meta:{id}           Hash        — cluster metadata (label, sample texts …)
trending:now              SortedSet   — top topics in last 15 min (score = count)
flagged:recent            List        — last 200 flagged post URIs
counter:total             Integer     — lifetime post count
counter:flagged           Integer     — lifetime flagged count
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger
from redis.asyncio import Redis

# Retention for time-series keys: 1 hour in milliseconds
_TS_RETENTION_MS = 3_600_000
# Trending window: 15 minutes in seconds
_TRENDING_WINDOW_S = 900
# How many flagged URIs to keep
_FLAGGED_MAX = 200


# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------

async def ensure_timeseries(client: Redis, key: str, labels: dict[str, str] | None = None) -> None:
    """Create a TimeSeries key if it does not already exist."""
    try:
        args: list[Any] = ["TS.CREATE", key, "RETENTION", _TS_RETENTION_MS, "DUPLICATE_POLICY", "SUM"]
        if labels:
            args += ["LABELS"] + [item for pair in labels.items() for item in pair]
        await client.execute_command(*args)
    except Exception as exc:
        if "already exists" in str(exc).lower() or "key already exists" in str(exc).lower():
            return
        logger.warning(f"TS.CREATE {key}: {exc}")


async def bootstrap(client: Redis, n_topics: int = 20) -> None:
    """Idempotently create all required TimeSeries keys."""
    for topic_id in range(n_topics):
        await ensure_timeseries(
            client,
            f"trend:topic:{topic_id}",
            labels={"type": "topic", "id": str(topic_id)},
        )
    for label in ("safe", "spam", "hate", "nsfw", "violence"):
        await ensure_timeseries(
            client,
            f"moderation:{label}",
            labels={"type": "moderation", "label": label},
        )
    logger.info("Redis bootstrap complete")
