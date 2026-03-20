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


# ------------------------------------------------------------------
# TimeSeries helpers
# ------------------------------------------------------------------

async def ts_add(client: Redis, key: str, value: float, ts_ms: int | None = None) -> None:
    """Append a data point. Uses current time if ts_ms is None."""
    ts = ts_ms if ts_ms is not None else int(time.time() * 1000)
    try:
        await client.execute_command("TS.ADD", key, ts, value)
    except Exception as exc:
        if "TSDB" in str(exc) and "does not exist" in str(exc):
            await ensure_timeseries(client, key)
            await client.execute_command("TS.ADD", key, ts, value)
        else:
            logger.error(f"TS.ADD {key}: {exc}")


async def ts_range(
    client: Redis,
    key: str,
    from_ms: int = 0,
    to_ms: int = -1,
    aggregation: str | None = "sum",
    bucket_ms: int = 5_000,
) -> list[tuple[int, float]]:
    """
    Return (timestamp_ms, value) pairs from a TimeSeries key.
    Defaults to SUM aggregation in 5-second buckets.
    """
    cmd = ["TS.RANGE", key, from_ms if from_ms else "-", to_ms if to_ms != -1 else "+"]
    if aggregation:
        cmd += ["AGGREGATION", aggregation.upper(), bucket_ms]
    try:
        result = await client.execute_command(*cmd)
        return [(int(ts), float(v)) for ts, v in result]
    except Exception as exc:
        logger.debug(f"TS.RANGE {key}: {exc}")
        return []


# ------------------------------------------------------------------
# Topic metadata
# ------------------------------------------------------------------

async def set_topic_meta(client: Redis, topic_id: int, meta: dict[str, str]) -> None:
    await client.hset(f"topic:meta:{topic_id}", mapping=meta)


async def get_topic_meta(client: Redis, topic_id: int) -> dict[str, str]:
    raw = await client.hgetall(f"topic:meta:{topic_id}")
    return {k.decode(): v.decode() for k, v in raw.items()}


async def append_topic_sample(client: Redis, topic_id: int, text: str, max_samples: int = 5) -> None:
    """Keep the last N sample texts for a topic for labelling."""
    key = f"topic:meta:{topic_id}"
    existing = await client.hget(key, "samples")
    samples: list[str] = []
    if existing:
        import json
        samples = json.loads(existing.decode())
    samples.append(text[:120])
    samples = samples[-max_samples:]
    import json
    await client.hset(key, "samples", json.dumps(samples))


# ------------------------------------------------------------------
# Trending sorted set
# ------------------------------------------------------------------

async def update_trending(client: Redis, topic_id: int, key: str = "trending:now") -> None:
    """
    Increment topic_id in the trending sorted set.
    Periodically prunes stale entries older than the trending window.
    """
    await client.zincrby(key, 1, str(topic_id))
    await client.expire(key, _TRENDING_WINDOW_S * 2)


async def get_trending(client: Redis, top_n: int = 10) -> list[tuple[str, float]]:
    """Return (topic_id_str, score) pairs for the top-N trending topics."""
    result = await client.zrevrange("trending:now", 0, top_n - 1, withscores=True)
    return [(k.decode(), score) for k, score in result]


# ------------------------------------------------------------------
# Flagged posts list
# ------------------------------------------------------------------

async def push_flagged(
    client: Redis,
    uri: str,
    label: str,
    text_snippet: str,
    source: str = "bluesky",
    key: str = "flagged:recent",
) -> None:
    import json
    payload = json.dumps({"uri": uri, "label": label, "text": text_snippet[:140], "source": source})
    await client.lpush(key, payload)
    await client.ltrim(key, 0, _FLAGGED_MAX - 1)


async def get_flagged(client: Redis, count: int = 50) -> list[dict]:
    import json
    raw = await client.lrange("flagged:recent", 0, count - 1)
    out = []
    for item in raw:
        try:
            out.append(json.loads(item.decode()))
        except Exception:
            pass
    return out


# ------------------------------------------------------------------
# Counters
# ------------------------------------------------------------------

async def increment_counter(client: Redis, key: str, amount: int = 1) -> int:
    return int(await client.incrby(key, amount))


async def get_counter(client: Redis, key: str) -> int:
    val = await client.get(key)
    return int(val) if val else 0
