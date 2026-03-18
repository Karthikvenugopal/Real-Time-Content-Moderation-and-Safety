"""
Tests for processing.redis_client

Uses AsyncMock to simulate Redis — no live Redis required.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from processing import redis_client


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_redis():
    """Return a Redis-like AsyncMock with common commands stubbed."""
    r = AsyncMock()
    r.execute_command = AsyncMock(return_value=True)
    r.hset = AsyncMock(return_value=1)
    r.hget = AsyncMock(return_value=None)
    r.hgetall = AsyncMock(return_value={})
    r.zincrby = AsyncMock(return_value=1.0)
    r.expire = AsyncMock(return_value=True)
    r.zrevrange = AsyncMock(return_value=[])
    r.lpush = AsyncMock(return_value=1)
    r.ltrim = AsyncMock(return_value=True)
    r.lrange = AsyncMock(return_value=[])
    r.incrby = AsyncMock(return_value=1)
    r.get = AsyncMock(return_value=None)
    return r


# ------------------------------------------------------------------
# ensure_timeseries
# ------------------------------------------------------------------

@pytest.mark.asyncio
class TestEnsureTimeseries:
    async def test_creates_new_key(self, mock_redis):
        mock_redis.execute_command.return_value = "OK"
        await redis_client.ensure_timeseries(mock_redis, "test:key")
        mock_redis.execute_command.assert_called_once()
        args = mock_redis.execute_command.call_args[0]
        assert args[0] == "TS.CREATE"
        assert args[1] == "test:key"

    async def test_already_exists_is_silent(self, mock_redis):
        mock_redis.execute_command.side_effect = Exception("key already exists")
        # Should not raise
        await redis_client.ensure_timeseries(mock_redis, "test:key")

    async def test_labels_included(self, mock_redis):
        await redis_client.ensure_timeseries(
            mock_redis, "trend:topic:0", labels={"type": "topic"}
        )
        args = mock_redis.execute_command.call_args[0]
        assert "LABELS" in args
        assert "type" in args
        assert "topic" in args


# ------------------------------------------------------------------
# ts_add
# ------------------------------------------------------------------

@pytest.mark.asyncio
class TestTsAdd:
    async def test_add_with_explicit_ts(self, mock_redis):
        await redis_client.ts_add(mock_redis, "moderation:safe", 1.0, ts_ms=12345)
        mock_redis.execute_command.assert_called_with("TS.ADD", "moderation:safe", 12345, 1.0)

    async def test_add_with_auto_ts(self, mock_redis):
        before = int(time.time() * 1000)
        await redis_client.ts_add(mock_redis, "moderation:spam", 1.0)
        after = int(time.time() * 1000)
        args = mock_redis.execute_command.call_args[0]
        assert args[0] == "TS.ADD"
        assert before <= args[2] <= after

    async def test_auto_creates_on_missing_key(self, mock_redis):
        # First call raises TSDB key-not-found; second succeeds
        mock_redis.execute_command.side_effect = [
            Exception("TSDB: the key does not exist"),
            "OK",
            "OK",  # for the ensure_timeseries re-attempt
        ]
        # Should not raise
        await redis_client.ts_add(mock_redis, "new:key", 1.0)


# ------------------------------------------------------------------
# update_trending / get_trending
# ------------------------------------------------------------------

@pytest.mark.asyncio
class TestTrending:
    async def test_update_trending_increments(self, mock_redis):
        await redis_client.update_trending(mock_redis, topic_id=3)
        mock_redis.zincrby.assert_called_once_with("trending:now", 1, "3")

    async def test_update_sets_expire(self, mock_redis):
        await redis_client.update_trending(mock_redis, topic_id=5)
        mock_redis.expire.assert_called_once()

    async def test_get_trending_decodes(self, mock_redis):
        mock_redis.zrevrange.return_value = [
            (b"7", 42.0),
            (b"2", 18.0),
        ]
        result = await redis_client.get_trending(mock_redis, top_n=2)
        assert result == [("7", 42.0), ("2", 18.0)]

    async def test_get_trending_empty(self, mock_redis):
        mock_redis.zrevrange.return_value = []
        result = await redis_client.get_trending(mock_redis)
        assert result == []


# ------------------------------------------------------------------
# push_flagged / get_flagged
# ------------------------------------------------------------------

@pytest.mark.asyncio
class TestFlagged:
    async def test_push_flagged_calls_lpush(self, mock_redis):
        await redis_client.push_flagged(mock_redis, "at://did/post/1", "hate", "bad text")
        mock_redis.lpush.assert_called_once()
        payload = json.loads(mock_redis.lpush.call_args[0][1])
        assert payload["label"] == "hate"
        assert payload["uri"] == "at://did/post/1"

    async def test_push_flagged_trims_list(self, mock_redis):
        await redis_client.push_flagged(mock_redis, "uri", "spam", "text")
        mock_redis.ltrim.assert_called_with("flagged:recent", 0, redis_client._FLAGGED_MAX - 1)

    async def test_get_flagged_parses_json(self, mock_redis):
        item = json.dumps({"uri": "x", "label": "nsfw", "text": "..."}).encode()
        mock_redis.lrange.return_value = [item]
        result = await redis_client.get_flagged(mock_redis)
        assert len(result) == 1
        assert result[0]["label"] == "nsfw"

    async def test_get_flagged_skips_bad_json(self, mock_redis):
        mock_redis.lrange.return_value = [b"not json", b'{"label":"safe","uri":"x","text":"ok"}']
        result = await redis_client.get_flagged(mock_redis)
        assert len(result) == 1


# ------------------------------------------------------------------
# Counters
# ------------------------------------------------------------------

@pytest.mark.asyncio
class TestCounters:
    async def test_increment_counter(self, mock_redis):
        mock_redis.incrby.return_value = 42
        result = await redis_client.increment_counter(mock_redis, "counter:total")
        assert result == 42
        mock_redis.incrby.assert_called_with("counter:total", 1)

    async def test_get_counter_zero_when_missing(self, mock_redis):
        mock_redis.get.return_value = None
        result = await redis_client.get_counter(mock_redis, "counter:total")
        assert result == 0

    async def test_get_counter_returns_value(self, mock_redis):
        mock_redis.get.return_value = b"1337"
        result = await redis_client.get_counter(mock_redis, "counter:total")
        assert result == 1337


# ------------------------------------------------------------------
# Topic metadata
# ------------------------------------------------------------------

@pytest.mark.asyncio
class TestTopicMeta:
    async def test_set_topic_meta(self, mock_redis):
        await redis_client.set_topic_meta(mock_redis, 5, {"label": "news"})
        mock_redis.hset.assert_called_with("topic:meta:5", mapping={"label": "news"})

    async def test_get_topic_meta_decodes(self, mock_redis):
        mock_redis.hgetall.return_value = {b"label": b"politics"}
        result = await redis_client.get_topic_meta(mock_redis, 2)
        assert result == {"label": "politics"}

    async def test_append_topic_sample(self, mock_redis):
        mock_redis.hget.return_value = json.dumps(["old sample"]).encode()
        await redis_client.append_topic_sample(mock_redis, 3, "new sample text")
        mock_redis.hset.assert_called()
        # Verify the new sample is included
        stored = json.loads(mock_redis.hset.call_args[0][2])
        assert "new sample text" in stored
