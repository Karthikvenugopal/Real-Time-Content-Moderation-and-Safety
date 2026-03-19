"""
YouTube Shorts → Redpanda (Kafka) Producer

Polls the YouTube Data API v3 for recent Shorts, fetches captions via
youtube-transcript-api, and publishes records to the same Kafka topic as
the BlueSky producer — so the existing Faust pipeline handles them unchanged.

Field mapping to existing schema:
  did           ← channel ID
  uri           ← https://www.youtube.com/shorts/{video_id}
  text          ← caption transcript (falls back to title + description)
  created_at    ← video publishedAt
  langs         ← ["en"]
  has_embed     ← True
  reply         ← False

Extra fields (passed through by faust_app.py, available for future use):
  source, video_id, channel_title, title

Env vars (add to .env):
  YOUTUBE_API_KEY        — Google Cloud API key (required)
  YOUTUBE_SEARCH_QUERY   — keyword filter, e.g. "#Shorts news" (default: "#Shorts")
  YOUTUBE_POLL_INTERVAL  — seconds between polls (default: 30)
  YOUTUBE_MAX_RESULTS    — results per poll, 1–50 (default: 10)
  TOPIC_YOUTUBE_RAW      — Kafka topic (default: same as TOPIC_BLUESKY_RAW → bluesky.raw)

Install extra deps:
  pip install google-api-python-client youtube-transcript-api

Run:
  python -m ingestion.youtube_producer
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

from aiokafka import AIOKafkaProducer
from dotenv import load_dotenv
from googleapiclient.discovery import build as yt_build
from googleapiclient.errors import HttpError
from loguru import logger
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
SEARCH_QUERY = os.getenv("YOUTUBE_SEARCH_QUERY", "#Shorts")
POLL_INTERVAL = int(os.getenv("YOUTUBE_POLL_INTERVAL", "30"))
MAX_RESULTS = min(int(os.getenv("YOUTUBE_MAX_RESULTS", "10")), 50)
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
# Default to same topic as BlueSky so faust_app.py needs no changes
KAFKA_TOPIC = os.getenv("TOPIC_YOUTUBE_RAW", os.getenv("TOPIC_BLUESKY_RAW", "bluesky.raw"))

# Deduplication: bounded in-memory set of seen video IDs
_SEEN_MAX = 2000
_seen_ids: set[str] = set()
_seen_order: list[str] = []  # FIFO for eviction

_stats = {"polls": 0, "published": 0, "skipped": 0, "errors": 0}


# ------------------------------------------------------------------
# Deduplication helpers
# ------------------------------------------------------------------

def _mark_seen(video_id: str) -> None:
    if video_id in _seen_ids:
        return
    _seen_ids.add(video_id)
    _seen_order.append(video_id)
    if len(_seen_order) > _SEEN_MAX:
        evict = _seen_order.pop(0)
        _seen_ids.discard(evict)


# ------------------------------------------------------------------
# Synchronous API calls (run in thread pool)
# ------------------------------------------------------------------

def _fetch_shorts(youtube, published_after: str | None) -> list[dict]:
    """Search YouTube for recent videos. Returns raw API items."""
    params: dict = {
        "part": "snippet",
        "type": "video",
        "order": "date",
        "maxResults": MAX_RESULTS,
    }
    if SEARCH_QUERY:
        params["q"] = SEARCH_QUERY
    if published_after:
        params["publishedAfter"] = published_after
    response = youtube.search().list(**params).execute()
    return response.get("items", [])


def _fetch_transcript(video_id: str) -> str | None:
    """Return joined caption text for a video, or None if unavailable."""
    try:
        segments = YouTubeTranscriptApi().fetch(
            video_id, languages=["en", "en-US", "en-GB"]
        )
        return " ".join(seg.text for seg in segments).strip() or None
    except (NoTranscriptFound, TranscriptsDisabled):
        return None
    except Exception as exc:
        logger.debug(f"Transcript fetch failed for {video_id}: {exc}")
        return None


def _build_record(item: dict, text: str) -> dict:
    """Map a YouTube API item to the pipeline's standard message schema."""
    snippet = item["snippet"]
    video_id = item["id"]["videoId"]
    return {
        # Standard fields consumed by faust_app.py
        "did": snippet.get("channelId", ""),
        "uri": f"https://www.youtube.com/watch?v={video_id}",
        "text": text,
        "created_at": snippet.get("publishedAt", datetime.now(timezone.utc).isoformat()),
        "langs": ["en"],
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "has_embed": True,
        "reply": False,
        # YouTube-specific extras (passed through, usable in future processing)
        "source": "youtube",
        "video_id": video_id,
        "channel_title": snippet.get("channelTitle", ""),
        "title": snippet.get("title", ""),
    }


# ------------------------------------------------------------------
# Async main loop
# ------------------------------------------------------------------

async def run() -> None:
    if not YOUTUBE_API_KEY:
        logger.error("YOUTUBE_API_KEY is not set. Add it to .env and retry.")
        sys.exit(1)

    loop = asyncio.get_event_loop()

    # Build the YouTube client once (sync; fetches discovery doc)
    youtube = await loop.run_in_executor(
        None, lambda: yt_build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    )

    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        compression_type="gzip",
        max_batch_size=65536,
        linger_ms=50,
    )
    await producer.start()

    logger.info(f"Publishing to Kafka topic '{KAFKA_TOPIC}' @ {KAFKA_BROKER}")
    logger.info(
        f"YouTube poller started | query='{SEARCH_QUERY}' | "
        f"max_results={MAX_RESULTS} | interval={POLL_INTERVAL}s"
    )

    published_after: str | None = None  # set to newest seen timestamp after first poll

    try:
        while True:
            t0 = time.monotonic()
            _stats["polls"] += 1
            batch_count = 0

            try:
                items: list[dict] = await loop.run_in_executor(
                    None, _fetch_shorts, youtube, published_after
                )
            except HttpError as exc:
                logger.error(f"YouTube API error (poll #{_stats['polls']}): {exc}")
                _stats["errors"] += 1
                await asyncio.sleep(POLL_INTERVAL)
                continue

            newest_ts: str | None = None

            for item in items:
                video_id = item["id"].get("videoId", "")
                if not video_id or video_id in _seen_ids:
                    _stats["skipped"] += 1
                    continue

                published_at = item["snippet"].get("publishedAt", "")
                if published_at and (newest_ts is None or published_at > newest_ts):
                    newest_ts = published_at

                # Captions (blocking I/O → thread pool)
                transcript = await loop.run_in_executor(None, _fetch_transcript, video_id)

                if transcript:
                    text = transcript
                else:
                    # Fall back to title + description so the video isn't silently dropped
                    snippet = item["snippet"]
                    text = f"{snippet.get('title', '')} {snippet.get('description', '')}".strip()
                    if not text:
                        _stats["skipped"] += 1
                        _mark_seen(video_id)
                        continue
                    logger.debug(f"No transcript for {video_id} — using title+description")

                record = _build_record(item, text)
                key = (record["did"] or video_id).encode()
                fut = await producer.send(KAFKA_TOPIC, key=key, value=json.dumps(record).encode())
                meta = await fut  # wait for broker ACK
                logger.debug(f"Published {video_id} → offset {meta.offset}")
                _mark_seen(video_id)
                _stats["published"] += 1
                batch_count += 1

            if newest_ts:
                published_after = newest_ts

            await producer.flush()
            elapsed = time.monotonic() - t0
            logger.info(
                f"[poll #{_stats['polls']}] new={batch_count}  "
                f"total={_stats['published']:,}  skipped={_stats['skipped']}  "
                f"errors={_stats['errors']}  elapsed={elapsed:.1f}s"
            )

            await asyncio.sleep(max(0.0, POLL_INTERVAL - elapsed))

    finally:
        await producer.stop()
        logger.info(
            f"Stopped. polls={_stats['polls']}  "
            f"published={_stats['published']:,}  errors={_stats['errors']}"
        )


def _handle_signal(*_) -> None:
    logger.info("Shutdown signal received")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    asyncio.run(run())
