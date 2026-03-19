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
