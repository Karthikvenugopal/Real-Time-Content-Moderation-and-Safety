"""
BlueSky Jetstream → Redpanda (Kafka) Producer

Connects to the BlueSky Jetstream WebSocket firehose and publishes
app.bsky.feed.post events to the `bluesky.raw` Kafka topic.

BlueSky Jetstream docs:
  https://github.com/bluesky-social/jetstream
"""

import asyncio
import json
import os
import signal
import ssl
import sys
from datetime import datetime, timezone

import certifi
import websockets
from aiokafka import AIOKafkaProducer
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

JETSTREAM_URL = os.getenv(
    "BLUESKY_JETSTREAM_URL",
    "wss://jetstream2.us-east.bsky.network/subscribe",
)
# Only subscribe to feed post events to reduce volume
JETSTREAM_URL_FULL = (
    f"{JETSTREAM_URL}?wantedCollections=app.bsky.feed.post"
)
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
KAFKA_TOPIC = os.getenv("TOPIC_BLUESKY_RAW", "bluesky.raw")

# Stats
_stats = {"received": 0, "published": 0, "errors": 0}
