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


def _extract_post(event: dict) -> dict | None:
    """
    Parse a Jetstream event into a flat post record.
    Returns None for events that are not text posts.
    """
    try:
        if event.get("kind") != "commit":
            return None
        commit = event.get("commit", {})
        if commit.get("collection") != "app.bsky.feed.post":
            return None
        if commit.get("operation") not in ("create", "update"):
            return None

        record = commit.get("record", {})
        text = record.get("text", "").strip()
        if not text:
            return None

        langs = record.get("langs", ["und"])
        # Skip non-English posts to keep moderation focused
        if langs and not any(l.startswith("en") for l in langs):
            return None

        return {
            "did": event.get("did", ""),
            "uri": f"at://{event.get('did')}/app.bsky.feed.post/{commit.get('rkey', '')}",
            "text": text,
            "created_at": record.get("createdAt", datetime.now(timezone.utc).isoformat()),
            "langs": langs,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "has_embed": "embed" in record,
            "reply": record.get("reply") is not None,
        }
    except Exception as exc:
        logger.debug(f"Parse error: {exc}")
        return None


async def _produce(producer: AIOKafkaProducer, post: dict) -> None:
    key = post["did"].encode()
    value = json.dumps(post).encode()
    await producer.send(KAFKA_TOPIC, key=key, value=value)
    _stats["published"] += 1


async def run() -> None:
    logger.info(f"Connecting to {JETSTREAM_URL_FULL}")
    logger.info(f"Publishing to Kafka topic '{KAFKA_TOPIC}' @ {KAFKA_BROKER}")

    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        compression_type="gzip",
        max_batch_size=65536,
        linger_ms=50,
    )
    await producer.start()
    logger.info("Kafka producer started")

    reconnect_delay = 1.0

    try:
        while True:
            try:
                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
                async with websockets.connect(
                    JETSTREAM_URL_FULL,
                    ssl=ssl_ctx,
                    ping_interval=20,
                    ping_timeout=30,
                    max_size=2**20,  # 1 MB
                ) as ws:
                    logger.info("WebSocket connected to BlueSky Jetstream")
                    reconnect_delay = 1.0  # reset on successful connect

                    async for raw_msg in ws:
                        try:
                            event = json.loads(raw_msg)
                            _stats["received"] += 1

                            post = _extract_post(event)
                            if post:
                                await _produce(producer, post)

                            if _stats["received"] % 1000 == 0:
                                logger.info(
                                    f"recv={_stats['received']:,}  "
                                    f"published={_stats['published']:,}  "
                                    f"errors={_stats['errors']}"
                                )
                        except json.JSONDecodeError:
                            _stats["errors"] += 1
                        except Exception as exc:
                            _stats["errors"] += 1
                            logger.warning(f"Event error: {exc}")

            except websockets.ConnectionClosed as exc:
                logger.warning(f"WebSocket closed ({exc}), reconnecting in {reconnect_delay}s")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)
            except OSError as exc:
                logger.error(f"Connection error: {exc}, retrying in {reconnect_delay}s")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)
    finally:
        await producer.stop()
        logger.info(
            f"Producer stopped. Final stats: recv={_stats['received']:,}  "
            f"published={_stats['published']:,}  errors={_stats['errors']}"
        )


def _handle_signal(*_) -> None:
    logger.info("Shutdown signal received")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    asyncio.run(run())
