# Real-Time Content Moderation & Safety Pipeline

End-to-end streaming ML system that ingests social content (BlueSky firehose or YouTube Shorts),
performs online topic clustering and LLM-based content moderation, and displays live metrics
in a Streamlit dashboard — all running locally on Apple Silicon.

```
BlueSky Jetstream  ──▶ ╮
 (WebSocket)            │
                        ├──▶  Redpanda (Kafka)  ──▶  Faust Agent  ──▶  Redis TimeSeries
YouTube Data API  ──▶ ╯       (bluesky.raw)         (embed +              │
 (polling, 30s)                                      moderate +        Streamlit
                                                     cluster)          Dashboard
```

## Architecture

| Layer | Technology | Purpose |
|---|---|---|
| Ingestion (BlueSky) | Jetstream WebSocket | Real-time post stream (~1–5k posts/min) |
| Ingestion (YouTube) | YouTube Data API v3 + polling | Recent Shorts, captions via transcript API |
| Message Bus | Redpanda (Kafka-compatible) | Durable, ordered event log |
| Stream Processing | Faust (async Python) | Per-event orchestration |
| Embedding | sentence-transformers `all-MiniLM-L6-v2` | 384-dim text vectors |
| Moderation | Ollama + `llama3.2:3b` (Metal GPU) | 5-class safety classification |
| Clustering | scikit-learn `MiniBatchKMeans` (n=20) | Online topic modeling |
| Storage | Redis Stack (TimeSeries + SortedSet) | Trend metrics, 1h retention |
| Dashboard | Streamlit + Plotly | Live charts, 5s auto-refresh |

### Why these choices

- **Redpanda** instead of Kafka — single binary, no ZooKeeper, identical API. Runs on a laptop without the JVM overhead.
- **Faust** instead of Spark Streaming — pure Python, async, fits naturally with `asyncio`-based Ollama client. Spark would be overkill for a single-machine workload.
- **MiniBatchKMeans** instead of full BERTopic — BERTopic's online variant (`merge_models`) requires buffering thousands of docs per merge step, adding minutes of latency. `partial_fit` achieves true per-batch updates with sub-millisecond overhead (Grootendorst, 2022).
- **llama3.2:3b via Ollama** — runs on Apple Metal without Docker. vLLM (Kwon et al., 2023 — PagedAttention) would be preferable for multi-GPU server deployments; Ollama achieves comparable single-request latency on M-series chips.
- **Redis TimeSeries** — native time-series aggregation (`TS.RANGE` with `AGGREGATION SUM`) eliminates the need for a separate TSDB. 1-hour retention keeps memory bounded.
- **YouTube Data API v3 + polling** — no open firehose exists for short-form video platforms. Polling `search.list` with `order=date` approximates a stream. Captions are fetched via `youtube-transcript-api`; title + description serves as fallback when captions are unavailable.

## Key Research References

1. **Grootendorst, M. (2022).** *BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure.* arXiv:2203.05794. — Basis for online topic modeling approach.
2. **Kwon et al. (2023).** *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). SOSP 2023. — Motivates high-throughput LLM inference design.
3. **Aggarwal et al. (2003).** *A Framework for Clustering Evolving Data Streams.* VLDB. — Theoretical basis for streaming cluster algorithms.

## Setup

### Prerequisites
- Docker Desktop
- [Ollama](https://ollama.ai) installed natively (**not** via Docker)
- Python 3.11+
- (YouTube only) A Google Cloud project with the **YouTube Data API v3** enabled and an API key

### 1. Clone and configure

```bash
git clone <repo-url>
cd <repo-name>
cp .env.example .env
```

For YouTube ingestion, add your API key to `.env`:

```
YOUTUBE_API_KEY=your_google_cloud_api_key_here
```

All other values in `.env.example` work out of the box.

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **macOS + Python 3.13 SSL fix** — if you installed Python from python.org and see
> `CERTIFICATE_VERIFY_FAILED` errors, run this once:
> ```bash
> open "/Applications/Python 3.13/Install Certificates.command"
> ```

### 4. Pull the LLM

```bash
ollama pull llama3.2:3b
```

### Run (each command in its own terminal tab)

> **Reminder:** activate the virtual environment first in each new terminal tab:
> ```bash
> source venv/bin/activate   # Windows: venv\Scripts\activate
> ```

```bash
# 1. Start infrastructure (Redpanda + Redis)
docker compose up -d

# 2. Start local LLM — skip if Ollama is already running
ollama serve

# 3a. Ingest from BlueSky firehose (real-time WebSocket)
python -m ingestion.bluesky_producer

# 3b. OR ingest from YouTube Shorts (polling, requires YOUTUBE_API_KEY)
python -m ingestion.youtube_producer

# 3c. OR run both simultaneously in separate tabs — both publish to the same Kafka topic

# 4. Stream processor (embed + moderate + cluster + store)
python -m faust -A processing.faust_app worker -l info

# 5. Dashboard — open http://localhost:8501
python -m streamlit run dashboard/app.py

# 6. Tests
python -m pytest tests/ -v
```

Wait ~2 minutes after starting for the topic clusterer to initialise (needs 100 posts for the first `partial_fit`).

### Service URLs

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| Redpanda Console | http://localhost:8080 |
| RedisInsight | http://localhost:8001 |
| Redpanda Kafka API | localhost:19092 |
| Redis | localhost:6379 |

## Ingestion Sources

### BlueSky (`ingestion/bluesky_producer.py`)

Connects to the [Jetstream WebSocket firehose](https://github.com/bluesky-social/jetstream) and streams all English-language posts in real time. No API key required. Expects roughly 1–5k posts/minute depending on global activity.

### YouTube Shorts (`ingestion/youtube_producer.py`)

Polls the YouTube Data API v3 every 30 seconds for recent Shorts matching a configurable search query. Captions are fetched via `youtube-transcript-api`; when unavailable the video title and description are used as the text signal.

| Env var | Default | Description |
|---|---|---|
| `YOUTUBE_API_KEY` | *(required)* | Google Cloud API key |
| `YOUTUBE_SEARCH_QUERY` | `#Shorts` | Keyword filter (e.g. `#Shorts news`) |
| `YOUTUBE_POLL_INTERVAL` | `30` | Seconds between polls |
| `YOUTUBE_MAX_RESULTS` | `10` | Videos per poll (max 50) |
| `TOPIC_YOUTUBE_RAW` | `bluesky.raw` | Kafka topic (shared with BlueSky by default) |

**API quota:** YouTube Data API provides 10,000 units/day free. Each `search.list` costs 100 units. At the default 30s interval this uses ~288,000 units/day — set `YOUTUBE_POLL_INTERVAL=300` to stay within the free tier.

Both producers publish records in the same schema so the Faust processor handles them identically. YouTube-specific fields (`source`, `video_id`, `channel_title`, `title`) are passed through to the moderated topic for downstream use.

## Project Structure

```
.
├── ingestion/
│   ├── bluesky_producer.py   # WebSocket firehose → Kafka
│   └── youtube_producer.py   # YouTube Shorts polling → Kafka
├── processing/
│   ├── faust_app.py          # Main stream agent (orchestrator)
│   ├── embedder.py           # sentence-transformers wrapper
│   ├── moderator.py          # Ollama async HTTP client
│   ├── topic_clusterer.py    # Online MiniBatchKMeans
│   └── redis_client.py       # TimeSeries / SortedSet helpers
├── dashboard/
│   └── app.py                # Streamlit live dashboard
├── tests/
│   ├── test_moderator.py     # respx mocks, label parsing, timeout fallback
│   ├── test_embedder.py      # shape, dtype, L2-norm checks
│   └── test_redis_client.py  # AsyncMock-based, no live Redis
├── models/                   # Saved MiniBatchKMeans checkpoints (.pkl)
├── .env.example              # Copy to .env before running
├── docker-compose.yml
└── requirements.txt
```

## Redis Key Schema

| Key | Type | Content |
|---|---|---|
| `trend:topic:{id}` | TimeSeries | Post volume per topic, 1h retention |
| `moderation:{label}` | TimeSeries | Post volume per label, 1h retention |
| `topic:meta:{id}` | Hash | Sample texts, cluster label |
| `trending:now` | Sorted Set | Topic scores in 15-min window |
| `flagged:recent` | List | Last 200 flagged post payloads |
| `counter:total` | Integer | Lifetime post count |
| `counter:flagged` | Integer | Lifetime flagged count |

## Moderation Labels

| Label | Description |
|---|---|
| `safe` | Normal discussion, news, personal updates |
| `spam` | Unsolicited ads, repetitive links, bot content |
| `hate` | Slurs, identity-based discrimination |
| `nsfw` | Sexual or explicit content |
| `violence` | Threats, graphic violence, self-harm encouragement |

Moderation uses temperature=0 for deterministic outputs. On timeout (>8s) or parse failure the system **fails open** (labels as `safe`) to maintain throughput — adjust `_FALLBACK_LABEL` in `moderator.py` for stricter pipelines.
