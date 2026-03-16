"""
Real-Time Content Moderation Dashboard

Streamlit app that reads from Redis and renders live metrics.
Auto-refreshes every 5 seconds using streamlit-autorefresh.

Run:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import redis
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REFRESH_INTERVAL_MS = 5_000  # 5 s

st.set_page_config(
    page_title="Content Moderation Live",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Auto-refresh: re-run the script every REFRESH_INTERVAL_MS ms
# ------------------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    st_autorefresh(interval=REFRESH_INTERVAL_MS, key="refresh")
except ImportError:
    # Graceful fallback if streamlit-autorefresh is not installed
    pass


# ------------------------------------------------------------------
# Redis connection (cached so it survives re-runs)
# ------------------------------------------------------------------

@st.cache_resource
def get_redis() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def safe_ts_range(
    r: redis.Redis,
    key: str,
    from_ms: int,
    to_ms: int,
    agg: str = "SUM",
    bucket_ms: int = 10_000,
) -> list[tuple[int, float]]:
    try:
        result = r.execute_command(
            "TS.RANGE", key, from_ms, to_ms, "AGGREGATION", agg, bucket_ms
        )
        return [(int(ts), float(v)) for ts, v in result]
    except Exception:
        return []
