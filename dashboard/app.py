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


# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------

st.title("🛡️ Real-Time Content Moderation · BlueSky Stream")
st.caption(f"Last refreshed: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

r = get_redis()

# ── Row 1: KPI Cards ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

total = int(r.get("counter:total") or 0)
flagged = int(r.get("counter:flagged") or 0)
flag_rate = round(flagged / total * 100, 2) if total else 0.0

with col1:
    st.metric("Posts Processed", f"{total:,}")
with col2:
    st.metric("Posts Flagged", f"{flagged:,}")
with col3:
    st.metric("Flag Rate", f"{flag_rate}%")
with col4:
    trending_raw = r.zrevrange("trending:now", 0, 0, withscores=True)
    top_topic = f"Topic {trending_raw[0][0]}" if trending_raw else "—"
    st.metric("Hottest Topic", top_topic)

st.divider()

# ── Row 2: Moderation time-series + Label distribution ───────────
col_left, col_right = st.columns([2, 1])

LABELS = ["safe", "spam", "hate", "nsfw", "violence"]
LABEL_COLORS = {
    "safe": "#4CAF50",
    "spam": "#FF9800",
    "hate": "#F44336",
    "nsfw": "#9C27B0",
    "violence": "#E91E63",
}

now_ms = int(time.time() * 1000)
window_ms = now_ms - 60 * 60 * 1000  # last hour

with col_left:
    st.subheader("Moderation Events — Last Hour")
    ts_data: list[dict] = []
    for label in LABELS:
        pts = safe_ts_range(r, f"moderation:{label}", window_ms, now_ms, bucket_ms=30_000)
        for ts, val in pts:
            ts_data.append(
                {
                    "time": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                    "count": val,
                    "label": label,
                }
            )
    if ts_data:
        df_ts = pd.DataFrame(ts_data)
        fig_ts = px.line(
            df_ts,
            x="time",
            y="count",
            color="label",
            color_discrete_map=LABEL_COLORS,
            labels={"count": "Posts / 30 s", "time": ""},
        )
        fig_ts.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Waiting for data …")

with col_right:
    st.subheader("Label Distribution")
    label_counts: dict[str, int] = {}
    for label in LABELS:
        pts = safe_ts_range(r, f"moderation:{label}", window_ms, now_ms, agg="SUM", bucket_ms=3_600_000)
        label_counts[label] = int(pts[0][1]) if pts else 0

    if sum(label_counts.values()) > 0:
        fig_pie = px.pie(
            names=list(label_counts.keys()),
            values=list(label_counts.values()),
            color=list(label_counts.keys()),
            color_discrete_map=LABEL_COLORS,
            hole=0.4,
        )
        fig_pie.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Waiting for data …")

st.divider()
