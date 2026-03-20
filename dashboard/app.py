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
import redis
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REFRESH_INTERVAL_MS = 5_000

st.set_page_config(
    page_title="Content Moderation Live",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    st_autorefresh(interval=REFRESH_INTERVAL_MS, key="refresh")
except ImportError:
    pass


# ------------------------------------------------------------------
# Redis connection
# ------------------------------------------------------------------

@st.cache_resource
def get_redis() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

LABELS = ["safe", "spam", "hate", "nsfw", "violence"]
LABEL_COLORS = {
    "safe": "#4CAF50",
    "spam": "#FF9800",
    "hate": "#F44336",
    "nsfw": "#9C27B0",
    "violence": "#E91E63",
}
SOURCE_COLORS = {"bluesky": "#0085FF", "youtube": "#FF0000"}


def _k(src: str | None, *parts: str) -> str:
    """Build a Redis key with optional source namespace.

    _k(None, "moderation", "safe")      → "moderation:safe"
    _k("bluesky", "moderation", "safe") → "moderation:bluesky:safe"
    """
    if src:
        return ":".join([parts[0], src] + list(parts[1:]))
    return ":".join(parts)


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


def _get_topic_samples(r: redis.Redis, topic_id: str) -> str:
    try:
        raw = r.hget(f"topic:meta:{topic_id}", "samples")
        if raw:
            samples = json.loads(raw)
            return " · ".join(s[:40] for s in samples[:2])
    except Exception:
        pass
    return ""


# ------------------------------------------------------------------
# Per-source dashboard renderer
# ------------------------------------------------------------------

def render_source(r: redis.Redis, src: str | None) -> None:
    """Render the full dashboard for one source (None = all combined)."""

    now_ms = int(time.time() * 1000)
    window_ms = now_ms - 60 * 60 * 1000  # last hour

    # ── KPI Cards ─────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    total = int(r.get(_k(src, "counter", "total")) or 0)
    flagged = int(r.get(_k(src, "counter", "flagged")) or 0)
    flag_rate = round(flagged / total * 100, 2) if total else 0.0

    with col1:
        st.metric("Posts Processed", f"{total:,}")
    with col2:
        st.metric("Posts Flagged", f"{flagged:,}")
    with col3:
        st.metric("Flag Rate", f"{flag_rate}%")
    with col4:
        trending_raw = r.zrevrange(_k(src, "trending", "now"), 0, 0, withscores=True)
        top_topic = f"Topic {trending_raw[0][0]}" if trending_raw else "—"
        st.metric("Hottest Topic", top_topic)

    st.divider()

    # ── Moderation time-series + Label distribution ───────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Moderation Events — Last Hour")
        ts_data: list[dict] = []
        for label in LABELS:
            pts = safe_ts_range(
                r, _k(src, "moderation", label), window_ms, now_ms, bucket_ms=30_000
            )
            for ts, val in pts:
                ts_data.append({
                    "time": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                    "count": val,
                    "label": label,
                })
        if ts_data:
            df_ts = pd.DataFrame(ts_data)
            fig_ts = px.line(
                df_ts, x="time", y="count", color="label",
                color_discrete_map=LABEL_COLORS,
                labels={"count": "Posts / 30 s", "time": ""},
            )
            fig_ts.update_layout(
                height=320, margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_ts, use_container_width=True, key=f"{src}_ts")
        else:
            st.info("Waiting for data …")

    with col_right:
        st.subheader("Label Distribution")
        label_counts: dict[str, int] = {}
        for label in LABELS:
            pts = safe_ts_range(
                r, _k(src, "moderation", label), window_ms, now_ms,
                agg="SUM", bucket_ms=3_600_000,
            )
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
                height=320, margin=dict(l=0, r=0, t=0, b=0), showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True, key=f"{src}_pie")
        else:
            st.info("Waiting for data …")

    st.divider()

    # ── Trending Topics + Topic volume heatmap ────────────────────
    col_trend, col_heat = st.columns([1, 2])

    with col_trend:
        st.subheader("Trending Topics (15 min)")
        trending = r.zrevrange(_k(src, "trending", "now"), 0, 9, withscores=True)
        if trending:
            trend_df = pd.DataFrame([
                {
                    "Topic": f"Topic {tid}",
                    "Posts": int(score),
                    "Samples": _get_topic_samples(r, tid),
                }
                for tid, score in trending
            ])
            fig_bar = px.bar(
                trend_df, x="Posts", y="Topic", orientation="h",
                color="Posts", color_continuous_scale="Viridis",
            )
            fig_bar.update_layout(
                height=340, margin=dict(l=0, r=0, t=0, b=0),
                yaxis=dict(autorange="reversed"),
                showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True, key=f"{src}_bar")
        else:
            st.info("Waiting for topics to form …")

    with col_heat:
        st.subheader("Topic Activity Heatmap — Last Hour")
        heat_data: list[dict] = []
        for topic_id in range(20):
            pts = safe_ts_range(
                r, _k(src, "trend", f"topic:{topic_id}"),
                window_ms, now_ms, bucket_ms=300_000,
            )
            for ts, val in pts:
                heat_data.append({
                    "time": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%H:%M"),
                    "topic": f"T{topic_id:02d}",
                    "count": val,
                })
        if heat_data:
            df_heat = pd.DataFrame(heat_data)
            pivot = df_heat.pivot_table(
                index="topic", columns="time", values="count", aggfunc="sum"
            ).fillna(0)
            fig_heat = px.imshow(
                pivot, color_continuous_scale="YlOrRd",
                labels=dict(x="Time (UTC)", y="Topic", color="Posts"),
                aspect="auto",
            )
            fig_heat.update_layout(height=340, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_heat, use_container_width=True, key=f"{src}_heat")
        else:
            st.info("Waiting for topic data …")

    st.divider()

    # ── Flagged Posts Feed ────────────────────────────────────────
    st.subheader("🚨 Recent Flagged Posts")
    flagged_raw = r.lrange(_k(src, "flagged", "recent"), 0, 19)
    if flagged_raw:
        flagged_posts = []
        for item in flagged_raw:
            try:
                flagged_posts.append(json.loads(item))
            except Exception:
                pass

        for post in flagged_posts:
            badge_color = LABEL_COLORS.get(post.get("label", "safe"), "#999")
            post_source = post.get("source", "bluesky")
            src_color = SOURCE_COLORS.get(post_source, "#999")
            with st.container():
                cols = st.columns([1, 1, 9]) if src is None else st.columns([1, 10])
                with cols[0]:
                    st.markdown(
                        f"<span style='background:{badge_color};color:white;"
                        f"padding:3px 8px;border-radius:4px;font-size:12px'>"
                        f"{post.get('label','?').upper()}</span>",
                        unsafe_allow_html=True,
                    )
                if src is None:
                    with cols[1]:
                        st.markdown(
                            f"<span style='background:{src_color};color:white;"
                            f"padding:3px 8px;border-radius:4px;font-size:12px'>"
                            f"{post_source.upper()}</span>",
                            unsafe_allow_html=True,
                        )
                with cols[-1]:
                    st.text(post.get("text", "")[:200])
    else:
        st.info("No flagged posts yet.")


# ------------------------------------------------------------------
# Page header + tabs
# ------------------------------------------------------------------

st.title("🛡️ Real-Time Content Moderation")
st.caption(f"Last refreshed: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

r = get_redis()

tab_all, tab_bluesky, tab_youtube = st.tabs(["All Sources", "🦋 BlueSky", "▶️ YouTube"])

with tab_all:
    render_source(r, src=None)

with tab_bluesky:
    render_source(r, src="bluesky")

with tab_youtube:
    render_source(r, src="youtube")
