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
