"""
Online Topic Clusterer — MiniBatchKMeans (scikit-learn)

Implements the "Online Clustering" paradigm for streaming text:
  - Maintains a rolling buffer of embeddings
  - Calls partial_fit() every BATCH_SIZE posts (incremental update)
  - Saves the model checkpoint to disk every CHECKPOINT_INTERVAL seconds
  - Exposes cluster centroids and per-post cluster assignments

Design reference:
  Grootendorst (2022) BERTopic — we replicate the core online-update
  mechanic without the full BERTopic stack to keep inference latency
  under 5 ms/post on CPU.
"""

from __future__ import annotations

import os
import pickle
import threading
import time
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.cluster import MiniBatchKMeans

N_CLUSTERS = 20
BATCH_SIZE = 100            # call partial_fit every N embeddings
CHECKPOINT_INTERVAL = 300   # seconds between .pkl saves
MODEL_PATH = Path("models/topic_clusterer.pkl")

_lock = threading.Lock()
