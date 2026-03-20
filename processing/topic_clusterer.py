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


class TopicClusterer:
    """Thread-safe online topic clusterer."""

    def __init__(
        self,
        n_clusters: int = N_CLUSTERS,
        batch_size: int = BATCH_SIZE,
        checkpoint_path: Path = MODEL_PATH,
    ) -> None:
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

        self._kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            n_init=3,
            random_state=42,
            compute_labels=True,
        )
        self._buffer: list[np.ndarray] = []
        self._initialized = False
        self._last_checkpoint = time.time()
        self._total_seen = 0

        # Restore from checkpoint if available
        self._load_checkpoint()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, embedding: np.ndarray) -> int | None:
        """
        Add a single embedding (shape 384,) to the buffer.
        Triggers a partial_fit when the buffer is full.

        Returns the cluster ID once the model is initialised,
        otherwise returns None.
        """
        with _lock:
            self._buffer.append(embedding.astype(np.float64))
            self._total_seen += 1

            if len(self._buffer) >= self.batch_size:
                self._flush()

            if self._initialized:
                return int(self._kmeans.predict([embedding.astype(np.float64)])[0])
            return None

    def predict(self, embedding: np.ndarray) -> int | None:
        """Predict cluster for a single embedding without buffering."""
        with _lock:
            if not self._initialized:
                return None
            return int(self._kmeans.predict([embedding.astype(np.float64)])[0])

    def get_centroids(self) -> np.ndarray | None:
        """Return cluster centroids, shape (n_clusters, 384), or None."""
        with _lock:
            if not self._initialized:
                return None
            return self._kmeans.cluster_centers_.copy()

    @property
    def total_seen(self) -> int:
        return self._total_seen

    @property
    def is_ready(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Call partial_fit on the current buffer and clear it."""
        if not self._buffer:
            return

        X = np.vstack(self._buffer)
        self._buffer.clear()

        try:
            self._kmeans.partial_fit(X)
            if not self._initialized:
                logger.info(
                    f"Topic clusterer initialised with first {len(X)} embeddings"
                )
            self._initialized = True
        except Exception as exc:
            logger.error(f"partial_fit error: {exc}")
            return

        now = time.time()
        if now - self._last_checkpoint >= CHECKPOINT_INTERVAL:
            self._save_checkpoint()
            self._last_checkpoint = now

    def _save_checkpoint(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(self._kmeans, f)
            tmp.replace(self.checkpoint_path)
            logger.info(
                f"Checkpoint saved → {self.checkpoint_path}  "
                f"(total_seen={self._total_seen:,})"
            )
        except Exception as exc:
            logger.error(f"Checkpoint save failed: {exc}")

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        try:
            with open(self.checkpoint_path, "rb") as f:
                self._kmeans = pickle.load(f)
            self._initialized = True
            logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
        except Exception as exc:
            logger.warning(f"Could not load checkpoint: {exc}")


# Module-level singleton shared across Faust workers in the same process
_clusterer: TopicClusterer | None = None


def get_clusterer() -> TopicClusterer:
    global _clusterer
    if _clusterer is None:
        _clusterer = TopicClusterer()
    return _clusterer
