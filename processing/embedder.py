"""
Text Embedder — sentence-transformers all-MiniLM-L6-v2

Produces L2-normalised float32 embeddings of shape (n, 384).
The model is loaded once at import time and reused across calls
(thread-safe for inference).
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model '{_MODEL_NAME}' …")
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Embedding model loaded")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings.

    Returns
    -------
    np.ndarray
        Shape (len(texts), 384), dtype float32, L2-normalised.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = _get_model()
    # normalize_embeddings=True → each row has unit L2 norm
    vectors = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_one(text: str) -> np.ndarray:
    """Convenience wrapper for a single string. Returns shape (384,)."""
    return embed([text])[0]
