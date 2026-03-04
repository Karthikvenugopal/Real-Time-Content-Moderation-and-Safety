"""
Tests for processing.embedder

Validates output shape, dtype, and L2-normalisation.
The model is downloaded once and cached by sentence-transformers.
These tests do NOT require GPU or Ollama.
"""

from __future__ import annotations

import numpy as np
import pytest

from processing.embedder import embed, embed_one

EMBEDDING_DIM = 384


class TestEmbed:
    def test_single_text_shape(self):
        result = embed(["hello world"])
        assert result.shape == (1, EMBEDDING_DIM)

    def test_batch_shape(self):
        texts = ["hello", "world", "foo bar baz"]
        result = embed(texts)
        assert result.shape == (len(texts), EMBEDDING_DIM)

    def test_dtype_float32(self):
        result = embed(["test"])
        assert result.dtype == np.float32

    def test_l2_normalised(self):
        result = embed(["test sentence for normalisation check"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_empty_input_returns_empty(self):
        result = embed([])
        assert result.shape == (0, EMBEDDING_DIM)
        assert result.dtype == np.float32

    def test_different_texts_different_embeddings(self):
        a = embed(["I love cats"])
        b = embed(["The stock market crashed"])
        # Cosine similarity should be low for unrelated topics
        cos_sim = float(np.dot(a[0], b[0]))  # already unit-norm
        assert cos_sim < 0.9

    def test_similar_texts_high_similarity(self):
        a = embed(["The weather is sunny today"])
        b = embed(["Today's weather is very sunny"])
        cos_sim = float(np.dot(a[0], b[0]))
        assert cos_sim > 0.8

    def test_batch_consistency(self):
        """Embedding a batch vs individually should give the same results."""
        texts = ["apple", "banana", "cherry"]
        batch = embed(texts)
        individuals = np.vstack([embed([t]) for t in texts])
        np.testing.assert_allclose(batch, individuals, atol=1e-5)

    def test_long_text_does_not_crash(self):
        long_text = "word " * 500  # ~2500 tokens → truncated by model
        result = embed([long_text])
        assert result.shape == (1, EMBEDDING_DIM)


class TestEmbedOne:
    def test_shape(self):
        result = embed_one("hello")
        assert result.shape == (EMBEDDING_DIM,)

    def test_dtype(self):
        result = embed_one("hello")
        assert result.dtype == np.float32

    def test_l2_norm(self):
        result = embed_one("normalisation test")
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5
