"""Retrieval abstractions for memory-aware hooks.

Provides:
  - Encoder Protocol: text → L2-normalized embedding vectors
  - Retriever Protocol: (query, corpus, k) → top-k (index, score) pairs
  - FakeEncoder: deterministic hash-based encoder for tests (no model download)
  - CosineRetriever: cosine-similarity baseline using any Encoder
  - SentenceTransformerEncoder: production encoder (BAAI/bge-m3); see below
"""
from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 matrix of L2-normalized embeddings."""
        ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, corpus: list[str], k: int) -> list[tuple[int, float]]:
        """Return top-k (corpus_index, score) sorted by score descending."""
        ...


class FakeEncoder:
    """Deterministic test encoder. Hashes each text to a fixed-dim vector via
    SHA-256 bytes. No model download, no network. Produces L2-normalized output."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            for j in range(self.dim):
                out[i, j] = (h[j % len(h)] - 128) / 128.0
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out / norms


class CosineRetriever:
    """Cosine-similarity retriever. Embeds corpus + query via the given encoder,
    computes dot products (L2-normalized → equivalent to cosine), and returns
    top-k results sorted by score descending."""

    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def retrieve(self, query: str, corpus: list[str], k: int) -> list[tuple[int, float]]:
        if not corpus:
            return []
        all_texts = [query] + list(corpus)
        embeddings = self.encoder.encode(all_texts)
        q_vec = embeddings[0]
        c_vecs = embeddings[1:]
        scores = c_vecs @ q_vec
        top_k = min(k, len(corpus))
        indices = np.argsort(-scores)[:top_k]
        return [(int(idx), float(scores[idx])) for idx in indices]


class SentenceTransformerEncoder:
    """Production encoder wrapping sentence-transformers. Lazy-loads the model
    on first encode() call. Caches embeddings by content hash (SHA-256) so
    identical texts are never re-encoded within the same instance lifetime.

    Default model: BAAI/bge-m3 (2.3 GB, strong on Chinese + English, consistent
    with legacy Memento-S semantic router).
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self._model_name = model_name
        self._model = None
        self._cache: dict[bytes, np.ndarray] = {}
        self._dim: int | None = None

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)
        probe = self._model.encode(["probe"], normalize_embeddings=True)
        self._dim = probe.shape[1]

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            self._load_model()

        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, t in enumerate(texts):
            key = hashlib.sha256(t.encode("utf-8")).digest()
            if key in self._cache:
                out[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            vecs = self._model.encode(uncached_texts, normalize_embeddings=True)
            if not isinstance(vecs, np.ndarray):
                vecs = np.array(vecs, dtype=np.float32)
            for j, idx in enumerate(uncached_indices):
                key = hashlib.sha256(uncached_texts[j].encode("utf-8")).digest()
                self._cache[key] = vecs[j]
                out[idx] = vecs[j]

        return out
