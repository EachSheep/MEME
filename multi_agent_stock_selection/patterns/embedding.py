"""
Embedding helpers for OpenRouter powered models.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Simple on-disk cache that uses the raw thesis content as key."""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._partitions: dict[str, dict[str, List[float]]] = {}
        self._dirty_partitions: set[str] = set()

    def _ensure_partition(self, partition: str) -> dict[str, List[float]]:
        if partition in self._partitions:
            return self._partitions[partition]
        storage: dict[str, List[float]] = {}
        if self.cache_dir:
            path = self._partition_path(partition)
            if path.exists():
                try:
                    with path.open("rb") as f:
                        storage = pickle.load(f)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to load embedding cache %s: %s", path, exc)
        self._partitions[partition] = storage
        return storage

    def _partition_path(self, partition: str) -> Path:
        assert self.cache_dir is not None
        sanitized = str(partition).replace("/", "_")
        return self.cache_dir / f"{sanitized}.pkl"

    def get(self, text: str, partition: Optional[str]) -> Optional[np.ndarray]:
        if not self.cache_dir or partition is None:
            return None
        storage = self._ensure_partition(partition)
        record = storage.get(text)
        if record is None:
            return None
        return np.asarray(record, dtype=np.float32)

    def set(self, text: str, embedding: Sequence[float], partition: Optional[str]) -> None:
        if not self.cache_dir or partition is None:
            return
        storage = self._ensure_partition(partition)
        storage[text] = list(map(float, embedding))
        self._dirty_partitions.add(partition)

    def flush(self) -> None:
        if not self.cache_dir or not self._dirty_partitions:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for partition in list(self._dirty_partitions):
            path = self._partition_path(partition)
            storage = self._partitions.get(partition, {})
            with path.open("wb") as f:
                pickle.dump(storage, f)
            self._dirty_partitions.discard(partition)


class OpenRouterEmbeddingClient:
    """Thin wrapper around OpenRouter embedding endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen3-embedding-8b",
        batch_size: int = 64,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        site_url: Optional[str] = None,
        site_title: Optional[str] = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model = model
        self.batch_size = max(1, batch_size)
        self.max_retries = max(1, max_retries)
        self.retry_backoff = max(0.5, retry_backoff)
        self.site_url = site_url
        self.site_title = site_title

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Return embeddings for all texts."""
        outputs: List[List[float]] = []
        headers = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_title:
            headers["X-Title"] = self.site_title

        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                        encoding_format="float",
                        extra_headers=headers or None,
                    )
                    outputs.extend([item.embedding for item in response.data])
                    break
                except Exception as exc:  # pragma: no cover - network errors
                    if attempt + 1 == self.max_retries:
                        raise
                    sleep_time = self.retry_backoff * (attempt + 1)
                    logger.warning(
                        "Embedding batch failed (%s), retrying in %.1fs (%d/%d)",
                        exc,
                        sleep_time,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(sleep_time)
        return outputs


class EmbeddingManager:
    """Coordinates caching and remote embedding generation."""

    def __init__(self, client: OpenRouterEmbeddingClient, cache: Optional[EmbeddingCache] = None) -> None:
        self.client = client
        self.cache = cache

    def embed(self, texts: Sequence[str], partition: Optional[str] = None) -> List[np.ndarray]:
        cached_embeddings: List[Optional[np.ndarray]] = [None] * len(texts)
        missing_indices: list[int] = []

        for idx, text in enumerate(texts):
            cached = self.cache.get(text, partition) if self.cache else None
            if cached is not None:
                cached_embeddings[idx] = cached
            else:
                missing_indices.append(idx)

        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            fresh_embeddings = self.client.embed_texts(missing_texts)
            for idx, emb in zip(missing_indices, fresh_embeddings):
                vector = np.asarray(emb, dtype=np.float32)
                cached_embeddings[idx] = vector
                if self.cache:
                    self.cache.set(texts[idx], vector.tolist(), partition)
            if self.cache:
                self.cache.flush()

        if any(embedding is None for embedding in cached_embeddings):
            missing = [i for i, emb in enumerate(cached_embeddings) if emb is None]
            raise RuntimeError(f"Missing embeddings for indices {missing}")

        return [embedding for embedding in cached_embeddings if embedding is not None]
