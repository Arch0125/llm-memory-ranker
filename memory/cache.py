"""Embedding cache.

A small wrapper that memoizes `embedder.embed_many` calls keyed by
`(model_name, content_sha1)`. Useful when the same texts (queries, key memory
strings) are embedded repeatedly within a session and across runs.

Two backends are provided:

- `InMemoryEmbeddingCache`: an LRU dict (default).
- `DiskEmbeddingCache`: optional persistent JSONL store, useful when iterating
  on benchmark runs and you want to keep embeddings between processes.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from typing import Sequence


def _content_key(model_name: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{model_name}:{digest}"


class InMemoryEmbeddingCache:
    """Thread-safe LRU cache. Set capacity=0 to disable eviction."""

    def __init__(self, capacity: int = 50000):
        self._capacity = capacity
        self._data: "OrderedDict[str, list[float]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, model_name: str, text: str):
        key = _content_key(model_name, text)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, model_name: str, text: str, vector: Sequence[float]) -> None:
        key = _content_key(model_name, text)
        with self._lock:
            self._data[key] = list(vector)
            self._data.move_to_end(key)
            if self._capacity and len(self._data) > self._capacity:
                self._data.popitem(last=False)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._data)}


class DiskEmbeddingCache(InMemoryEmbeddingCache):
    """Persistent variant. Loads/saves JSONL on disk; in-memory for hot path."""

    def __init__(self, path: str, capacity: int = 200000):
        super().__init__(capacity=capacity)
        self.path = path
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self._data[record["key"]] = record["vector"]
        except (OSError, ValueError):
            return
        if self._capacity:
            while len(self._data) > self._capacity:
                self._data.popitem(last=False)

    def put(self, model_name: str, text: str, vector: Sequence[float]) -> None:
        super().put(model_name, text, vector)
        key = _content_key(model_name, text)
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "vector": list(vector)}) + "\n")
        except OSError:
            pass


class CachedEmbedder:
    """Decorator: wraps an Embedder and serves repeated calls from a cache."""

    def __init__(self, embedder, cache: InMemoryEmbeddingCache | None = None):
        self.embedder = embedder
        self.cache = cache or InMemoryEmbeddingCache()

    @property
    def model_name(self):
        return self.embedder.model_name

    def embed(self, text):
        return self.embed_many([text])[0]

    def embed_many(self, texts):
        texts = list(texts)
        results: list[list[float] | None] = [None] * len(texts)
        misses = []
        miss_indices = []
        for i, text in enumerate(texts):
            cached = self.cache.get(self.embedder.model_name, text)
            if cached is None:
                misses.append(text)
                miss_indices.append(i)
            else:
                results[i] = cached
        if misses:
            new_vectors = self.embedder.embed_many(misses)
            for offset, vector in enumerate(new_vectors):
                idx = miss_indices[offset]
                results[idx] = list(vector)
                self.cache.put(self.embedder.model_name, misses[offset], vector)
        return results
