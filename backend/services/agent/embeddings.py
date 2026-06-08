"""Shared embedding service for agent features.

The first implementation reuses the existing ONNX runtime backend while keeping
knowledge-base callers behind a small, stable interface.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Iterable, List, Optional

from backend.services.agent import settings
from backend.services.agent.tool_selection import NoopEmbeddingBackend, OnnxEmbeddingBackend


class EmbeddingService:
    def __init__(self, backend: Any, *, timeout_ms: int, query_cache_size: int) -> None:
        self._backend = backend
        self._timeout_ms = timeout_ms
        self._query_cache_size = query_cache_size
        self._query_cache: OrderedDict[str, Optional[List[float]]] = OrderedDict()
        self._lock = threading.Lock()

    def is_ready(self) -> bool:
        return bool(getattr(self._backend, "is_ready", lambda: False)())

    def embed_query(self, text: str) -> Optional[List[float]]:
        payload = f"query: {str(text or '').strip()}"
        if not payload.strip():
            return None
        with self._lock:
            if payload in self._query_cache:
                value = self._query_cache.pop(payload)
                self._query_cache[payload] = value
                return value
        vector = self._to_list(self._backend.embed_query(payload, self._timeout_ms))
        with self._lock:
            if self._query_cache_size > 0:
                self._query_cache[payload] = vector
                while len(self._query_cache) > self._query_cache_size:
                    self._query_cache.popitem(last=False)
        return vector

    def embed_documents(self, texts: Iterable[str]) -> List[Optional[List[float]]]:
        payloads = [f"passage: {str(text or '').strip()}" for text in texts]
        if not payloads:
            return []
        matrix = self._backend.embed_documents(payloads)
        if matrix is None:
            return [None for _ in payloads]
        try:
            return [self._to_list(row) for row in matrix]
        except TypeError:
            return [None for _ in payloads]

    @staticmethod
    def _to_list(vector: Any) -> Optional[List[float]]:
        if vector is None:
            return None
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not isinstance(vector, (list, tuple)):
            return None
        values: List[float] = []
        for item in vector:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                return None
        return values


class _EmbeddingServiceManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._config_key = ""
        self._service = EmbeddingService(NoopEmbeddingBackend(), timeout_ms=1, query_cache_size=0)

    def get_service(self) -> EmbeddingService:
        enabled = settings.is_knowledge_base_embedding_enabled()
        backend_name = settings.get_knowledge_base_embedding_backend()
        model_path = settings.get_knowledge_base_embedding_model_path()
        timeout_ms = settings.get_knowledge_base_embedding_timeout_ms()
        cache_size = settings.get_knowledge_base_query_cache_size()
        config_key = f"{enabled}:{backend_name}:{model_path}:{timeout_ms}:{cache_size}"
        with self._lock:
            if config_key == self._config_key:
                return self._service
            backend = NoopEmbeddingBackend()
            if enabled and backend_name == "onnx":
                backend = OnnxEmbeddingBackend(model_path)
            self._service = EmbeddingService(backend, timeout_ms=timeout_ms, query_cache_size=cache_size)
            self._config_key = config_key
            return self._service


_manager = _EmbeddingServiceManager()


def get_embedding_service() -> EmbeddingService:
    return _manager.get_service()


def warm_knowledge_base_embedding_model() -> bool:
    """Preload the configured knowledge-base embedding model if it is available."""

    service = _manager.get_service()
    return service.is_ready()
