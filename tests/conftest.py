"""Gemeinsame pytest-Fixtures: vollständig offline (Hash-Embedder + In-Memory-Qdrant).

Kein Modell-Download, kein API-Key, kein laufender Server — die Suite testet
Pipeline-Logik und Verträge, nicht Modellqualität.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pytest

from rag_module import AdvancedRAGModule, RAGSettings
from rag_module.embeddings import BaseDenseEmbedder, BaseSparseEmbedder
from rag_module.models import SparseVector
from rag_module.utils import HeuristicTokenCounter, configure_token_counter
from rag_module.vector_store import QdrantVectorStore

_DIMENSION = 64


def _tokens(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _stable_hash(token: str) -> int:
    return int.from_bytes(hashlib.sha1(token.encode("utf-8")).digest()[:4], "big")


class HashDenseEmbedder(BaseDenseEmbedder):
    """Deterministischer Bag-of-Words-Embedder — ohne Modell, ohne Netz."""

    @property
    def dimension(self) -> int:
        return _DIMENSION

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * _DIMENSION
            for token in _tokens(text):
                vector[_stable_hash(token) % _DIMENSION] += 1.0
            norm = sum(component * component for component in vector) ** 0.5 or 1.0
            vectors.append([component / norm for component in vector])
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts)


class HashSparseEmbedder(BaseSparseEmbedder):
    """Deterministischer lexikalischer Sparse-Embedder auf Token-Hash-Basis."""

    async def _embed(self, texts: Sequence[str]) -> list[SparseVector]:
        vectors: list[SparseVector] = []
        for text in texts:
            weights: dict[int, float] = {}
            for token in _tokens(text):
                index = _stable_hash(token)
                weights[index] = weights.get(index, 0.0) + 1.0
            items = sorted(weights.items())
            vectors.append(
                SparseVector(
                    indices=[index for index, _ in items],
                    values=[value for _, value in items],
                )
            )
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return await self._embed(texts)

    async def embed_queries(self, texts: Sequence[str]) -> list[SparseVector]:
        return await self._embed(texts)


@pytest.fixture(autouse=True)
def _reset_token_counter() -> None:
    """Jeder Test startet mit der Heuristik (Modul-Global sauber halten)."""
    configure_token_counter(HeuristicTokenCounter())


@pytest.fixture
def offline_settings() -> RAGSettings:
    return RAGSettings(
        qdrant_url=":memory:",
        rerank_backend="none",
        expansion_backend="none",
        tokenizer_backend="heuristic",
    )


def build_offline_module(settings: RAGSettings) -> AdvancedRAGModule:
    return AdvancedRAGModule(
        settings=settings,
        vector_store=QdrantVectorStore(
            url=":memory:", collection_name=settings.collection_name
        ),
        dense_embedder=HashDenseEmbedder(),
        sparse_embedder=HashSparseEmbedder(),
        reranker=None,
    )


@pytest.fixture
async def offline_module(offline_settings: RAGSettings) -> AdvancedRAGModule:
    module = build_offline_module(offline_settings)
    yield module
    await module.close()
