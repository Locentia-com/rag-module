"""Embedding-Provider: dichte (Cohere v3, BGE-M3) und lexikalische (BM25, BGE-M3) Vektoren.

Alle Provider sind asynchron, batchen ihre Aufrufe, begrenzen die Parallelität
über Semaphores und nutzen Exponential-Backoff-Retries für transiente Fehler.
Optionale Abhängigkeiten (cohere, FlagEmbedding, fastembed) werden lazy geladen,
damit das Modul auch mit Teilinstallationen importierbar bleibt.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from .exceptions import ConfigurationError, EmbeddingError
from .models import SparseVector
from .utils import require_module, retry_async

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


class BaseDenseEmbedder(ABC):
    """Interface für dichte (semantische) Embedding-Provider."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionalität der erzeugten Vektoren."""

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embeddet Dokument-Chunks (Retrieval-Korpus-Seite)."""

    @abstractmethod
    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embeddet Suchanfragen (Query-Seite; kann asymmetrisch sein)."""

    async def close(self) -> None:  # noqa: B027 – optionaler Hook
        """Gibt Ressourcen frei (Default: nichts zu tun)."""


class BaseSparseEmbedder(ABC):
    """Interface für lexikalische Sparse-Embedding-Provider (BM25-artig)."""

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    @abstractmethod
    async def embed_queries(self, texts: Sequence[str]) -> list[SparseVector]: ...

    async def close(self) -> None:  # noqa: B027 – optionaler Hook
        """Gibt Ressourcen frei (Default: nichts zu tun)."""


# ---------------------------------------------------------------------------
# FastEmbed (dense, lokal via ONNX – Default-Backend)
# ---------------------------------------------------------------------------


def _model_description_get(description: Any, key: str) -> Any:
    """Liest ein Feld aus einer fastembed-Modellbeschreibung (dict oder Objekt)."""
    if isinstance(description, dict):
        return description.get(key)
    return getattr(description, key, None)


class FastEmbedDenseEmbedder(BaseDenseEmbedder):
    """Lokale Dense-Embeddings via fastembed (ONNX-Runtime, kein API-Key, kein Torch).

    Default-Modell: ``intfloat/multilingual-e5-large`` (1024 Dimensionen, stark
    multilingual). Für E5-Modelle werden die erforderlichen Präfixe
    ``passage:``/``query:`` automatisch ergänzt – fastembed selbst tut das nicht,
    und ohne Präfixe verlieren E5-Modelle deutlich an Retrieval-Qualität.

    Die Vektor-Dimension wird ohne Modell-Download aus der offiziellen
    fastembed-Modellliste ermittelt; für Modelle außerhalb der Liste muss
    ``dimension`` explizit angegeben werden.
    """

    def __init__(
        self,
        *,
        model_name: str = "intfloat/multilingual-e5-large",
        dimension: Optional[int] = None,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None
        self._lock = asyncio.Lock()
        self._needs_e5_prefix = "e5" in model_name.lower()
        self._dimension = dimension or self._lookup_dimension(model_name)

    @staticmethod
    def _lookup_dimension(model_name: str) -> int:
        fastembed = require_module(
            "fastembed", hint="Installation: pip install fastembed (Kern-Abhängigkeit)."
        )
        for description in fastembed.TextEmbedding.list_supported_models():
            if _model_description_get(description, "model") == model_name:
                return int(_model_description_get(description, "dim"))
        raise ConfigurationError(
            f"Dense-Modell '{model_name}' ist fastembed nicht bekannt. Entweder ein "
            "unterstütztes Modell wählen (fastembed.TextEmbedding.list_supported_models()) "
            "oder die Dimension explizit setzen (RAG_DENSE_DIMENSION_OVERRIDE)."
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is None:
                fastembed = require_module(
                    "fastembed", hint="Installation: pip install fastembed (Kern-Abhängigkeit)."
                )
                logger.info("Lade Dense-Modell '%s' (einmalig) …", self._model_name)
                self._model = await asyncio.to_thread(
                    fastembed.TextEmbedding, self._model_name
                )

    async def _embed(self, texts: Sequence[str], *, prefix: str) -> list[list[float]]:
        if not texts:
            return []
        await self._ensure_model()
        prepared = [
            (prefix + text) if self._needs_e5_prefix else text for text in texts
        ]
        try:
            vectors = await asyncio.to_thread(
                lambda: list(self._model.embed(prepared, batch_size=self._batch_size))
            )
        except Exception as exc:  # noqa: BLE001 – lokale Inferenzfehler kapseln
            raise EmbeddingError(f"FastEmbed-Dense-Encoding fehlgeschlagen: {exc}") from exc
        result = [[float(component) for component in vector] for vector in vectors]
        if result and len(result[0]) != self._dimension:
            raise EmbeddingError(
                f"Dense-Modell '{self._model_name}' liefert Dimension {len(result[0])}, "
                f"erwartet wurden {self._dimension}."
            )
        return result

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, prefix="passage: ")

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, prefix="query: ")


# ---------------------------------------------------------------------------
# Cohere v3 (dense, Andockstelle für externe API)
# ---------------------------------------------------------------------------

_COHERE_DIMENSIONS: dict[str, int] = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}

#: Cohere v3 hat ein 512-Token-Kontextfenster; längere Texte werden serverseitig
#: gekürzt. Wir kappen clientseitig großzügig, um Payload-Größen zu begrenzen.
_COHERE_MAX_CHARS = 8192


class CohereDenseEmbedder(BaseDenseEmbedder):
    """Dense Embeddings via Cohere-v3-API (asymmetrisch: search_document/search_query)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "embed-multilingual-v3.0",
        dimension: Optional[int] = None,
        batch_size: int = 96,
        concurrency: int = 4,
        timeout_s: float = 60.0,
        retry_attempts: int = 4,
        retry_base_delay_s: float = 0.5,
        retry_max_delay_s: float = 20.0,
    ) -> None:
        cohere = require_module("cohere", hint="Installation: pip install 'rag-module[cohere]'")
        self._client = cohere.AsyncClientV2(api_key=api_key, timeout=timeout_s)
        self._model = model
        self._dimension = dimension or _COHERE_DIMENSIONS.get(model, 1024)
        self._batch_size = min(batch_size, 96)  # API-Limit
        self._semaphore = asyncio.Semaphore(concurrency)
        self._retry_attempts = retry_attempts
        self._retry_base_delay_s = retry_base_delay_s
        self._retry_max_delay_s = retry_max_delay_s
        self._timeout_s = timeout_s

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, input_type="search_document")

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, input_type="search_query")

    async def _embed(self, texts: Sequence[str], *, input_type: str) -> list[list[float]]:
        if not texts:
            return []
        prepared = [text[:_COHERE_MAX_CHARS] if text.strip() else " " for text in texts]
        batches = [
            prepared[i : i + self._batch_size]
            for i in range(0, len(prepared), self._batch_size)
        ]
        results = await asyncio.gather(
            *(self._embed_batch(batch, input_type) for batch in batches)
        )
        return [vector for batch_result in results for vector in batch_result]

    async def _embed_batch(self, batch: list[str], input_type: str) -> list[list[float]]:
        async def call() -> Any:
            async with self._semaphore:
                return await self._client.embed(
                    model=self._model,
                    texts=batch,
                    input_type=input_type,
                    embedding_types=["float"],
                )

        try:
            response = await retry_async(
                call,
                op_name=f"cohere.embed({input_type})",
                attempts=self._retry_attempts,
                base_delay=self._retry_base_delay_s,
                max_delay=self._retry_max_delay_s,
                timeout=self._timeout_s,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Cohere-Embedding ({input_type}) nach allen Retries fehlgeschlagen: {exc}"
            ) from exc

        vectors = getattr(response.embeddings, "float_", None)
        if vectors is None:
            vectors = getattr(response.embeddings, "float", None)
        if vectors is None or len(vectors) != len(batch):
            raise EmbeddingError(
                "Cohere-Antwort enthält keine bzw. unvollständige Float-Embeddings."
            )
        if vectors and len(vectors[0]) != self._dimension:
            raise EmbeddingError(
                f"Cohere lieferte Dimension {len(vectors[0])}, konfiguriert ist "
                f"{self._dimension} – Modell/Dimension in den Settings prüfen."
            )
        return [list(map(float, vector)) for vector in vectors]


# ---------------------------------------------------------------------------
# BGE-M3 (dense + sparse, lokal)
# ---------------------------------------------------------------------------


class BGEM3Embedder:
    """Lokales BGE-M3-Modell: liefert dichte UND lexikalische Vektoren.

    Dies ist der geteilte Kern; die Interface-Implementierungen sind
    :class:`BGEM3DenseView` und :class:`BGEM3SparseView`, die beide auf
    dieselbe Instanz (und damit dasselbe geladene Modell) zeigen.

    Das Modell wird lazy geladen und Aufrufe werden serialisiert (das Modell
    ist nicht thread-sicher); die Inferenz selbst läuft in einem Worker-Thread.
    """

    def __init__(self, *, model_name: str = "BAAI/bge-m3", use_fp16: bool = True) -> None:
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._model: Any = None
        self._lock = asyncio.Lock()

    @property
    def dimension(self) -> int:
        return 1024

    async def _encode(
        self, texts: Sequence[str], *, dense: bool, sparse: bool
    ) -> dict[str, Any]:
        async with self._lock:
            if self._model is None:
                flag_embedding = require_module(
                    "FlagEmbedding", hint="Installation: pip install 'rag-module[bge]'"
                )
                logger.info("Lade BGE-M3-Modell '%s' (einmalig) …", self._model_name)
                self._model = await asyncio.to_thread(
                    flag_embedding.BGEM3FlagModel, self._model_name, use_fp16=self._use_fp16
                )
            try:
                return await asyncio.to_thread(
                    self._model.encode,
                    list(texts),
                    return_dense=dense,
                    return_sparse=sparse,
                    return_colbert_vecs=False,
                )
            except Exception as exc:  # noqa: BLE001 – lokale Inferenzfehler kapseln
                raise EmbeddingError(f"BGE-M3-Encoding fehlgeschlagen: {exc}") from exc

    async def embed_documents_dense(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        output = await self._encode(texts, dense=True, sparse=False)
        return [[float(x) for x in vector] for vector in output["dense_vecs"]]

    async def embed_documents_sparse(self, texts: Sequence[str]) -> list[SparseVector]:
        if not texts:
            return []
        output = await self._encode(texts, dense=False, sparse=True)
        return [self._to_sparse(weights) for weights in output["lexical_weights"]]

    @staticmethod
    def _to_sparse(lexical_weights: dict[str, float]) -> SparseVector:
        items = sorted(
            ((int(token_id), float(weight)) for token_id, weight in lexical_weights.items()),
            key=lambda pair: pair[0],
        )
        items = [(index, weight) for index, weight in items if weight > 0.0]
        return SparseVector(
            indices=[index for index, _ in items], values=[weight for _, weight in items]
        )


class BGEM3DenseView(BaseDenseEmbedder):
    """Adapter: präsentiert einen geteilten BGEM3Embedder als reinen Dense-Embedder."""

    def __init__(self, core: BGEM3Embedder) -> None:
        self._core = core

    @property
    def dimension(self) -> int:
        return self._core.dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._core.embed_documents_dense(texts)

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        # BGE-M3 ist symmetrisch: Queries und Dokumente teilen denselben Encoder.
        return await self._core.embed_documents_dense(texts)


class BGEM3SparseView(BaseSparseEmbedder):
    """Adapter: präsentiert einen geteilten BGEM3Embedder als reinen Sparse-Embedder."""

    def __init__(self, core: BGEM3Embedder) -> None:
        self._core = core

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return await self._core.embed_documents_sparse(texts)

    async def embed_queries(self, texts: Sequence[str]) -> list[SparseVector]:
        return await self._core.embed_documents_sparse(texts)


# ---------------------------------------------------------------------------
# FastEmbed BM25 (sparse, lokal)
# ---------------------------------------------------------------------------


class FastEmbedBM25SparseEmbedder(BaseSparseEmbedder):
    """Lexikalische BM25-Vektoren via fastembed (Modell ``Qdrant/bm25``).

    Kombiniert mit dem IDF-Modifier der Qdrant-Collection ergibt das eine
    vollwertige BM25-Suche über die Sparse-Vektoren.
    """

    def __init__(self, *, model_name: str = "Qdrant/bm25") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is None:
                fastembed = require_module(
                    "fastembed", hint="Installation: pip install 'rag-module[bm25]'"
                )
                logger.info("Lade BM25-Sparse-Modell '%s' (einmalig) …", self._model_name)
                self._model = await asyncio.to_thread(
                    fastembed.SparseTextEmbedding, self._model_name
                )

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return await self._embed(texts, is_query=False)

    async def embed_queries(self, texts: Sequence[str]) -> list[SparseVector]:
        return await self._embed(texts, is_query=True)

    async def _embed(self, texts: Sequence[str], *, is_query: bool) -> list[SparseVector]:
        if not texts:
            return []
        await self._ensure_model()
        try:
            if is_query:
                embeddings = await asyncio.to_thread(
                    lambda: list(self._model.query_embed(list(texts)))
                )
            else:
                embeddings = await asyncio.to_thread(
                    lambda: list(self._model.embed(list(texts), batch_size=64))
                )
        except Exception as exc:  # noqa: BLE001 – lokale Inferenzfehler kapseln
            raise EmbeddingError(f"BM25-Sparse-Encoding fehlgeschlagen: {exc}") from exc
        return [
            SparseVector(
                indices=[int(index) for index in embedding.indices],
                values=[float(value) for value in embedding.values],
            )
            for embedding in embeddings
        ]
