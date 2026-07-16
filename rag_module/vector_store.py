"""Vektordatenbank-Abstraktion (Repository Pattern) mit Qdrant-Standard-Implementierung.

Die Qdrant-Implementierung nutzt benannte Vektoren (``dense`` + ``sparse`` mit
IDF-Modifier für BM25-Semantik), legt Payload-Indizes für alle Filterfelder an
und übersetzt :class:`~rag_module.models.TemporalFilter` sowie beliebige
Metadaten-Filter in native Qdrant-Filter.

Alle Netzwerk-Operationen laufen über Exponential-Backoff-Retries; transiente
Fehler werden nach Erschöpfung der Retries als
:class:`~rag_module.exceptions.VectorStoreConnectionError` gemeldet,
nicht-transiente als :class:`~rag_module.exceptions.VectorStoreError`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from .exceptions import (
    ConfigurationError,
    VectorStoreConnectionError,
    VectorStoreError,
)
from .models import ScoredChunk, SparseVector, TemporalFilter
from .utils import is_retryable_error, retry_async

try:  # Optionaler Import: Modul bleibt ohne qdrant-client importierbar.
    from qdrant_client import AsyncQdrantClient
    from qdrant_client import models as qmodels
except ImportError:  # pragma: no cover - nur bei Teilinstallation relevant
    AsyncQdrantClient = None  # type: ignore[assignment, misc]
    qmodels = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

T = TypeVar("T")

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


@dataclass(slots=True)
class ChunkPoint:
    """Ein upsert-fertiger Punkt: ID, Vektoren und vollständiger Payload."""

    point_id: str
    dense: list[float]
    sparse: SparseVector | None
    payload: dict[str, Any]


class BaseVectorStore(ABC):
    """Repository-Interface der Vektordatenbank.

    Bewusst schmal gehalten: Die Retrieval-Pipeline erhält getrennte
    Dense-/Sparse-Suchprimitiva und übernimmt die Fusion (RRF) selbst –
    so bleibt die Fusion implementierungsunabhängig testbar.
    """

    @abstractmethod
    async def ensure_ready(self, dense_dimension: int) -> None:
        """Erstellt Collection + Indizes idempotent; validiert die Vektor-Dimension."""

    @abstractmethod
    async def upsert(self, points: Sequence[ChunkPoint]) -> None:
        """Schreibt Punkte in Batches in die Datenbank."""

    @abstractmethod
    async def search_dense(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        temporal: TemporalFilter | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Semantische Suche über den Dense-Vektor."""

    @abstractmethod
    async def search_sparse(
        self,
        vector: SparseVector,
        *,
        limit: int,
        temporal: TemporalFilter | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Lexikalische Suche (BM25) über den Sparse-Vektor."""

    @abstractmethod
    async def fetch_by_ids(self, point_ids: Sequence[str]) -> list[ScoredChunk]:
        """Lädt Punkte (z. B. Parent-Chunks) direkt über ihre IDs."""

    @abstractmethod
    async def get_latest_version(self, document_id: str) -> int | None:
        """Höchste existierende Version eines Dokuments (aktiv oder historisch)."""

    @abstractmethod
    async def deactivate_document(
        self,
        document_id: str,
        *,
        valid_to: float,
        valid_to_iso: str,
        before_version: int | None = None,
    ) -> int:
        """Setzt alle aktiven Punkte eines Dokuments auf ``is_active=False``.

        ``before_version`` begrenzt die Deaktivierung auf ältere Versionen –
        damit ist die Operation auch nach einem teilweisen Ingest idempotent.
        Rückgabe: Anzahl deaktivierter Punkte.
        """

    @abstractmethod
    async def get_document_versions(self, document_id: str) -> list[dict[str, Any]]:
        """Versionshistorie eines Dokuments (Version, Gültigkeit, Chunk-Anzahl)."""

    @abstractmethod
    async def close(self) -> None:
        """Schließt Verbindungen."""


# ---------------------------------------------------------------------------
# Qdrant-Implementierung
# ---------------------------------------------------------------------------


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-Repository mit Hybrid-Vektoren (dense + sparse/IDF) und Temporal-Payload.

    ``url=":memory:"`` startet eine eingebettete In-Process-Instanz –
    nützlich für Tests und lokale Entwicklung ohne laufenden Qdrant-Server.
    """

    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        upsert_batch_size: int = 128,
        retry_attempts: int = 4,
        retry_base_delay_s: float = 0.5,
        retry_max_delay_s: float = 20.0,
    ) -> None:
        if AsyncQdrantClient is None:
            raise ConfigurationError(
                "Das Paket 'qdrant-client' ist nicht installiert "
                "(Installation: pip install qdrant-client)."
            )
        self._is_local_mode = url == ":memory:"
        if self._is_local_mode:
            self._client = AsyncQdrantClient(location=":memory:")
        else:
            self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=int(timeout_s))
        self._collection = collection_name
        self._upsert_batch_size = upsert_batch_size
        self._retry_attempts = retry_attempts
        self._retry_base_delay_s = retry_base_delay_s
        self._retry_max_delay_s = retry_max_delay_s
        self._ready = False

    # -- Infrastruktur ---------------------------------------------------------

    async def _call(self, op_name: str, factory: Callable[[], Awaitable[T]]) -> T:
        """Führt eine Qdrant-Operation mit Retries aus und übersetzt Fehler."""
        try:
            return await retry_async(
                factory,
                op_name=f"qdrant.{op_name}",
                attempts=self._retry_attempts,
                base_delay=self._retry_base_delay_s,
                max_delay=self._retry_max_delay_s,
            )
        except (VectorStoreError, ConfigurationError):
            raise
        except Exception as exc:  # noqa: BLE001 – gezielte Übersetzung in Modul-Fehler
            error_cls = (
                VectorStoreConnectionError if is_retryable_error(exc) else VectorStoreError
            )
            raise error_cls(f"Qdrant-Operation '{op_name}' fehlgeschlagen: {exc}") from exc

    async def ensure_ready(self, dense_dimension: int) -> None:
        if self._ready:
            return
        exists = await self._call(
            "collection_exists", lambda: self._client.collection_exists(self._collection)
        )
        if not exists:
            await self._call(
                "create_collection",
                lambda: self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config={
                        DENSE_VECTOR_NAME: qmodels.VectorParams(
                            size=dense_dimension, distance=qmodels.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                            modifier=qmodels.Modifier.IDF
                        )
                    },
                ),
            )
            logger.info(
                "Qdrant-Collection '%s' angelegt (dense=%d/cosine, sparse=IDF).",
                self._collection,
                dense_dimension,
            )
        else:
            info = await self._call(
                "get_collection", lambda: self._client.get_collection(self._collection)
            )
            vectors_config = info.config.params.vectors
            dense_config = (
                vectors_config.get(DENSE_VECTOR_NAME)
                if isinstance(vectors_config, dict)
                else None
            )
            if dense_config is not None and dense_config.size != dense_dimension:
                raise ConfigurationError(
                    f"Collection '{self._collection}' hat Dense-Dimension "
                    f"{dense_config.size}, der Embedder liefert {dense_dimension}. "
                    "Collection migrieren oder anderes Embedding-Modell konfigurieren."
                )

        if self._is_local_mode:
            # Der eingebettete Lokalmodus unterstützt keine Payload-Indizes
            # (Filter funktionieren dort auch ohne).
            self._ready = True
            return

        index_fields: list[tuple[str, Any]] = [
            ("document_id", qmodels.PayloadSchemaType.KEYWORD),
            ("document_type", qmodels.PayloadSchemaType.KEYWORD),
            ("chunk_type", qmodels.PayloadSchemaType.KEYWORD),
            ("chunk_role", qmodels.PayloadSchemaType.KEYWORD),
            ("is_active", qmodels.PayloadSchemaType.BOOL),
            ("searchable", qmodels.PayloadSchemaType.BOOL),
            ("version", qmodels.PayloadSchemaType.INTEGER),
            ("valid_from", qmodels.PayloadSchemaType.FLOAT),
            ("valid_to", qmodels.PayloadSchemaType.FLOAT),
        ]
        for field_name, schema in index_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as exc:  # noqa: BLE001 – Index existiert ggf. bereits
                logger.debug("Payload-Index '%s' nicht (neu) angelegt: %s", field_name, exc)
        self._ready = True

    # -- Filter-Übersetzung ------------------------------------------------------

    @staticmethod
    def _metadata_condition(key: str, value: Any) -> Any:
        field_key = f"meta.{key}"
        if isinstance(value, bool) or isinstance(value, (int, str)):
            return qmodels.FieldCondition(key=field_key, match=qmodels.MatchValue(value=value))
        if isinstance(value, float):
            return qmodels.FieldCondition(
                key=field_key, range=qmodels.Range(gte=value, lte=value)
            )
        if isinstance(value, (list, tuple)):
            return qmodels.FieldCondition(
                key=field_key, match=qmodels.MatchAny(any=list(value))
            )
        if isinstance(value, dict) and value and set(value) <= {"gt", "gte", "lt", "lte"}:
            return qmodels.FieldCondition(key=field_key, range=qmodels.Range(**value))
        raise VectorStoreError(
            f"Metadaten-Filter für '{key}' hat einen nicht unterstützten Wert-Typ: "
            f"{type(value).__name__}. Erlaubt: str, int, bool, float, Liste, "
            "oder Range-Dict mit gt/gte/lt/lte."
        )

    def _build_filter(
        self,
        *,
        temporal: TemporalFilter | None,
        metadata_filter: dict[str, Any] | None,
        document_id: str | None = None,
        only_searchable: bool = False,
        only_active: bool | None = None,
    ) -> Any | None:
        must: list[Any] = []
        if only_searchable:
            must.append(
                qmodels.FieldCondition(key="searchable", match=qmodels.MatchValue(value=True))
            )
        if document_id is not None:
            must.append(
                qmodels.FieldCondition(
                    key="document_id", match=qmodels.MatchValue(value=document_id)
                )
            )
        if only_active is not None:
            must.append(
                qmodels.FieldCondition(
                    key="is_active", match=qmodels.MatchValue(value=only_active)
                )
            )

        effective = temporal or TemporalFilter()
        if effective.version is not None:
            must.append(
                qmodels.FieldCondition(
                    key="version", match=qmodels.MatchValue(value=int(effective.version))
                )
            )
        elif effective.as_of is not None:
            timestamp = effective.as_of.timestamp()
            must.append(
                qmodels.FieldCondition(key="valid_from", range=qmodels.Range(lte=timestamp))
            )
            # "Zum Zeitpunkt T gültig": valid_to offen ODER valid_to > T.
            must.append(
                qmodels.Filter(
                    should=[
                        qmodels.IsEmptyCondition(
                            is_empty=qmodels.PayloadField(key="valid_to")
                        ),
                        qmodels.FieldCondition(
                            key="valid_to", range=qmodels.Range(gt=timestamp)
                        ),
                    ]
                )
            )
        elif not effective.include_inactive and only_active is None:
            must.append(
                qmodels.FieldCondition(key="is_active", match=qmodels.MatchValue(value=True))
            )

        for key, value in (metadata_filter or {}).items():
            must.append(self._metadata_condition(key, value))

        return qmodels.Filter(must=must) if must else None

    # -- Schreiben ---------------------------------------------------------------

    async def upsert(self, points: Sequence[ChunkPoint]) -> None:
        for start in range(0, len(points), self._upsert_batch_size):
            batch = points[start : start + self._upsert_batch_size]
            structs = []
            for point in batch:
                vector: dict[str, Any] = {DENSE_VECTOR_NAME: point.dense}
                if point.sparse is not None and not point.sparse.is_empty():
                    vector[SPARSE_VECTOR_NAME] = qmodels.SparseVector(
                        indices=point.sparse.indices, values=point.sparse.values
                    )
                structs.append(
                    qmodels.PointStruct(id=point.point_id, vector=vector, payload=point.payload)
                )
            await self._call(
                "upsert",
                lambda batch_structs=structs: self._client.upsert(
                    collection_name=self._collection, points=batch_structs, wait=True
                ),
            )

    # -- Suchen -------------------------------------------------------------------

    async def search_dense(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        temporal: TemporalFilter | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        query_filter = self._build_filter(
            temporal=temporal, metadata_filter=metadata_filter, only_searchable=True
        )
        response = await self._call(
            "query_points(dense)",
            lambda: self._client.query_points(
                collection_name=self._collection,
                query=list(vector),
                using=DENSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            ),
        )
        return [
            ScoredChunk(point_id=str(point.id), score=float(point.score), payload=point.payload or {})
            for point in response.points
        ]

    async def search_sparse(
        self,
        vector: SparseVector,
        *,
        limit: int,
        temporal: TemporalFilter | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        if vector.is_empty():
            return []
        query_filter = self._build_filter(
            temporal=temporal, metadata_filter=metadata_filter, only_searchable=True
        )
        response = await self._call(
            "query_points(sparse)",
            lambda: self._client.query_points(
                collection_name=self._collection,
                query=qmodels.SparseVector(indices=vector.indices, values=vector.values),
                using=SPARSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            ),
        )
        return [
            ScoredChunk(point_id=str(point.id), score=float(point.score), payload=point.payload or {})
            for point in response.points
        ]

    async def fetch_by_ids(self, point_ids: Sequence[str]) -> list[ScoredChunk]:
        if not point_ids:
            return []
        records = await self._call(
            "retrieve",
            lambda: self._client.retrieve(
                collection_name=self._collection, ids=list(point_ids), with_payload=True
            ),
        )
        return [
            ScoredChunk(point_id=str(record.id), score=0.0, payload=record.payload or {})
            for record in records
        ]

    # -- Versionierung --------------------------------------------------------------

    async def _scroll_all(
        self,
        scroll_filter: Any,
        *,
        with_payload: Any = False,
        page_size: int = 256,
    ) -> list[Any]:
        results: list[Any] = []
        offset: Any = None
        while True:
            points, offset = await self._call(
                "scroll",
                lambda current_offset=offset: self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=scroll_filter,
                    limit=page_size,
                    offset=current_offset,
                    with_payload=with_payload,
                    with_vectors=False,
                ),
            )
            results.extend(points)
            if offset is None:
                return results

    async def get_latest_version(self, document_id: str) -> int | None:
        # Schneller Pfad: Der aktive Stand trägt (per Invariante) die höchste Version.
        active_filter = self._build_filter(
            temporal=TemporalFilter(include_inactive=True),
            metadata_filter=None,
            document_id=document_id,
            only_active=True,
        )
        points, _ = await self._call(
            "scroll(active)",
            lambda: self._client.scroll(
                collection_name=self._collection,
                scroll_filter=active_filter,
                limit=1,
                with_payload=["version"],
                with_vectors=False,
            ),
        )
        if points:
            return int(points[0].payload["version"])

        # Fallback: Kein aktiver Stand (z. B. nach Teil-Ingest) – Maximum über alle Punkte.
        all_filter = self._build_filter(
            temporal=TemporalFilter(include_inactive=True),
            metadata_filter=None,
            document_id=document_id,
        )
        records = await self._scroll_all(all_filter, with_payload=["version"])
        versions = [int(record.payload["version"]) for record in records if record.payload]
        return max(versions) if versions else None

    async def deactivate_document(
        self,
        document_id: str,
        *,
        valid_to: float,
        valid_to_iso: str,
        before_version: int | None = None,
    ) -> int:
        must: list[Any] = [
            qmodels.FieldCondition(
                key="document_id", match=qmodels.MatchValue(value=document_id)
            ),
            qmodels.FieldCondition(key="is_active", match=qmodels.MatchValue(value=True)),
        ]
        if before_version is not None:
            must.append(
                qmodels.FieldCondition(key="version", range=qmodels.Range(lt=before_version))
            )
        records = await self._scroll_all(qmodels.Filter(must=must), with_payload=False)
        point_ids = [record.id for record in records]
        for start in range(0, len(point_ids), 512):
            id_batch = point_ids[start : start + 512]
            await self._call(
                "set_payload",
                lambda batch=id_batch: self._client.set_payload(
                    collection_name=self._collection,
                    payload={
                        "is_active": False,
                        "valid_to": valid_to,
                        "valid_to_iso": valid_to_iso,
                    },
                    points=batch,
                    wait=True,
                ),
            )
        return len(point_ids)

    async def get_document_versions(self, document_id: str) -> list[dict[str, Any]]:
        all_filter = self._build_filter(
            temporal=TemporalFilter(include_inactive=True),
            metadata_filter=None,
            document_id=document_id,
        )
        records = await self._scroll_all(
            all_filter,
            with_payload=["version", "valid_from_iso", "valid_to_iso", "is_active"],
        )
        versions: dict[int, dict[str, Any]] = {}
        for record in records:
            payload = record.payload or {}
            version = int(payload.get("version", 0))
            entry = versions.setdefault(
                version,
                {
                    "version": version,
                    "valid_from": payload.get("valid_from_iso"),
                    "valid_to": payload.get("valid_to_iso"),
                    "is_active": bool(payload.get("is_active", False)),
                    "chunk_count": 0,
                },
            )
            entry["chunk_count"] += 1
        return [versions[key] for key in sorted(versions)]

    async def close(self) -> None:
        try:
            await self._client.close()
        except Exception as exc:  # noqa: BLE001 – Schließen ist best effort
            logger.debug("Qdrant-Client konnte nicht sauber geschlossen werden: %s", exc)
