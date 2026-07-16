"""Service-Fassade: :class:`AdvancedRAGModule`.

Arbeitsprinzip: "Raw Query + Metadaten-Filter REIN -> Top-N reranked Chunks RAUS".
Kontext- und Chat-Management (Memory) verbleiben in der Host-Applikation.

Nebenläufigkeit / Thread-Sicherheit:
- Alle öffentlichen Methoden sind Coroutinen und **coroutine-safe**: Die
  Collection-Initialisierung ist per Lock geschützt, Versionsübergänge werden
  pro ``document_id`` serialisiert (dokument-spezifische ``asyncio.Lock``s),
  und sämtliche Komponenten (Chunker, Embedder, Stores) sind zustandslos bzw.
  intern synchronisiert.
- Eine Instanz ist an die Event-Loop gebunden, in der sie verwendet wird
  (Standard-Semantik von ``asyncio``-Primitiven). Für Multi-Thread-Szenarien
  pro Loop eine eigene Instanz erzeugen; die Qdrant-Collection darf dabei
  geteilt werden.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from .chunking import ChunkingEngine
from .config import RAGSettings
from .embeddings import (
    BaseDenseEmbedder,
    BaseSparseEmbedder,
    BGEM3DenseView,
    BGEM3Embedder,
    BGEM3SparseView,
    CohereDenseEmbedder,
    FastEmbedBM25SparseEmbedder,
    FastEmbedDenseEmbedder,
)
from .exceptions import ConfigurationError, EmbeddingError
from .models import Chunk, DocumentType, TemporalFilter
from .retrieval import (
    AnthropicLLMClient,
    BaseReranker,
    BGEReranker,
    CohereReranker,
    FastEmbedReranker,
    OllamaLLMClient,
    QueryExpander,
    RetrievalPipeline,
)
from .utils import utc_now
from .vector_store import BaseVectorStore, ChunkPoint, QdrantVectorStore

logger = logging.getLogger(__name__)

_POINT_ID_NAMESPACE = uuid.NAMESPACE_URL


class AdvancedRAGModule:
    """Produktionsreifes RAG-Modul: Ingestion mit Versionierung + Advanced Retrieval.

    Alle Abhängigkeiten sind über den Konstruktor injizierbar (Dependency
    Injection); nicht übergebene Komponenten werden aus den ``settings``
    aufgebaut. Damit ist das Modul vollständig testbar und die Vektordatenbank
    bzw. die Embedding-Provider sind austauschbar (Repository Pattern).
    """

    def __init__(
        self,
        *,
        settings: RAGSettings | None = None,
        vector_store: BaseVectorStore | None = None,
        dense_embedder: BaseDenseEmbedder | None = None,
        sparse_embedder: BaseSparseEmbedder | None = None,
        chunking_engine: ChunkingEngine | None = None,
        query_expander: QueryExpander | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        self._settings = settings or RAGSettings()
        self._store = vector_store or self._build_vector_store(self._settings)
        self._chunking = chunking_engine or ChunkingEngine(self._settings)

        built_dense, built_sparse = self._build_embedders(
            self._settings,
            need_dense=dense_embedder is None,
            need_sparse=sparse_embedder is None,
        )
        self._dense = dense_embedder or built_dense
        self._sparse = sparse_embedder or built_sparse
        if self._dense is None:
            raise ConfigurationError("Es konnte kein Dense-Embedder aufgebaut werden.")

        self._reranker = reranker if reranker is not None else self._build_reranker(self._settings)
        self._expander = (
            query_expander
            if query_expander is not None
            else self._build_query_expander(self._settings)
        )

        self._pipeline = RetrievalPipeline(
            vector_store=self._store,
            dense_embedder=self._dense,
            sparse_embedder=self._sparse,
            query_expander=self._expander,
            reranker=self._reranker,
            per_query_limit=self._settings.per_query_limit,
            candidate_pool_size=self._settings.candidate_pool_size,
            rrf_k=self._settings.rrf_k,
            rerank_fail_open=self._settings.rerank_fail_open,
        )

        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._document_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Komponenten-Aufbau aus Settings
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: RAGSettings) -> AdvancedRAGModule:
        """Bequemer Konstruktor: baut alle Komponenten aus den Settings."""
        return cls(settings=settings)

    @staticmethod
    def _build_vector_store(settings: RAGSettings) -> BaseVectorStore:
        return QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.collection_name,
            timeout_s=settings.qdrant_timeout_s,
            upsert_batch_size=settings.upsert_batch_size,
            retry_attempts=settings.retry_attempts,
            retry_base_delay_s=settings.retry_base_delay_s,
            retry_max_delay_s=settings.retry_max_delay_s,
        )

    @staticmethod
    def _build_embedders(
        settings: RAGSettings, *, need_dense: bool, need_sparse: bool
    ) -> tuple[BaseDenseEmbedder | None, BaseSparseEmbedder | None]:
        dense: BaseDenseEmbedder | None = None
        sparse: BaseSparseEmbedder | None = None

        # Ein geteilter BGE-M3-Kern, falls beide Seiten ihn nutzen sollen –
        # das Modell wird dann nur einmal geladen.
        shared_bge: BGEM3Embedder | None = None
        if (need_dense and settings.dense_backend == "bge_m3") or (
            need_sparse and settings.sparse_backend == "bge_m3"
        ):
            shared_bge = BGEM3Embedder(model_name=settings.bge_m3_model)

        if need_dense:
            if settings.dense_backend == "cohere":
                if not settings.cohere_api_key:
                    raise ConfigurationError(
                        "dense_backend='cohere' benötigt RAG_COHERE_API_KEY "
                        "(lokale Alternativen: dense_backend='fastembed' oder 'bge_m3')."
                    )
                dense = CohereDenseEmbedder(
                    api_key=settings.cohere_api_key,
                    model=settings.cohere_embed_model,
                    batch_size=settings.embed_batch_size,
                    concurrency=settings.embed_concurrency,
                    timeout_s=settings.request_timeout_s,
                    retry_attempts=settings.retry_attempts,
                    retry_base_delay_s=settings.retry_base_delay_s,
                    retry_max_delay_s=settings.retry_max_delay_s,
                )
            elif settings.dense_backend == "bge_m3":
                assert shared_bge is not None
                dense = BGEM3DenseView(shared_bge)
            else:  # "fastembed" – lokaler Default
                dense = FastEmbedDenseEmbedder(
                    model_name=settings.fastembed_dense_model,
                    dimension=settings.dense_dimension_override,
                )

        if need_sparse:
            if settings.sparse_backend == "bge_m3":
                assert shared_bge is not None
                sparse = BGEM3SparseView(shared_bge)
            else:
                sparse = FastEmbedBM25SparseEmbedder(model_name=settings.fastembed_sparse_model)

        return dense, sparse

    @staticmethod
    def _build_reranker(settings: RAGSettings) -> BaseReranker | None:
        if settings.rerank_backend == "none":
            logger.info(
                "Re-Ranking ist deaktiviert (rerank_backend='none') – Ergebnisse "
                "folgen der RRF-Reihenfolge."
            )
            return None
        if settings.rerank_backend == "cohere":
            if not settings.cohere_api_key:
                raise ConfigurationError(
                    "rerank_backend='cohere' benötigt RAG_COHERE_API_KEY "
                    "(lokale Alternativen: rerank_backend='fastembed' oder 'bge')."
                )
            return CohereReranker(
                api_key=settings.cohere_api_key,
                model=settings.cohere_rerank_model,
                timeout_s=settings.request_timeout_s,
                retry_attempts=settings.retry_attempts,
                retry_base_delay_s=settings.retry_base_delay_s,
                retry_max_delay_s=settings.retry_max_delay_s,
            )
        if settings.rerank_backend == "bge":
            return BGEReranker(model_name=settings.bge_rerank_model)
        return FastEmbedReranker(model_name=settings.fastembed_rerank_model)

    @staticmethod
    def _build_query_expander(settings: RAGSettings) -> QueryExpander | None:
        if settings.expansion_backend == "none":
            return None
        if settings.expansion_backend == "ollama":
            llm: AnthropicLLMClient | OllamaLLMClient = OllamaLLMClient(
                base_url=settings.ollama_url,
                model=settings.ollama_model,
                timeout_s=settings.request_timeout_s,
            )
        else:  # "anthropic"
            if not settings.anthropic_api_key:
                raise ConfigurationError(
                    "expansion_backend='anthropic' benötigt RAG_ANTHROPIC_API_KEY "
                    "(lokale Alternative: expansion_backend='ollama')."
                )
            llm = AnthropicLLMClient(
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_expansion_model,
                timeout_s=settings.request_timeout_s,
            )
        return QueryExpander(llm, num_variants=settings.num_query_expansions)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._store.ensure_ready(self._dense.dimension)
            self._initialized = True

    def _document_lock(self, document_id: str) -> asyncio.Lock:
        # setdefault ist innerhalb einer Event-Loop atomar genug (kein await dazwischen).
        return self._document_locks.setdefault(document_id, asyncio.Lock())

    async def close(self) -> None:
        """Gibt alle Ressourcen frei (Datenbank-Verbindungen, Modelle)."""
        await self._store.close()
        await self._dense.close()
        if self._sparse is not None:
            await self._sparse.close()

    async def __aenter__(self) -> AdvancedRAGModule:
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Ingestion mit temporaler Versionierung
    # ------------------------------------------------------------------

    async def ingest_document(
        self, file_path: str, document_type: str, metadata: dict[str, Any]
    ) -> str:
        """Ingestiert ein Dokument als NEUE Version (alte Stände bleiben erhalten).

        Ablauf: layout-bewusstes Chunking -> Dense+Sparse-Embedding (parallel)
        -> Versionsübergang unter Dokument-Lock: alter Stand wird auf
        ``is_active=False`` gesetzt (mit ``valid_to``-Zeitstempel), der neue
        Stand mit ``is_active=True`` und inkrementierter ``version`` indiziert.

        Args:
            file_path: Pfad zur Quelldatei.
            document_type: z. B. ``pdf``, ``markdown``, ``json``, ``api_payload``,
                ``code``, ``sql_dump``, ``legal``, ``text`` (Aliase wie ``md``,
                ``contract``, ``sql`` werden akzeptiert).
            metadata: Frei definierbare, JSON-serialisierbare Metadaten. Der
                Schlüssel ``document_id`` steuert die Versionierung: gleiche ID
                => neue Version desselben Dokuments. Ohne ``document_id`` wird
                eine stabile ID aus dem Dateinamen abgeleitet.

        Returns:
            Die ``document_id`` des ingestierten Dokuments.
        """
        await self._ensure_initialized()

        metadata = dict(metadata or {})
        dtype = DocumentType.from_raw(document_type)
        document_id = str(metadata.pop("document_id", "") or "").strip() or _derive_document_id(
            file_path
        )
        _validate_metadata_serializable(metadata)

        # Layout-bewusstes Chunking (CPU-Arbeit im Worker-Thread).
        chunks = await self._chunking.chunk_file(file_path, dtype, metadata)

        return await self._index_chunks_as_new_version(
            chunks=chunks,
            document_id=document_id,
            document_type=dtype,
            metadata=metadata,
            source=Path(file_path).name,
        )

    async def ingest_text(
        self,
        content: str,
        document_type: str,
        metadata: dict[str, Any],
        source_name: str = "",
    ) -> str:
        """Ingestiert bereits vorliegenden Text (ohne Datei) als NEUE Version.

        Gleicher Ablauf wie :meth:`ingest_document`, nur ohne Datei-Loading;
        ``pdf`` ist ausgenommen (binäres Format).

        Versionierung: ``metadata["document_id"]`` steuert sie wie gehabt.
        Fehlt sie, wird die ID aus ``source_name`` abgeleitet; fehlt auch der,
        erhält der Text eine zufällige ID — dann gibt es KEINE Versionierung
        über mehrere Ingests hinweg.
        """
        if not content or not content.strip():
            raise ValueError("content darf nicht leer sein.")
        await self._ensure_initialized()

        metadata = dict(metadata or {})
        dtype = DocumentType.from_raw(document_type)
        if dtype is DocumentType.PDF:
            raise ValueError(
                "PDF kann nur über ingest_document(file_path=...) ingestiert werden."
            )
        document_id = str(metadata.pop("document_id", "") or "").strip()
        if not document_id:
            document_id = (
                _derive_document_id(source_name) if source_name.strip() else uuid.uuid4().hex
            )
        _validate_metadata_serializable(metadata)

        chunks = await self._chunking.chunk_text(content, dtype, source_name=source_name)

        return await self._index_chunks_as_new_version(
            chunks=chunks,
            document_id=document_id,
            document_type=dtype,
            metadata=metadata,
            source=source_name or "inline",
        )

    async def _index_chunks_as_new_version(
        self,
        *,
        chunks: list[Chunk],
        document_id: str,
        document_type: DocumentType,
        metadata: dict[str, Any],
        source: str,
    ) -> str:
        """Embeddet Chunks und indiziert sie als neue Dokumentversion."""
        # Dense- und Sparse-Embeddings parallel erzeugen.
        texts = [chunk.content for chunk in chunks]
        if self._sparse is not None:
            dense_vectors, sparse_vectors = await asyncio.gather(
                self._dense.embed_documents(texts),
                self._sparse.embed_documents(texts),
            )
        else:
            dense_vectors = await self._dense.embed_documents(texts)
            sparse_vectors = [None] * len(texts)
        if len(dense_vectors) != len(chunks) or len(sparse_vectors) != len(chunks):
            raise EmbeddingError(
                f"Embedding-Anzahl passt nicht zur Chunk-Anzahl "
                f"({len(dense_vectors)}/{len(sparse_vectors)} vs. {len(chunks)})."
            )

        # Versionsübergang – pro Dokument serialisiert.
        async with self._document_lock(document_id):
            latest_version = await self._store.get_latest_version(document_id)
            new_version = (latest_version or 0) + 1
            now = utc_now()
            timestamp = now.timestamp()
            iso = now.isoformat()

            points = self._build_points(
                chunks=chunks,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
                document_id=document_id,
                document_type=document_type,
                metadata=metadata,
                version=new_version,
                valid_from=timestamp,
                valid_from_iso=iso,
                source=source,
            )
            # Erst deaktivieren, dann schreiben: Es gibt nie zwei aktive Stände
            # gleichzeitig (Akkuratheit vor Verfügbarkeit; das Zeitfenster ohne
            # aktiven Stand ist die Dauer des Upserts).
            deactivated = await self._store.deactivate_document(
                document_id,
                valid_to=timestamp,
                valid_to_iso=iso,
                before_version=new_version,
            )
            await self._store.upsert(points)

        logger.info(
            "Dokument '%s' ingestiert: Version %d, %d Chunks (%d Punkte des "
            "Vorgängerstands deaktiviert).",
            document_id,
            new_version,
            len(chunks),
            deactivated,
        )
        return document_id

    def _build_points(
        self,
        *,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[Any],
        document_id: str,
        document_type: DocumentType,
        metadata: dict[str, Any],
        version: int,
        valid_from: float,
        valid_from_iso: str,
        source: str,
    ) -> list[ChunkPoint]:
        # Deterministische Punkt-IDs: identischer (Dokument, Version, Sequenz)-Tripel
        # ergibt dieselbe ID – ein wiederholter Upsert derselben Version ist idempotent.
        id_map = {
            chunk.chunk_id: str(
                uuid.uuid5(
                    _POINT_ID_NAMESPACE, f"rag://{document_id}/v{version}/{chunk.sequence}"
                )
            )
            for chunk in chunks
        }
        points: list[ChunkPoint] = []
        for chunk, dense_vector, sparse_vector in zip(
            chunks, dense_vectors, sparse_vectors, strict=True
        ):
            payload: dict[str, Any] = {
                "document_id": document_id,
                "document_type": document_type.value,
                "source": source,
                "chunk_type": chunk.chunk_type.value,
                "chunk_role": chunk.role.value,
                "searchable": chunk.searchable,
                "sequence": chunk.sequence,
                "hierarchy": list(chunk.hierarchy),
                "parent_point_id": id_map.get(chunk.parent_id) if chunk.parent_id else None,
                "version": version,
                "valid_from": valid_from,
                "valid_from_iso": valid_from_iso,
                "valid_to": None,
                "valid_to_iso": None,
                "is_active": True,
                "token_estimate": chunk.token_estimate,
                "content": chunk.content,
                "meta": metadata,
                "chunk_extra": chunk.extra,
            }
            points.append(
                ChunkPoint(
                    point_id=id_map[chunk.chunk_id],
                    dense=dense_vector,
                    sparse=sparse_vector,
                    payload=payload,
                )
            )
        return points

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        temporal_filter: dict[str, Any] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        include_parent_context: bool = True,
    ) -> list[dict[str, Any]]:
        """Advanced Retrieval: Expansion -> Hybrid-Suche -> RRF -> Re-Ranking.

        Args:
            query: Rohe Suchanfrage der Host-Applikation.
            limit: Anzahl der zurückgegebenen Top-Chunks.
            temporal_filter: Optional; Schlüssel ``as_of`` (ISO-String/datetime/
                Unix-Timestamp), ``version`` (int) oder ``include_inactive``
                (bool). Ohne Filter wird der aktuell aktive Stand durchsucht.
            metadata_filter: Exakte Filter auf Ingestion-Metadaten, z. B.
                ``{"tenant": "acme", "tags": ["hr", "legal"]}`` (Listen =>
                Match auf beliebigen Wert; dicts mit gt/gte/lt/lte => Range).
            include_parent_context: Lädt zu Child-Chunks den Parent-Kontext nach.

        Returns:
            Liste von Ergebnis-Dictionaries, absteigend nach Relevanz sortiert
            (Felder: content, score, parent_content, version, valid_from, …).
        """
        if not query or not query.strip():
            raise ValueError("query darf nicht leer sein.")
        if limit < 1:
            raise ValueError("limit muss >= 1 sein.")
        missing_required_keys = [
            key
            for key in self._settings.required_filter_keys
            if metadata_filter is None or metadata_filter.get(key) is None
        ]
        if missing_required_keys:
            raise ValueError(
                f"metadata_filter muss die Pflicht-Schlüssel {missing_required_keys} "
                "enthalten (RAG_REQUIRED_FILTER_KEYS, z. B. für Mandanten-Isolation)."
            )
        await self._ensure_initialized()
        temporal = TemporalFilter.from_value(temporal_filter)
        results = await self._pipeline.retrieve(
            query.strip(),
            limit=limit,
            temporal=temporal,
            metadata_filter=metadata_filter,
            include_parent_context=include_parent_context,
        )
        return [result.to_dict() for result in results]

    # ------------------------------------------------------------------
    # Auskunft
    # ------------------------------------------------------------------

    async def get_document_history(self, document_id: str) -> list[dict[str, Any]]:
        """Versionshistorie eines Dokuments (Version, Gültigkeitszeitraum, Chunk-Anzahl)."""
        await self._ensure_initialized()
        return await self._store.get_document_versions(document_id)


def _validate_metadata_serializable(metadata: dict[str, Any]) -> None:
    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "metadata muss JSON-serialisierbar sein (nur str/int/float/bool/"
            f"list/dict/None): {exc}"
        ) from exc


def _derive_document_id(file_path: str) -> str:
    """Stabile Dokument-ID aus dem Dateinamen (ohne Endung, URL-sicher)."""
    stem = Path(file_path).stem.strip().lower()
    slug = re.sub(r"[^a-z0-9äöüß]+", "-", stem).strip("-")
    return slug or uuid.uuid4().hex
