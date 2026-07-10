"""Advanced Retrieval Pipeline.

Ablauf pro Anfrage:
1. **Query Expansion**: Ein leichtes LLM generiert alternative Formulierungen
   (Synonyme, Fachbegriffe) der Original-Query. Fällt das LLM aus, läuft die
   Pipeline fail-open mit der Original-Query weiter – Expansion ist eine
   Recall-Optimierung, kein Single Point of Failure.
2. **Asynchrone Hybrid-Suche**: Für jede Query-Variante laufen Dense- und
   Sparse-Suche parallel (``asyncio.gather``) gegen die Vektordatenbank.
3. **Reciprocal Rank Fusion**: Alle Ranking-Listen werden per RRF zu einem
   Kandidaten-Pool fusioniert (score = Σ 1/(k + rank)).
4. **Re-Ranking**: Die Top-Kandidaten (Default 50) gehen an den Cohere-Reranker;
   die Top-N präzisesten Chunks werden – optional mit Parent-Kontext
   angereichert – zurückgegeben.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .embeddings import BaseDenseEmbedder, BaseSparseEmbedder
from .exceptions import QueryExpansionError, RerankingError
from .models import (
    FusedCandidate,
    RetrievalResult,
    ScoredChunk,
    TemporalFilter,
)
from .utils import require_module, retry_async
from .vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Leichtes LLM-Interface (für Query Expansion)
# ---------------------------------------------------------------------------


class BaseLLMClient(ABC):
    """Minimales LLM-Interface; bewusst schmal, damit jede Host-Applikation
    ihr bevorzugtes (leichtes) Modell anbinden kann."""

    @abstractmethod
    async def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        """Erzeugt eine einzelne Text-Vervollständigung."""


class OllamaLLMClient(BaseLLMClient):
    """Lokale LLM-Anbindung via Ollama (kein API-Key; Default: http://localhost:11434).

    Damit läuft die Query Expansion vollständig lokal, z. B. mit ``llama3.2``
    oder ``qwen2.5:3b``. Das Modell muss zuvor per ``ollama pull <modell>``
    geladen worden sein.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout_s: float = 60.0,
        retry_attempts: int = 3,
        retry_base_delay_s: float = 0.5,
        retry_max_delay_s: float = 10.0,
    ) -> None:
        httpx = require_module("httpx", hint="Installation: pip install httpx (Kern-Abhängigkeit).")
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s)
        self._model = model
        self._timeout_s = timeout_s
        self._retry_attempts = retry_attempts
        self._retry_base_delay_s = retry_base_delay_s
        self._retry_max_delay_s = retry_max_delay_s

    async def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        async def call() -> str:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.3},
                },
            )
            response.raise_for_status()
            return str(response.json()["message"]["content"])

        return await retry_async(
            call,
            op_name="ollama.chat",
            attempts=self._retry_attempts,
            base_delay=self._retry_base_delay_s,
            max_delay=self._retry_max_delay_s,
            timeout=self._timeout_s,
        )


class AnthropicLLMClient(BaseLLMClient):
    """Andockstelle Anthropic-API (Default: Claude Haiku als leichtes Expansions-Modell)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        timeout_s: float = 30.0,
        retry_attempts: int = 3,
        retry_base_delay_s: float = 0.5,
        retry_max_delay_s: float = 10.0,
    ) -> None:
        anthropic = require_module(
            "anthropic", hint="Installation: pip install 'rag-module[anthropic]'"
        )
        # Retries übernimmt unsere eigene Backoff-Logik, daher max_retries=0.
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout_s, max_retries=0
        )
        self._model = model
        self._timeout_s = timeout_s
        self._retry_attempts = retry_attempts
        self._retry_base_delay_s = retry_base_delay_s
        self._retry_max_delay_s = retry_max_delay_s

    async def complete(self, *, system: str, user: str, max_tokens: int = 512) -> str:
        async def call() -> Any:
            return await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )

        message = await retry_async(
            call,
            op_name="anthropic.messages.create",
            attempts=self._retry_attempts,
            base_delay=self._retry_base_delay_s,
            max_delay=self._retry_max_delay_s,
            timeout=self._timeout_s,
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )


# ---------------------------------------------------------------------------
# Query Expansion
# ---------------------------------------------------------------------------

_EXPANSION_SYSTEM_PROMPT = (
    "Du bist ein Retrieval-Assistent. Du erhältst eine Suchanfrage und erzeugst "
    "alternative Formulierungen, die dieselbe Informationsabsicht mit anderen "
    "Worten ausdrücken: Synonyme, Fachbegriffe, ausgeschriebene Abkürzungen, "
    "eine andere Perspektive. Bleibe in der Sprache der Original-Anfrage. "
    "Antworte AUSSCHLIESSLICH mit einem JSON-Array aus Strings, ohne Erklärung, "
    "ohne Markdown-Codeblock."
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_string_array(raw: str) -> list[str]:
    """Extrahiert robust ein String-Array aus einer LLM-Antwort.

    Reihenfolge: direktes JSON-Parsing -> erstes JSON-Array im Text ->
    zeilenweises Parsen mit Entfernung von Aufzählungszeichen.
    """
    candidates: list[Any] = []
    for attempt in (raw, *( [m.group(0)] if (m := _JSON_ARRAY_RE.search(raw)) else [] )):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, list):
                candidates = parsed
                break
        except (json.JSONDecodeError, TypeError):
            continue
    if not candidates:
        for line in raw.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"\'')
            if len(cleaned) > 2:
                candidates.append(cleaned)
    return [str(item).strip() for item in candidates if str(item).strip()]


class QueryExpander:
    """Generiert alternative Suchanfragen über ein leichtes LLM (fail-open)."""

    def __init__(self, llm: BaseLLMClient, *, num_variants: int = 3) -> None:
        self._llm = llm
        self._num_variants = num_variants

    async def expand(self, query: str) -> list[str]:
        """Liefert bis zu ``num_variants`` alternative Queries; bei Fehlern ``[]``."""
        user_prompt = (
            f"Original-Suchanfrage: {query!r}\n"
            f"Erzeuge genau {self._num_variants} alternative Suchanfragen als JSON-Array."
        )
        try:
            raw = await self._llm.complete(
                system=_EXPANSION_SYSTEM_PROMPT, user=user_prompt, max_tokens=400
            )
            variants = parse_string_array(raw)
            if not variants:
                raise QueryExpansionError(f"Keine Varianten aus LLM-Antwort extrahierbar: {raw!r}")
        except Exception as exc:  # noqa: BLE001 – Expansion ist fail-open by design
            logger.warning("Query-Expansion fehlgeschlagen (fahre ohne fort): %s", exc)
            return []

        seen = {query.casefold().strip()}
        unique: list[str] = []
        for variant in variants:
            key = variant.casefold().strip()
            if key not in seen:
                seen.add(key)
                unique.append(variant)
            if len(unique) >= self._num_variants:
                break
        return unique


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], *, k: int = 60
) -> list[FusedCandidate]:
    """Fusioniert mehrere Ranking-Listen per RRF: score(d) = Σ 1/(k + rank(d) + 1).

    Roh-Scores der Einzelsuchen werden bewusst ignoriert – RRF ist gegenüber
    unterschiedlich skalierten Score-Räumen (Cosine vs. BM25) invariant.
    Deterministische Tie-Breaks über besten Einzelrang und Punkt-ID.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    hit_counts: dict[str, int] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for ranking in rankings:
        for rank, scored in enumerate(ranking):
            pid = scored.point_id
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            best_rank[pid] = min(best_rank.get(pid, rank), rank)
            hit_counts[pid] = hit_counts.get(pid, 0) + 1
            payloads.setdefault(pid, scored.payload)

    fused = [
        FusedCandidate(
            point_id=pid,
            payload=payloads[pid],
            rrf_score=scores[pid],
            best_rank=best_rank[pid],
            hit_count=hit_counts[pid],
        )
        for pid in scores
    ]
    fused.sort(key=lambda candidate: (-candidate.rrf_score, candidate.best_rank, candidate.point_id))
    return fused


# ---------------------------------------------------------------------------
# Re-Ranking
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RerankItem:
    """Ein Reranker-Ergebnis: Index im Kandidaten-Pool + Relevanz-Score."""

    index: int
    relevance_score: float


class BaseReranker(ABC):
    """Interface für dedizierte Re-Ranking-Stufen."""

    @abstractmethod
    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankItem]:
        """Ordnet ``documents`` nach Relevanz zur ``query``; absteigend sortiert."""


#: Obergrenze pro Dokument, um Payloads zu begrenzen (Modelle kürzen darüber
#: hinaus auf ihr eigenes Token-Fenster).
_RERANK_MAX_DOC_CHARS = 16_000


class FastEmbedReranker(BaseReranker):
    """Lokaler Cross-Encoder-Reranker via fastembed/ONNX (Default-Backend, kein API-Key).

    Default-Modell: ``BAAI/bge-reranker-base``. Alternativen (siehe
    ``fastembed.rerank.cross_encoder.TextCrossEncoder.list_supported_models()``):
    ``jinaai/jina-reranker-v2-base-multilingual`` (stärker multilingual,
    Achtung: CC-BY-NC-Lizenz) oder ``Xenova/ms-marco-MiniLM-L-6-v2`` (sehr klein).
    """

    def __init__(self, *, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is None:
                require_module(
                    "fastembed", hint="Installation: pip install fastembed (Kern-Abhängigkeit)."
                )
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                logger.info("Lade Reranker-Modell '%s' (einmalig) …", self._model_name)
                self._model = await asyncio.to_thread(TextCrossEncoder, self._model_name)

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankItem]:
        if not documents:
            return []
        await self._ensure_model()
        prepared = [doc[:_RERANK_MAX_DOC_CHARS] for doc in documents]
        try:
            scores = await asyncio.to_thread(
                lambda: list(self._model.rerank(query, prepared))
            )
        except Exception as exc:  # noqa: BLE001 – lokale Inferenzfehler kapseln
            raise RerankingError(f"Lokales Cross-Encoder-Reranking fehlgeschlagen: {exc}") from exc
        items = [
            RerankItem(index=index, relevance_score=float(score))
            for index, score in enumerate(scores)
        ]
        items.sort(key=lambda item: (-item.relevance_score, item.index))
        return items[:top_n]


class BGEReranker(BaseReranker):
    """Lokaler FlagEmbedding-Reranker (Torch), z. B. ``BAAI/bge-reranker-v2-m3``.

    Beste lokale Multilingual-Qualität; benötigt das ``bge``-Extra
    (FlagEmbedding inkl. Torch). Aufrufe werden serialisiert, die Inferenz
    läuft in einem Worker-Thread.
    """

    def __init__(self, *, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True) -> None:
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankItem]:
        if not documents:
            return []
        async with self._lock:
            if self._model is None:
                flag_embedding = require_module(
                    "FlagEmbedding", hint="Installation: pip install 'rag-module[bge]'"
                )
                logger.info("Lade BGE-Reranker '%s' (einmalig) …", self._model_name)
                self._model = await asyncio.to_thread(
                    flag_embedding.FlagReranker, self._model_name, use_fp16=self._use_fp16
                )
            pairs = [[query, doc[:_RERANK_MAX_DOC_CHARS]] for doc in documents]
            try:
                scores = await asyncio.to_thread(
                    self._model.compute_score, pairs, normalize=True
                )
            except Exception as exc:  # noqa: BLE001 – lokale Inferenzfehler kapseln
                raise RerankingError(f"BGE-Reranking fehlgeschlagen: {exc}") from exc
        if isinstance(scores, float):  # FlagEmbedding gibt bei einem Paar einen Skalar zurück
            scores = [scores]
        items = [
            RerankItem(index=index, relevance_score=float(score))
            for index, score in enumerate(scores)
        ]
        items.sort(key=lambda item: (-item.relevance_score, item.index))
        return items[:top_n]


class CohereReranker(BaseReranker):
    """Andockstelle Cohere-Rerank-API (Default: rerank-v3.5; benötigt API-Key)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "rerank-v3.5",
        timeout_s: float = 60.0,
        retry_attempts: int = 4,
        retry_base_delay_s: float = 0.5,
        retry_max_delay_s: float = 20.0,
    ) -> None:
        cohere = require_module("cohere", hint="Installation: pip install 'rag-module[cohere]'")
        self._client = cohere.AsyncClientV2(api_key=api_key, timeout=timeout_s)
        self._model = model
        self._timeout_s = timeout_s
        self._retry_attempts = retry_attempts
        self._retry_base_delay_s = retry_base_delay_s
        self._retry_max_delay_s = retry_max_delay_s

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankItem]:
        if not documents:
            return []
        prepared = [doc[:_RERANK_MAX_DOC_CHARS] for doc in documents]

        async def call() -> Any:
            return await self._client.rerank(
                model=self._model,
                query=query,
                documents=prepared,
                top_n=min(top_n, len(prepared)),
            )

        try:
            response = await retry_async(
                call,
                op_name="cohere.rerank",
                attempts=self._retry_attempts,
                base_delay=self._retry_base_delay_s,
                max_delay=self._retry_max_delay_s,
                timeout=self._timeout_s,
            )
        except Exception as exc:
            raise RerankingError(
                f"Cohere-Reranking nach allen Retries fehlgeschlagen: {exc}"
            ) from exc
        return [
            RerankItem(index=result.index, relevance_score=float(result.relevance_score))
            for result in response.results
        ]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RetrievalPipeline:
    """Orchestriert Expansion -> Hybrid-Suche -> RRF -> Re-Ranking -> Anreicherung."""

    def __init__(
        self,
        *,
        vector_store: BaseVectorStore,
        dense_embedder: BaseDenseEmbedder,
        sparse_embedder: Optional[BaseSparseEmbedder],
        query_expander: Optional[QueryExpander],
        reranker: Optional[BaseReranker],
        per_query_limit: int = 50,
        candidate_pool_size: int = 50,
        rrf_k: int = 60,
        rerank_fail_open: bool = False,
    ) -> None:
        self._store = vector_store
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._expander = query_expander
        self._reranker = reranker
        self._per_query_limit = per_query_limit
        self._candidate_pool_size = candidate_pool_size
        self._rrf_k = rrf_k
        self._rerank_fail_open = rerank_fail_open

    async def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        temporal: Optional[TemporalFilter] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        include_parent_context: bool = True,
    ) -> list[RetrievalResult]:
        # Schritt 1: Query Expansion (fail-open)
        queries: list[str] = [query]
        if self._expander is not None:
            queries.extend(await self._expander.expand(query))

        # Schritt 2: Alle Query-Varianten parallel einbetten …
        dense_vectors, sparse_vectors = await asyncio.gather(
            self._dense.embed_queries(queries),
            self._sparse.embed_queries(queries)
            if self._sparse is not None
            else _none_list(len(queries)),
        )

        # … und Dense- + Sparse-Suchen für alle Varianten parallel ausführen.
        search_tasks = []
        for dense_vector, sparse_vector in zip(dense_vectors, sparse_vectors):
            search_tasks.append(
                self._store.search_dense(
                    dense_vector,
                    limit=self._per_query_limit,
                    temporal=temporal,
                    metadata_filter=metadata_filter,
                )
            )
            if sparse_vector is not None and not sparse_vector.is_empty():
                search_tasks.append(
                    self._store.search_sparse(
                        sparse_vector,
                        limit=self._per_query_limit,
                        temporal=temporal,
                        metadata_filter=metadata_filter,
                    )
                )
        ranking_lists = await asyncio.gather(*search_tasks)

        # Schritt 3: Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(ranking_lists, k=self._rrf_k)
        pool = fused[: self._candidate_pool_size]
        if not pool:
            return []

        # Schritt 4: Re-Ranking der Top-Kandidaten
        effective_limit = min(limit, len(pool))
        if self._reranker is not None:
            try:
                order = await self._reranker.rerank(
                    query,
                    [candidate.payload.get("content", "") for candidate in pool],
                    top_n=effective_limit,
                )
                selected = [(pool[item.index], item.relevance_score, "rerank") for item in order]
            except RerankingError:
                if not self._rerank_fail_open:
                    raise
                logger.warning(
                    "Reranker nicht verfügbar – fail-open: liefere RRF-Reihenfolge."
                )
                selected = [
                    (candidate, candidate.rrf_score, "rrf")
                    for candidate in pool[:effective_limit]
                ]
        else:
            selected = [
                (candidate, candidate.rrf_score, "rrf")
                for candidate in pool[:effective_limit]
            ]

        # Parent-Kontext (Parent-Child-Chunking) in einem Batch nachladen.
        parent_contents: dict[str, str] = {}
        if include_parent_context:
            parent_ids = {
                candidate.payload.get("parent_point_id")
                for candidate, _, _ in selected
                if candidate.payload.get("parent_point_id")
            }
            if parent_ids:
                parents = await self._store.fetch_by_ids(sorted(parent_ids))
                parent_contents = {
                    parent.point_id: str(parent.payload.get("content", ""))
                    for parent in parents
                }

        results: list[RetrievalResult] = []
        for candidate, score, origin in selected:
            payload = candidate.payload
            parent_point_id = payload.get("parent_point_id")
            results.append(
                RetrievalResult(
                    chunk_id=candidate.point_id,
                    content=str(payload.get("content", "")),
                    score=float(score),
                    score_origin=origin,
                    rrf_score=candidate.rrf_score,
                    chunk_type=str(payload.get("chunk_type", "")),
                    chunk_role=str(payload.get("chunk_role", "")),
                    document_id=str(payload.get("document_id", "")),
                    document_type=str(payload.get("document_type", "")),
                    version=int(payload.get("version", 0)),
                    is_active=bool(payload.get("is_active", False)),
                    valid_from=payload.get("valid_from_iso"),
                    valid_to=payload.get("valid_to_iso"),
                    hierarchy=list(payload.get("hierarchy") or []),
                    parent_chunk_id=parent_point_id,
                    parent_content=parent_contents.get(parent_point_id)
                    if parent_point_id
                    else None,
                    source=payload.get("source"),
                    metadata=dict(payload.get("meta") or {}),
                    extra=dict(payload.get("chunk_extra") or {}),
                )
            )
        return results


async def _none_list(length: int) -> list[None]:
    """Hilfs-Coroutine: Platzhalter, wenn kein Sparse-Embedder konfiguriert ist."""
    return [None] * length
