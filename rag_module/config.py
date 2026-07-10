"""Zentrale Konfiguration des RAG-Moduls (Environment-Variablen mit Präfix ``RAG_``).

Philosophie: **local-first**. Die Defaults laufen vollständig ohne externe
API-Keys (fastembed/ONNX für Dense-Embeddings, BM25 und Cross-Encoder-Reranking;
optional Ollama für Query Expansion). Externe Dienste (Cohere, Anthropic) sind
reine Andockstellen, die über die ``*_backend``-Schalter aktiviert werden.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGSettings(BaseSettings):
    """Alle Einstellungen des Moduls; überschreibbar via Env-Vars (``RAG_*``) oder ``.env``.

    Beispiel: ``RAG_QDRANT_URL=http://qdrant:6333 RAG_RERANK_BACKEND=cohere python app.py``
    """

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- Vektordatenbank (Qdrant) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_timeout_s: float = 30.0
    collection_name: str = "rag_chunks"
    upsert_batch_size: int = Field(default=128, ge=1)

    # --- Dense-Embeddings ---
    # "fastembed": lokal via ONNX (Default, kein API-Key)
    # "bge_m3":    lokal via FlagEmbedding/Torch (beste lokale Qualität, große Installation)
    # "cohere":    Andockstelle Cohere v3 (benötigt RAG_COHERE_API_KEY)
    dense_backend: Literal["fastembed", "bge_m3", "cohere"] = "fastembed"
    fastembed_dense_model: str = "intfloat/multilingual-e5-large"
    #: Nur nötig, wenn ein fastembed-Modell außerhalb der offiziellen Liste genutzt wird.
    dense_dimension_override: Optional[int] = None
    bge_m3_model: str = "BAAI/bge-m3"
    cohere_api_key: Optional[str] = None
    cohere_embed_model: str = "embed-multilingual-v3.0"
    embed_batch_size: int = Field(default=96, ge=1, le=96)  # Cohere-API-Limit: 96 Texte/Call
    embed_concurrency: int = Field(default=4, ge=1)

    # --- Sparse-Embeddings (lexikalisch) ---
    # "fastembed_bm25": lokal, leichtgewichtig (Default)
    # "bge_m3":         lexical weights des (geteilten) BGE-M3-Modells
    sparse_backend: Literal["fastembed_bm25", "bge_m3"] = "fastembed_bm25"
    fastembed_sparse_model: str = "Qdrant/bm25"

    # --- Re-Ranking ---
    # "fastembed": lokaler Cross-Encoder via ONNX (Default, kein API-Key)
    # "bge":       lokaler FlagEmbedding-Reranker (z. B. bge-reranker-v2-m3, Torch)
    # "cohere":    Andockstelle Cohere-Rerank-API (benötigt RAG_COHERE_API_KEY)
    # "none":      Re-Ranking deaktivieren (Ergebnisse in RRF-Reihenfolge)
    rerank_backend: Literal["fastembed", "bge", "cohere", "none"] = "fastembed"
    fastembed_rerank_model: str = "BAAI/bge-reranker-base"
    bge_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    cohere_rerank_model: str = "rerank-v3.5"
    rerank_fail_open: bool = False  # True: Bei Reranker-Ausfall RRF-Reihenfolge liefern statt Fehler

    # --- Query Expansion (leichtes LLM) ---
    # "none":      keine Expansion (Default – läuft ohne weitere Infrastruktur)
    # "ollama":    lokales LLM via Ollama (kein API-Key, http://localhost:11434)
    # "anthropic": Andockstelle Claude Haiku (benötigt RAG_ANTHROPIC_API_KEY)
    expansion_backend: Literal["none", "ollama", "anthropic"] = "none"
    num_query_expansions: int = Field(default=3, ge=1, le=8)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    anthropic_api_key: Optional[str] = None
    anthropic_expansion_model: str = "claude-haiku-4-5-20251001"

    # --- Chunking ---
    chunk_max_tokens: int = Field(default=480, ge=64)
    chunk_overlap_tokens: int = Field(default=64, ge=0)
    parent_max_tokens: int = Field(default=1800, ge=256)
    min_chunk_tokens: int = Field(default=24, ge=1)
    table_hard_max_tokens: int = Field(default=4000, ge=256)  # Ab hier werden Tabellen zeilenweise geteilt

    # --- Retrieval-Pipeline ---
    per_query_limit: int = Field(default=50, ge=1)  # Treffer pro Einzelsuche (dense/sparse je Query-Variante)
    candidate_pool_size: int = Field(default=50, ge=1)  # Kandidaten, die an den Reranker gehen
    rrf_k: int = Field(default=60, ge=1)

    # --- Robustheit (Retry / Timeouts) ---
    retry_attempts: int = Field(default=4, ge=1)
    retry_base_delay_s: float = Field(default=0.5, gt=0)
    retry_max_delay_s: float = Field(default=20.0, gt=0)
    request_timeout_s: float = Field(default=60.0, gt=0)
