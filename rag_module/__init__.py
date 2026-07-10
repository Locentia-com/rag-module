"""Modulares, hochpräzises RAG-Modul für die Einbindung als eigenständiger Service.

Öffentliche API:
    - :class:`AdvancedRAGModule` – Service-Fassade (Ingestion + Retrieval)
    - :class:`RAGSettings` – Konfiguration (Env-Präfix ``RAG_``)
    - Interfaces für Dependency Injection: :class:`BaseVectorStore`,
      :class:`BaseDenseEmbedder`, :class:`BaseSparseEmbedder`,
      :class:`BaseReranker`, :class:`BaseLLMClient`, :class:`BaseChunker`
"""

from .chunking import BaseChunker, ChunkingEngine, MarkdownChunker, LegalChunker, CodeChunker, JSONChunker, PlainTextChunker, SQLDumpChunker, split_text
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
from .exceptions import (
    ChunkingError,
    ConfigurationError,
    DocumentLoadError,
    EmbeddingError,
    QueryExpansionError,
    RAGModuleError,
    RerankingError,
    VectorStoreConnectionError,
    VectorStoreError,
)
from .models import (
    Chunk,
    ChunkRole,
    ChunkType,
    DocumentType,
    FusedCandidate,
    RetrievalResult,
    ScoredChunk,
    SparseVector,
    TemporalFilter,
)
from .retrieval import (
    AnthropicLLMClient,
    BaseLLMClient,
    BaseReranker,
    BGEReranker,
    CohereReranker,
    FastEmbedReranker,
    OllamaLLMClient,
    QueryExpander,
    RerankItem,
    RetrievalPipeline,
    reciprocal_rank_fusion,
)
from .service import AdvancedRAGModule
from .vector_store import BaseVectorStore, ChunkPoint, QdrantVectorStore

__version__ = "1.0.0"

__all__ = [
    "AdvancedRAGModule",
    "AnthropicLLMClient",
    "BGEM3DenseView",
    "BGEM3Embedder",
    "BGEM3SparseView",
    "BGEReranker",
    "BaseChunker",
    "BaseDenseEmbedder",
    "BaseLLMClient",
    "BaseReranker",
    "BaseSparseEmbedder",
    "BaseVectorStore",
    "Chunk",
    "ChunkPoint",
    "ChunkRole",
    "ChunkType",
    "ChunkingEngine",
    "ChunkingError",
    "CodeChunker",
    "CohereDenseEmbedder",
    "CohereReranker",
    "ConfigurationError",
    "DocumentLoadError",
    "DocumentType",
    "EmbeddingError",
    "FastEmbedBM25SparseEmbedder",
    "FastEmbedDenseEmbedder",
    "FastEmbedReranker",
    "FusedCandidate",
    "JSONChunker",
    "LegalChunker",
    "MarkdownChunker",
    "OllamaLLMClient",
    "PlainTextChunker",
    "QdrantVectorStore",
    "QueryExpander",
    "QueryExpansionError",
    "RAGModuleError",
    "RAGSettings",
    "RerankItem",
    "RerankingError",
    "RetrievalPipeline",
    "RetrievalResult",
    "SQLDumpChunker",
    "ScoredChunk",
    "SparseVector",
    "TemporalFilter",
    "VectorStoreConnectionError",
    "VectorStoreError",
    "reciprocal_rank_fusion",
    "split_text",
]
