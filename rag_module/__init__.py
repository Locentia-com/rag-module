"""Modulares, hochpräzises RAG-Modul für die Einbindung als eigenständiger Service.

Öffentliche API:
    - :class:`AdvancedRAGModule` – Service-Fassade (Ingestion + Retrieval)
    - :class:`RAGSettings` – Konfiguration (Env-Präfix ``RAG_``)
    - Interfaces für Dependency Injection: :class:`BaseVectorStore`,
      :class:`BaseDenseEmbedder`, :class:`BaseSparseEmbedder`,
      :class:`BaseReranker`, :class:`BaseLLMClient`, :class:`BaseChunker`
"""

from .chunking import (
    BaseChunker,
    ChunkingEngine,
    CodeChunker,
    JSONChunker,
    LegalChunker,
    MarkdownChunker,
    PlainTextChunker,
    SQLDumpChunker,
    split_text,
)
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
from .utils import (
    BaseTokenCounter,
    HeuristicTokenCounter,
    HFTokenCounter,
    configure_token_counter,
    estimate_tokens,
)
from .vector_store import BaseVectorStore, ChunkPoint, QdrantVectorStore

__version__ = "1.1.0"

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
    "BaseTokenCounter",
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
    "HFTokenCounter",
    "HeuristicTokenCounter",
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
    "configure_token_counter",
    "estimate_tokens",
    "reciprocal_rank_fusion",
    "split_text",
]
