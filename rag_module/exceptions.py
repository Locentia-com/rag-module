"""Zentrale Exception-Hierarchie des RAG-Moduls.

Alle Fehler des Moduls erben von :class:`RAGModuleError`, sodass Host-
Applikationen mit einem einzigen ``except RAGModuleError`` sämtliche
Modul-Fehler abfangen können, ohne fremde Exceptions zu verschlucken.
"""

from __future__ import annotations


class RAGModuleError(Exception):
    """Basisklasse für alle Fehler des RAG-Moduls."""


class ConfigurationError(RAGModuleError):
    """Fehlende oder inkonsistente Konfiguration (API-Keys, Pakete, Dimensionen)."""


class DocumentLoadError(RAGModuleError):
    """Eine Quelldatei konnte nicht gelesen oder geparst werden."""


class ChunkingError(RAGModuleError):
    """Die Chunking-Engine konnte aus dem Dokument keine validen Chunks erzeugen."""


class EmbeddingError(RAGModuleError):
    """Embedding-Erzeugung (dense oder sparse) ist endgültig fehlgeschlagen."""


class VectorStoreError(RAGModuleError):
    """Nicht-transienter Fehler der Vektordatenbank (z. B. ungültiger Filter)."""


class VectorStoreConnectionError(VectorStoreError):
    """Transienter Verbindungs-/Timeout-Fehler der Vektordatenbank nach allen Retries."""


class QueryExpansionError(RAGModuleError):
    """Query-Expansion via LLM ist fehlgeschlagen (wird intern fail-open behandelt)."""


class RerankingError(RAGModuleError):
    """Re-Ranking via externer API ist endgültig fehlgeschlagen."""
