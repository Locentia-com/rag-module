"""Datenmodelle des RAG-Moduls: Chunks, Vektoren, Filter und Retrieval-Ergebnisse."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


class DocumentType(str, enum.Enum):
    """Unterstützte Dokumenttypen; steuern die Auswahl der Chunking-Strategie."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    API_PAYLOAD = "api_payload"
    CODE = "code"
    SQL_DUMP = "sql_dump"
    LEGAL = "legal"
    TEXT = "text"

    @classmethod
    def from_raw(cls, value: str) -> "DocumentType":
        """Normalisiert freie Typangaben (inkl. gängiger Aliase) auf einen DocumentType."""
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "md": cls.MARKDOWN,
            "markdown": cls.MARKDOWN,
            "pdf": cls.PDF,
            "json": cls.JSON,
            "payload": cls.API_PAYLOAD,
            "api_payload": cls.API_PAYLOAD,
            "ndjson": cls.API_PAYLOAD,
            "code": cls.CODE,
            "source": cls.CODE,
            "sql": cls.SQL_DUMP,
            "sql_dump": cls.SQL_DUMP,
            "legal": cls.LEGAL,
            "contract": cls.LEGAL,
            "law": cls.LEGAL,
            "vertrag": cls.LEGAL,
            "gesetz": cls.LEGAL,
            "text": cls.TEXT,
            "txt": cls.TEXT,
            "plain": cls.TEXT,
        }
        if normalized in aliases:
            return aliases[normalized]
        valid = ", ".join(sorted({m.value for m in cls}))
        raise ValueError(
            f"Unbekannter document_type '{value}'. Gültige Werte: {valid} (plus Aliase wie 'md', 'contract', 'sql')."
        )


class ChunkType(str, enum.Enum):
    """Struktureller Typ eines Chunks."""

    TEXT = "text"
    TABLE = "table"
    SECTION = "section"
    CODE_UNIT = "code_unit"
    JSON_FRAGMENT = "json_fragment"
    SQL_STATEMENT = "sql_statement"
    LEGAL_CLAUSE = "legal_clause"


class ChunkRole(str, enum.Enum):
    """Rolle im Parent-Child-Chunking.

    - ``parent``: Kontext-Chunk (ganze Sektion/Klasse/Schema); wird standardmäßig
      NICHT durchsucht, sondern zur Kontext-Anreicherung der Treffer geladen.
    - ``child``: Durchsuchbarer Teil-Chunk mit Verweis auf seinen Parent.
    - ``standalone``: Durchsuchbarer Chunk ohne Parent.
    """

    PARENT = "parent"
    CHILD = "child"
    STANDALONE = "standalone"


@dataclass(slots=True)
class SparseVector:
    """Lexikalischer Sparse-Vektor (BM25 / BGE-M3 lexical weights)."""

    indices: list[int]
    values: list[float]

    def is_empty(self) -> bool:
        return not self.indices


@dataclass(slots=True)
class Chunk:
    """Ein einzelner, von der Chunking-Engine erzeugter Chunk."""

    content: str
    chunk_type: ChunkType
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    role: ChunkRole = ChunkRole.STANDALONE
    parent_id: Optional[str] = None
    hierarchy: list[str] = field(default_factory=list)
    sequence: int = 0
    searchable: bool = True
    token_estimate: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoredChunk:
    """Ein Punkt aus der Vektordatenbank mit Roh-Score einer einzelnen Suche."""

    point_id: str
    score: float
    payload: dict[str, Any]


@dataclass(slots=True)
class FusedCandidate:
    """Kandidat nach Reciprocal Rank Fusion über mehrere Ranking-Listen."""

    point_id: str
    payload: dict[str, Any]
    rrf_score: float
    best_rank: int
    hit_count: int


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Akzeptiert datetime, Unix-Timestamp oder ISO-8601-String (auch mit 'Z')."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        raw = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(raw)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    raise ValueError(f"Nicht interpretierbarer Zeitstempel: {value!r}")


@dataclass(slots=True)
class TemporalFilter:
    """Zeit-/versionsbasierter Filter für historische Datenstände.

    Semantik (in dieser Prioritätsreihenfolge):
    - ``version``: Exakt diese Dokumentversion(en), unabhängig von ``is_active``.
    - ``as_of``: Der Datenstand, der zum Zeitpunkt ``as_of`` gültig war
      (``valid_from <= as_of < valid_to`` bzw. ``valid_to`` offen).
    - ``include_inactive=True``: Alle Stände, auch historische.
    - Default (leerer Filter): Nur der aktuell aktive Stand (``is_active=True``).
    """

    as_of: Optional[datetime] = None
    version: Optional[int] = None
    include_inactive: bool = False

    @classmethod
    def from_value(
        cls, value: "TemporalFilter | dict[str, Any] | None"
    ) -> Optional["TemporalFilter"]:
        """Erzeugt einen TemporalFilter aus dict-Eingaben der Host-Applikation."""
        if value is None or isinstance(value, TemporalFilter):
            return value
        if not isinstance(value, dict):
            raise ValueError(
                f"temporal_filter muss dict oder TemporalFilter sein, nicht {type(value).__name__}."
            )
        unknown = set(value) - {"as_of", "version", "include_inactive"}
        if unknown:
            raise ValueError(
                f"Unbekannte temporal_filter-Schlüssel: {sorted(unknown)}. "
                "Erlaubt: as_of, version, include_inactive."
            )
        version = value.get("version")
        return cls(
            as_of=_parse_datetime(value.get("as_of")),
            version=int(version) if version is not None else None,
            include_inactive=bool(value.get("include_inactive", False)),
        )


@dataclass(slots=True)
class RetrievalResult:
    """Ein finales, rerank-tes Retrieval-Ergebnis für die Host-Applikation."""

    chunk_id: str
    content: str
    score: float
    score_origin: str  # "rerank" oder "rrf"
    rrf_score: float
    chunk_type: str
    chunk_role: str
    document_id: str
    document_type: str
    version: int
    is_active: bool
    valid_from: Optional[str]
    valid_to: Optional[str]
    hierarchy: list[str]
    parent_chunk_id: Optional[str]
    parent_content: Optional[str]
    source: Optional[str]
    metadata: dict[str, Any]
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "score_origin": self.score_origin,
            "rrf_score": self.rrf_score,
            "chunk_type": self.chunk_type,
            "chunk_role": self.chunk_role,
            "document_id": self.document_id,
            "document_type": self.document_type,
            "version": self.version,
            "is_active": self.is_active,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "hierarchy": list(self.hierarchy),
            "parent_chunk_id": self.parent_chunk_id,
            "parent_content": self.parent_content,
            "source": self.source,
            "metadata": dict(self.metadata),
            "extra": dict(self.extra),
        }
