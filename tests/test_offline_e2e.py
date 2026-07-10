"""Offline-End-to-End-Test des RAG-Moduls – ohne API-Keys, ohne laufenden Qdrant.

Abgedeckt:
    1. Chunking: Markdown (Tabellen-Isolation, Parent-Child), Legal (§/Absätze),
       Python-Code (AST), JSON (valide Fragmente), SQL-Dump (Schema + Insert-Batches).
    2. Reciprocal Rank Fusion (deterministische Fusion).
    3. Voller Service-Durchlauf gegen eine eingebettete Qdrant-Instanz (":memory:"):
       Ingestion, Hybrid-Retrieval, Update mit Versionierung, temporale Filter
       (as_of / version / include_inactive), Parent-Kontext-Anreicherung.

Start: python tests/test_offline_e2e.py
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_module import (  # noqa: E402
    AdvancedRAGModule,
    BaseDenseEmbedder,
    BaseReranker,
    BaseSparseEmbedder,
    ChunkRole,
    ChunkType,
    ChunkingEngine,
    DocumentType,
    QdrantVectorStore,
    RAGSettings,
    RerankItem,
    ScoredChunk,
    SparseVector,
    reciprocal_rank_fusion,
)

_TOKEN_RE = re.compile(r"[\wäöüß]+", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _stable_hash(token: str) -> int:
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:4], "big")


class HashDenseEmbedder(BaseDenseEmbedder):
    """Deterministischer Bag-of-Words-Embedder (Token-Hashing in 64 Dimensionen)."""

    _DIM = 64

    @property
    def dimension(self) -> int:
        return self._DIM

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._DIM
        for token in _tokens(text):
            vector[_stable_hash(token) % self._DIM] += 1.0
        norm = math.sqrt(sum(component * component for component in vector))
        return [component / norm for component in vector] if norm else [0.0] * self._DIM

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


class HashSparseEmbedder(BaseSparseEmbedder):
    """Deterministischer Sparse-Embedder (Termfrequenzen auf gehashte Indizes)."""

    def _vector(self, text: str) -> SparseVector:
        frequencies: dict[int, float] = {}
        for token in _tokens(text):
            index = _stable_hash(token) % 1_000_000
            frequencies[index] = frequencies.get(index, 0.0) + 1.0
        indices = sorted(frequencies)
        return SparseVector(indices=indices, values=[frequencies[i] for i in indices])

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self._vector(text) for text in texts]

    async def embed_queries(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self._vector(text) for text in texts]


class OverlapReranker(BaseReranker):
    """Deterministischer Reranker: Score = Anteil der Query-Tokens im Dokument."""

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankItem]:
        query_tokens = set(_tokens(query))
        scored = []
        for index, document in enumerate(documents):
            document_tokens = set(_tokens(document))
            overlap = len(query_tokens & document_tokens) / max(1, len(query_tokens))
            scored.append(RerankItem(index=index, relevance_score=overlap))
        scored.sort(key=lambda item: (-item.relevance_score, item.index))
        return scored[:top_n]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MARKDOWN_V1 = """# Mitarbeiterhandbuch

## Arbeitszeiten

Die reguläre Arbeitszeit beträgt 40 Stunden pro Woche. Gleitzeit ist zwischen
7 und 20 Uhr möglich. Überstunden werden durch Freizeit ausgeglichen.

## Kündigung

Die Kündigungsfrist beträgt vier Wochen zum Monatsende. Nach fünf Jahren
Betriebszugehörigkeit verlängert sie sich auf drei Monate.

Details regelt die folgende Übersicht der Fristen nach Betriebszugehörigkeit.

| Betriebszugehörigkeit | Kündigungsfrist |
| --- | --- |
| unter 2 Jahre | 4 Wochen zum Monatsende |
| 2 bis 5 Jahre | 2 Monate zum Monatsende |
| über 5 Jahre | 3 Monate zum Monatsende |

## Urlaub

Der Urlaubsanspruch beträgt 30 Tage pro Kalenderjahr. Resturlaub verfällt
am 31. März des Folgejahres.
"""

MARKDOWN_V2 = MARKDOWN_V1.replace(
    "Die Kündigungsfrist beträgt vier Wochen zum Monatsende.",
    "Die Kündigungsfrist beträgt sechs Wochen zum Quartalsende.",
).replace("30 Tage pro Kalenderjahr", "32 Tage pro Kalenderjahr")

LEGAL_TEXT = """Rahmenvertrag über Softwareentwicklung

§ 1 Vertragsgegenstand
(1) Der Auftragnehmer erbringt Softwareentwicklungsleistungen nach Maßgabe der
Leistungsbeschreibung in Anlage 1.
(2) Nebenleistungen sind nur geschuldet, wenn sie ausdrücklich vereinbart wurden.

§ 2 Vergütung
(1) Die Vergütung erfolgt nach Aufwand zu einem Stundensatz von 150 Euro.
(2) Reisekosten werden nur nach vorheriger schriftlicher Zustimmung erstattet.
(3) Rechnungen sind innerhalb von 30 Tagen ohne Abzug zahlbar.

§ 3 Haftung
(1) Der Auftragnehmer haftet unbeschränkt für Vorsatz und grobe Fahrlässigkeit.
(2) Bei einfacher Fahrlässigkeit haftet er nur für die Verletzung wesentlicher
Vertragspflichten, begrenzt auf den vertragstypischen vorhersehbaren Schaden.
"""

PYTHON_CODE = '''"""Beispielmodul für den Code-Chunker."""

import math
from typing import Iterable

DEFAULT_FACTOR = 2.5


def scale(values: Iterable[float], factor: float = DEFAULT_FACTOR) -> list[float]:
    """Skaliert eine Zahlenreihe."""
    return [value * factor for value in values]


class GeometryHelper:
    """Sammlung geometrischer Hilfsfunktionen."""

    def circle_area(self, radius: float) -> float:
        return math.pi * radius ** 2

    def circle_circumference(self, radius: float) -> float:
        return 2 * math.pi * radius
'''

JSON_PAYLOAD = {
    "service": "billing-api",
    "endpoint": "/v2/invoices",
    "customers": [
        {"id": index, "name": f"Kunde {index}", "plan": "enterprise" if index % 2 else "starter",
         "notes": f"Langjähriger Bestandskunde Nummer {index} mit individuellen Konditionen."}
        for index in range(60)
    ],
    "pagination": {"page": 1, "per_page": 60, "total": 60},
}

SQL_DUMP = """-- Beispiel-Dump
BEGIN;

CREATE TABLE kunden (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    plan TEXT NOT NULL DEFAULT 'starter'
);

CREATE INDEX idx_kunden_plan ON kunden (plan);

""" + "\n".join(
    f"INSERT INTO kunden (name, plan) VALUES ('Kunde {index}', 'enterprise');"
    for index in range(80)
) + """

COMMIT;
"""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# 1. Chunking-Tests
# ---------------------------------------------------------------------------


def test_chunking(engine: ChunkingEngine, workdir: Path) -> None:
    # --- Markdown: Tabellen-Isolation + Parent-Child ---
    markdown_file = workdir / "handbuch.md"
    markdown_file.write_text(MARKDOWN_V1, encoding="utf-8")
    chunks = engine.chunk_file_sync(markdown_file, DocumentType.MARKDOWN, {})
    tables = [chunk for chunk in chunks if chunk.chunk_type is ChunkType.TABLE]
    _check(len(tables) == 1, f"Erwartet 1 Tabellen-Chunk, gefunden {len(tables)}")
    table = tables[0]
    _check(
        "unter 2 Jahre" in table.content and "über 5 Jahre" in table.content,
        "Tabelle wurde nicht als Ganzes isoliert",
    )
    _check(
        "Übersicht der Fristen" in table.content,
        "Tabellen-Chunk enthält den übergeordneten Textkontext nicht",
    )
    _check(
        any("Kündigung" in part for part in table.hierarchy),
        f"Tabellen-Chunk trägt falsche Hierarchie: {table.hierarchy}",
    )
    parent_ids = {chunk.chunk_id for chunk in chunks if chunk.role is ChunkRole.PARENT}
    children = [chunk for chunk in chunks if chunk.role is ChunkRole.CHILD]
    _check(
        all(child.parent_id in parent_ids for child in children),
        "Mindestens ein Child-Chunk verweist auf einen unbekannten Parent",
    )
    _check(
        all(not chunk.searchable for chunk in chunks if chunk.role is ChunkRole.PARENT),
        "Parent-Chunks müssen searchable=False sein",
    )
    sequences = [chunk.sequence for chunk in chunks]
    _check(sequences == list(range(len(chunks))), "Sequenznummern sind nicht fortlaufend")
    print(f"  Markdown: {len(chunks)} Chunks, Tabellen-Isolation OK")

    # --- Legal: §-Segmentierung ---
    legal_file = workdir / "vertrag.txt"
    legal_file.write_text(LEGAL_TEXT, encoding="utf-8")
    legal_chunks = engine.chunk_file_sync(legal_file, DocumentType.LEGAL, {})
    hierarchies = {" > ".join(chunk.hierarchy) for chunk in legal_chunks}
    for paragraph in ("§ 1", "§ 2", "§ 3"):
        _check(
            any(paragraph in hierarchy for hierarchy in hierarchies),
            f"{paragraph} fehlt in den Legal-Hierarchien: {hierarchies}",
        )
    haftung = [chunk for chunk in legal_chunks if any("§ 3" in h for h in chunk.hierarchy)]
    _check(
        any("grobe Fahrlässigkeit" in chunk.content for chunk in haftung),
        "Inhalt von § 3 wurde nicht korrekt segmentiert",
    )
    print(f"  Legal: {len(legal_chunks)} Chunks, §-Hierarchie OK")

    # --- Python-Code: AST-Segmentierung ---
    code_file = workdir / "geometry.py"
    code_file.write_text(PYTHON_CODE, encoding="utf-8")
    code_chunks = engine.chunk_file_sync(code_file, DocumentType.CODE, {})
    qualnames = {chunk.extra.get("qualname") for chunk in code_chunks}
    _check("scale" in qualnames, f"Funktion 'scale' fehlt: {qualnames}")
    _check("GeometryHelper" in qualnames, f"Klasse 'GeometryHelper' fehlt: {qualnames}")
    scale_chunk = next(chunk for chunk in code_chunks if chunk.extra.get("qualname") == "scale")
    _check(
        "def scale(" in scale_chunk.content and "return [value * factor" in scale_chunk.content,
        "Funktions-Chunk enthält nicht die vollständige Definition",
    )
    print(f"  Code: {len(code_chunks)} Chunks, AST-Einheiten OK")

    # --- JSON: valide Fragmente ---
    import json as json_module

    json_file = workdir / "payload.json"
    json_file.write_text(json_module.dumps(JSON_PAYLOAD, ensure_ascii=False), encoding="utf-8")
    json_chunks = engine.chunk_file_sync(json_file, DocumentType.API_PAYLOAD, {})
    fragments = [chunk for chunk in json_chunks if chunk.chunk_type is ChunkType.JSON_FRAGMENT]
    _check(len(fragments) >= 2, "Großes JSON hätte in mehrere Fragmente zerlegt werden müssen")
    for fragment in fragments:
        first_line, _, body = fragment.content.partition("\n")
        _check(first_line.startswith("JSON-Pfad: "), "JSON-Fragment ohne Pfad-Kontextzeile")
        json_module.loads(body)  # wirft, wenn das Fragment kein valides JSON ist
    print(f"  JSON: {len(fragments)} valide Fragmente OK")

    # --- SQL-Dump: Schema + Insert-Batches ---
    sql_file = workdir / "dump.sql"
    sql_file.write_text(SQL_DUMP, encoding="utf-8")
    sql_chunks = engine.chunk_file_sync(sql_file, DocumentType.SQL_DUMP, {})
    schema_chunks = [chunk for chunk in sql_chunks if chunk.extra.get("kind") == "schema"]
    insert_chunks = [chunk for chunk in sql_chunks if chunk.extra.get("kind") == "insert_batch"]
    _check(len(schema_chunks) == 1, f"Erwartet 1 Schema-Chunk, gefunden {len(schema_chunks)}")
    _check(
        "CREATE TABLE kunden" in schema_chunks[0].content
        and "idx_kunden_plan" in schema_chunks[0].content,
        "Schema-Chunk hält CREATE TABLE und Index nicht zusammen",
    )
    _check(len(insert_chunks) >= 2, "INSERTs hätten in mehrere Batches gebündelt werden müssen")
    _check(
        all(chunk.parent_id == schema_chunks[0].chunk_id for chunk in insert_chunks),
        "Insert-Batches verweisen nicht auf den Schema-Parent",
    )
    total_inserts = sum(chunk.extra["statement_count"] for chunk in insert_chunks)
    _check(total_inserts == 80, f"Es gingen INSERTs verloren: {total_inserts}/80")
    print(f"  SQL: Schema + {len(insert_chunks)} Insert-Batches ({total_inserts} Statements) OK")


# ---------------------------------------------------------------------------
# 2. RRF-Test
# ---------------------------------------------------------------------------


def test_rrf() -> None:
    def scored(pid: str) -> ScoredChunk:
        return ScoredChunk(point_id=pid, score=0.0, payload={"content": pid})

    ranking_a = [scored("a"), scored("b"), scored("c")]
    ranking_b = [scored("b"), scored("a"), scored("d")]
    fused = reciprocal_rank_fusion([ranking_a, ranking_b], k=60)
    order = [candidate.point_id for candidate in fused]
    # a und b haben identische RRF-Summen (Rang 0+1); Tie-Break: point_id.
    _check(set(order[:2]) == {"a", "b"}, f"RRF-Topplätze falsch: {order}")
    _check(order[2:] == ["c", "d"], f"RRF-Reihenfolge falsch: {order}")
    expected = 1 / 61 + 1 / 62
    _check(abs(fused[0].rrf_score - expected) < 1e-12, "RRF-Score-Formel falsch")
    _check(fused[0].hit_count == 2 and fused[-1].hit_count == 1, "RRF-Hit-Counts falsch")
    print("  RRF: Fusion, Scores und Tie-Breaks OK")


# ---------------------------------------------------------------------------
# 3. Service-E2E (In-Memory-Qdrant)
# ---------------------------------------------------------------------------


async def test_service_e2e(workdir: Path) -> None:
    settings = RAGSettings(
        expansion_backend="none",
        per_query_limit=25,
        candidate_pool_size=25,
    )
    store = QdrantVectorStore(url=":memory:", collection_name="test_chunks")
    rag = AdvancedRAGModule(
        settings=settings,
        vector_store=store,
        dense_embedder=HashDenseEmbedder(),
        sparse_embedder=HashSparseEmbedder(),
        reranker=OverlapReranker(),
    )
    try:
        # --- Version 1 ingestieren ---
        markdown_file = workdir / "handbuch.md"
        markdown_file.write_text(MARKDOWN_V1, encoding="utf-8")
        document_id = await rag.ingest_document(
            str(markdown_file), "markdown", {"document_id": "handbuch", "tenant": "acme"}
        )
        _check(document_id == "handbuch", f"document_id falsch: {document_id}")

        results = await rag.retrieve("Welche Kündigungsfrist gilt?", limit=5)
        _check(bool(results), "Retrieval (v1) lieferte keine Ergebnisse")
        top = results[0]
        _check(
            "Kündigungsfrist" in top["content"],
            f"Top-Treffer (v1) passt nicht zur Query: {top['content'][:100]}",
        )
        _check(top["version"] == 1 and top["is_active"], "Versionsfelder (v1) falsch")
        _check(top["score_origin"] == "rerank", "Reranker wurde nicht angewendet")
        table_hits = [r for r in results if r["chunk_type"] == "table"]
        _check(bool(table_hits), "Die Fristen-Tabelle wurde nicht gefunden")
        child_hits = [r for r in results if r["chunk_role"] == "child"]
        _check(
            all(r["parent_content"] for r in child_hits),
            "Parent-Kontext wurde für Child-Treffer nicht nachgeladen",
        )

        # --- Metadaten-Filter ---
        filtered = await rag.retrieve(
            "Kündigungsfrist", limit=3, metadata_filter={"tenant": "acme"}
        )
        _check(bool(filtered), "Metadaten-Filter (Treffer erwartet) schlug fehl")
        empty = await rag.retrieve(
            "Kündigungsfrist", limit=3, metadata_filter={"tenant": "andere-firma"}
        )
        _check(not empty, "Metadaten-Filter (keine Treffer erwartet) schlug fehl")

        # --- Version 2: Update ohne Löschung ---
        await asyncio.sleep(0.05)  # saubere Trennung der valid_from/valid_to-Zeitstempel
        from rag_module.utils import utc_now

        between = utc_now()
        await asyncio.sleep(0.05)
        markdown_file.write_text(MARKDOWN_V2, encoding="utf-8")
        await rag.ingest_document(
            str(markdown_file), "markdown", {"document_id": "handbuch", "tenant": "acme"}
        )

        history = await rag.get_document_history("handbuch")
        _check(len(history) == 2, f"Erwartet 2 Versionen, gefunden {len(history)}")
        _check(
            not history[0]["is_active"] and history[0]["valid_to"] is not None,
            "v1 wurde nicht korrekt deaktiviert (is_active/valid_to)",
        )
        _check(
            history[1]["is_active"] and history[1]["valid_to"] is None,
            "v2 ist nicht der aktive Stand",
        )

        # Default-Retrieval liefert den NEUEN Stand …
        current = await rag.retrieve("Welche Kündigungsfrist gilt?", limit=5)
        current_text = " ".join(result["content"] for result in current)
        _check("sechs Wochen zum Quartalsende" in current_text, "v2-Inhalt fehlt im Default-Retrieval")
        _check(
            all(result["version"] == 2 for result in current),
            "Default-Retrieval enthält historische Stände",
        )

        # … as_of zwischen den Ingests liefert den ALTEN Stand …
        historical = await rag.retrieve(
            "Welche Kündigungsfrist gilt?",
            limit=5,
            temporal_filter={"as_of": between},
        )
        historical_text = " ".join(result["content"] for result in historical)
        _check(
            "vier Wochen zum Monatsende" in historical_text,
            "as_of-Filter liefert nicht den historischen Stand",
        )
        _check(
            all(result["version"] == 1 for result in historical),
            "as_of-Filter mischt Versionen",
        )

        # … expliziter Versionsfilter ebenso …
        version_one = await rag.retrieve(
            "Urlaubsanspruch Tage", limit=3, temporal_filter={"version": 1}
        )
        _check(
            any("30 Tage" in result["content"] for result in version_one),
            "version=1-Filter liefert nicht den alten Urlaubsanspruch",
        )

        # … und include_inactive sieht beide Stände.
        all_versions = await rag.retrieve(
            "Kündigungsfrist",
            limit=10,
            temporal_filter={"include_inactive": True},
        )
        _check(
            {result["version"] for result in all_versions} == {1, 2},
            "include_inactive liefert nicht beide Versionen",
        )

        print("  Service-E2E: Ingestion, Hybrid-Retrieval, Versionierung, Temporal-Filter OK")
    finally:
        await rag.close()


async def main() -> None:
    print("== Offline-E2E-Test des RAG-Moduls ==")
    settings = RAGSettings(expansion_backend="none")
    engine = ChunkingEngine(settings)
    with tempfile.TemporaryDirectory(prefix="rag_e2e_") as tmp:
        workdir = Path(tmp)
        print("[1/3] Chunking …")
        test_chunking(engine, workdir)
        print("[2/3] Reciprocal Rank Fusion …")
        test_rrf()
        print("[3/3] Service-E2E gegen In-Memory-Qdrant …")
        await test_service_e2e(workdir)
    print("\nALLE TESTS BESTANDEN ✔")


if __name__ == "__main__":
    asyncio.run(main())
