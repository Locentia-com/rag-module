# rag-module

Modulares, hochpräzises RAG-Modul (Retrieval-Augmented Generation) als eigenständiger
Service-Baustein. Arbeitsprinzip: **Raw Query + Metadaten-Filter REIN → Top-N reranked
Chunks RAUS.** Kontext- und Chat-Management (Memory) verbleiben vollständig in der
Host-Applikation. Optimierungsziel ist maximale Akkuratheit und strukturelle
Korrektheit – nicht minimale Latenz.

**Local-first:** Der Default-Stack läuft vollständig ohne externe API-Keys –
Dense-Embeddings, BM25 und Cross-Encoder-Reranking laufen lokal via ONNX
(fastembed). Externe Dienste (Cohere, Anthropic) und lokale Torch-Modelle
(BGE-M3) sind optionale Andockstellen, die per Konfigurationsschalter
(`RAG_*_BACKEND`) oder Dependency Injection aktiviert werden.

## Architektur

```
                         ┌─────────────────────────────────────────────┐
                         │              AdvancedRAGModule              │
                         │       (service.py – Fassade, DI, Locks)     │
                         └─────────┬──────────────────────┬────────────┘
              ingest_document(...) │                      │ retrieve(...)
                                   ▼                      ▼
        ┌──────────────────────────────────┐   ┌──────────────────────────────────┐
        │        ChunkingEngine            │   │       RetrievalPipeline          │
        │        (chunking.py)             │   │       (retrieval.py)             │
        │  Markdown │ Legal │ Code │ JSON  │   │ 1. Query Expansion (Ollama/      │
        │  SQL-Dump │ PDF (+ Tabellen)     │   │    Anthropic, fail-open)         │
        │  Parent-Child + Tabellen-        │   │ 2. Async Hybrid Search           │
        │  Isolation                       │   │    (dense ∥ sparse, je Variante) │
        └──────────┬───────────────────────┘   │ 3. Reciprocal Rank Fusion        │
                   │ Chunks                    │ 4. Re-Ranking (Top-50 → Top-N):  │
                   ▼                           │    fastembed │ BGE │ Cohere      │
        ┌──────────────────────────────────┐   └───────────┬──────────────────────┘
        │  BaseDenseEmbedder /             │               │
        │  BaseSparseEmbedder              │               │
        │  (embeddings.py)                 │               │
        │  fastembed │ BGE-M3 │ Cohere v3  │               │
        │  BM25 (sparse)                   │               │
        └──────────┬───────────────────────┘               │
                   │ Vektoren                              │ Suchen/Fetch
                   ▼                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │        BaseVectorStore (Repository Pattern)                 │
        │        QdrantVectorStore (vector_store.py)                  │
        │  Named Vectors: dense (cosine) + sparse (IDF/BM25)          │
        │  Temporal-Payload: version, valid_from, valid_to, is_active │
        └─────────────────────────────────────────────────────────────┘
```

### Modul-Aufteilung

| Modul | Verantwortung |
|---|---|
| `service.py` | `AdvancedRAGModule`: Fassade, Dependency Injection, Versionsübergänge unter Dokument-Locks |
| `chunking.py` | Layout-bewusste Chunking-Engine für alle Dokumenttypen |
| `vector_store.py` | `BaseVectorStore`-Interface + Qdrant-Implementierung (Hybrid, Temporal-Filter) |
| `retrieval.py` | Query Expansion, Hybrid-Suche, RRF, Cohere-Reranker, `BaseLLMClient` |
| `embeddings.py` | Dense-/Sparse-Embedder-Interfaces + Cohere v3, BGE-M3, FastEmbed-BM25 |
| `models.py` | Chunk-, Filter- und Ergebnis-Datenmodelle |
| `config.py` | `RAGSettings` (Env-Präfix `RAG_`, `.env`-Unterstützung) |
| `utils.py` | Exponential-Backoff-Retry, Token-Schätzung |
| `exceptions.py` | Fehlerhierarchie unter `RAGModuleError` |

### Chunking-Strategien

- **Tabellen** (Markdown, HTML, PDF-Extraktion) werden als **ein** Chunk isoliert –
  inklusive Breadcrumb der Überschriften-Hierarchie und dem letzten Absatz davor als
  übergeordnetem Textkontext. Nur extrem große Tabellen (> `table_hard_max_tokens`)
  werden zeilenweise geteilt, wobei die Kopfzeilen in jedem Teil wiederholt werden.
- **Verträge/Gesetze** (`legal`): Segmentierung nach `§`, `Artikel`, `Abschnitt`,
  nummerierten Klauseln (`3.2 Haftung`) und Markdown-Überschriften. Pro Sektion
  entsteht ein **Parent-Chunk** (ganze Sektion, gekappt) plus **Child-Chunks** entlang
  der Absatz-Marker `(1)`, `(2)` …
- **Code**: Python exakt per AST (Funktionen, Klassen, Dekoratoren; große Klassen →
  Parent + Methoden-Children), C-artige Sprachen per Signatur-Regex + string-/
  kommentar-bewusstem Brace-Matching.
- **JSON / API-Payloads**: Jeder Chunk ist ein *valides* JSON-Fragment mit JSON-Pfad;
  NDJSON wird erkannt; ein Parent-Chunk trägt die Strukturübersicht.
- **SQL-Dumps**: `CREATE TABLE` + zugehörige `ALTER`/`INDEX`/`COMMENT` bleiben als
  Schema-Chunk zusammen (zugleich Parent); `INSERT`s und `COPY … FROM stdin`-Blöcke
  werden pro Tabelle gebatcht, mit wiederholtem Header.
- **PDF**: Text- und Tabellen-Extraktion via `pdfplumber`; Fließtext läuft durch den
  Legal- (bei `document_type="legal"` oder `metadata.domain="legal"`) bzw.
  Plain-Text-Chunker mit Seitenmarkern.

Parent-Chunks sind `searchable=false`: Sie werden nicht durchsucht, sondern beim
Retrieval als `parent_content` zu den Treffern nachgeladen (Parent-Child-Chunking).

### Temporale Versionierung

Jeder Punkt trägt `version`, `valid_from`, `valid_to`, `is_active`. Ein erneuter
Ingest mit derselben `document_id` löscht **nichts**: Der alte Stand wird auf
`is_active=false` gesetzt und erhält `valid_to = Ingest-Zeitpunkt`; der neue Stand
wird mit inkrementierter `version` und `is_active=true` indiziert. Abfragen:

```python
await rag.retrieve("...")                                             # aktueller Stand
await rag.retrieve("...", temporal_filter={"as_of": "2026-01-01T00:00:00Z"})  # Stand zum Zeitpunkt
await rag.retrieve("...", temporal_filter={"version": 2})             # exakte Version
await rag.retrieve("...", temporal_filter={"include_inactive": True}) # alle Stände
```

### Backends: lokal per Default, extern als Andockstelle

| Stufe | Lokal (Default, kein Key) | Lokal (max. Qualität, Torch) | Extern (API-Key) |
|---|---|---|---|
| Dense-Embeddings | `fastembed` → `intfloat/multilingual-e5-large` (ONNX, 1024 Dim.) | `bge_m3` → BGE-M3 (FlagEmbedding) | `cohere` → embed-multilingual-v3.0 |
| Sparse/BM25 | `fastembed_bm25` → `Qdrant/bm25` + IDF-Modifier | `bge_m3` (lexical weights) | – |
| Re-Ranking | `fastembed` → `BAAI/bge-reranker-base` (ONNX-Cross-Encoder) | `bge` → `BAAI/bge-reranker-v2-m3` | `cohere` → rerank-v3.5 |
| Query Expansion | `ollama` → lokales LLM (z. B. `llama3.2`) | – | `anthropic` → Claude Haiku |

Auswahl über `RAG_DENSE_BACKEND`, `RAG_SPARSE_BACKEND`, `RAG_RERANK_BACKEND`,
`RAG_EXPANSION_BACKEND` (Default für Expansion: `none`). Die Modellgewichte der
lokalen Backends werden beim ersten Start automatisch von Hugging Face geladen
und danach lokal gecacht; für vollständig air-gapped Betrieb den HF-Cache
vorbefüllen. Darüber hinaus lässt sich jede Stufe per Dependency Injection durch
eigene Implementierungen ersetzen (`BaseDenseEmbedder`, `BaseSparseEmbedder`,
`BaseReranker`, `BaseLLMClient`, `BaseVectorStore`).

### Token-Zählung

Chunk-Budgets werden per Default über eine Zeichen-Heuristik (~4 Zeichen/Token)
geprüft — modellagnostisch und ohne Download. Für exakte Budgets:
`RAG_TOKENIZER_BACKEND=hf` zählt mit dem HuggingFace-Tokenizer des
Dense-Embedding-Modells (überschreibbar via `RAG_TOKENIZER_MODEL`). Fail-closed:
Lädt der konfigurierte Tokenizer nicht (z. B. air-gapped ohne vorbefüllten
HF-Cache), wirft die Engine `ConfigurationError` statt still auf die Heuristik
zurückzufallen.

### Pflicht-Filter (z. B. Mandanten-Isolation)

`RAG_REQUIRED_FILTER_KEYS='["tenant"]'` erzwingt, dass jeder `retrieve()`-Aufruf
die genannten Schlüssel im `metadata_filter` setzt — fehlen sie, gibt es einen
`ValueError` statt einer mandantenübergreifenden Trefferliste. Damit wird die
Filter-Konvention der Host-Applikation erzwingbar statt vertrauensbasiert.

### Robustheit

Alle externen Aufrufe (Qdrant, optionale APIs) laufen über Exponential Backoff
mit Full Jitter (`utils.retry_async`); transiente Fehler (Timeouts, 429, 5xx,
Verbindungsabbrüche) werden erneut versucht, permanente Fehler sofort gemeldet.
Query Expansion ist fail-open (Ausfall ⇒ nur Original-Query); das Re-Ranking ist
per Default fail-closed (`rerank_fail_open=false` wirft `RerankingError`), damit
ein stiller Präzisionsverlust nicht unbemerkt bleibt.

## Installation

```bash
pip install .            # lokaler Default-Stack (fastembed, kein API-Key nötig)
pip install .[pdf]       # + PDF-Verarbeitung (pdfplumber)
pip install .[bge]       # + BGE-M3 / bge-reranker-v2-m3 (FlagEmbedding, inkl. Torch)
pip install .[cohere]    # + Andockstelle Cohere (Embeddings/Rerank)
pip install .[anthropic] # + Andockstelle Anthropic (Query Expansion)
```

Qdrant lokal starten:

```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Konfiguration über Env-Vars (`RAG_*`) oder `.env` – siehe [.env.example](.env.example).

## Verwendung

```python
import asyncio
from rag_module import AdvancedRAGModule, RAGSettings

async def main() -> None:
    rag = AdvancedRAGModule(settings=RAGSettings())
    try:
        doc_id = await rag.ingest_document(
            "verträge/rahmenvertrag.pdf",
            document_type="legal",
            metadata={"document_id": "rahmenvertrag-acme", "tenant": "acme"},
        )
        results = await rag.retrieve(
            "Welche Kündigungsfrist gilt für den Rahmenvertrag?",
            limit=5,
            metadata_filter={"tenant": "acme"},
        )
        for r in results:
            print(r["score"], r["hierarchy"], r["content"][:120])
    finally:
        await rag.close()

asyncio.run(main())
```

Liegt der Inhalt bereits als String vor (z. B. aus einer API oder Datenbank),
übernimmt `ingest_text` denselben Ablauf ohne Datei — inklusive Versionierung
über `metadata["document_id"]` (`pdf` ist ausgenommen, da binär):

```python
await rag.ingest_text(
    "# Handbuch\n\n…",
    document_type="markdown",
    metadata={"document_id": "handbuch", "tenant": "acme"},
    source_name="handbuch.md",
)
```

Für Tests/lokale Entwicklung ohne Server: `QdrantVectorStore(url=":memory:", ...)`
startet eine eingebettete Qdrant-Instanz.

## Tests & CI

```bash
pip install -e .[dev]
ruff check rag_module tests   # Lint
pytest                        # Offline-Suite (Hash-Embedder + In-Memory-Qdrant, keine Downloads)
python tests/test_offline_e2e.py   # Ausführliches Standalone-E2E-Skript
```

Der GitHub-Actions-Workflow (`.github/workflows/ci.yml`) führt Lint, die
pytest-Suite und das Offline-E2E-Skript bei jedem Push/PR aus — vollständig
offline, ohne API-Keys.

## Erweiterungspunkte (Dependency Injection)

Alle Komponenten sind über Interfaces austauschbar und werden dem Konstruktor von
`AdvancedRAGModule` injiziert: `BaseVectorStore` (andere Vektordatenbank),
`BaseDenseEmbedder`/`BaseSparseEmbedder` (andere Embedding-Modelle),
`BaseReranker` (anderer Reranker), `BaseLLMClient` (anderes Expansions-LLM),
`BaseChunker`/`ChunkingEngine` (eigene Chunking-Strategien).
