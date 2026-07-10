"""Integrationstest des ECHTEN lokalen Stacks – keine API-Keys, keine Injection.

Baut das AdvancedRAGModule ausschließlich aus Settings auf (kompletter
Builder-Pfad): fastembed-Dense-Embedder, BM25-Sparse-Embedder und lokaler
Cross-Encoder-Reranker gegen eine eingebettete Qdrant-Instanz.

Es werden bewusst kleine Modelle verwendet (~170 MB Gesamt-Download beim ersten
Lauf, danach aus dem Hugging-Face-Cache), um den Test praktikabel zu halten;
die Produktions-Defaults (multilingual-e5-large, bge-reranker-base) nutzen
exakt dieselben Code-Pfade.

Start: python tests/test_local_stack.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_module import AdvancedRAGModule, RAGSettings  # noqa: E402

HANDBOOK = """# Employee Handbook

## Working Hours

Regular working time is 40 hours per week. Flexible hours are possible
between 7 am and 8 pm. Overtime is compensated with time off.

## Termination

The notice period is four weeks to the end of the month. After five years
of employment it increases to three months.

| Tenure | Notice period |
| --- | --- |
| under 2 years | 4 weeks |
| 2 to 5 years | 2 months |
| over 5 years | 3 months |

## Vacation

The vacation entitlement is 30 days per calendar year. Remaining vacation
expires on March 31 of the following year.
"""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def main() -> None:
    print("== Integrationstest: echter lokaler Stack (fastembed + BM25 + Cross-Encoder) ==")
    settings = RAGSettings(
        qdrant_url=":memory:",
        collection_name="local_stack_test",
        # Kleine englische Modelle für schnellen Download; Code-Pfade identisch
        # zu den Produktions-Defaults.
        fastembed_dense_model="sentence-transformers/all-MiniLM-L6-v2",
        fastembed_rerank_model="Xenova/ms-marco-MiniLM-L-6-v2",
        expansion_backend="none",
        per_query_limit=20,
        candidate_pool_size=20,
    )
    rag = AdvancedRAGModule(settings=settings)  # KEINE Injection: voller Builder-Pfad
    try:
        with tempfile.TemporaryDirectory(prefix="rag_local_") as tmp:
            handbook = Path(tmp) / "handbook.md"
            handbook.write_text(HANDBOOK, encoding="utf-8")
            document_id = await rag.ingest_document(
                str(handbook), "markdown", {"document_id": "handbook", "tenant": "local"}
            )
            _check(document_id == "handbook", f"document_id falsch: {document_id}")
            print("  Ingestion mit echten lokalen Embeddings OK")

            results = await rag.retrieve("What is the notice period?", limit=3)
            _check(bool(results), "Retrieval lieferte keine Ergebnisse")
            _check(
                results[0]["score_origin"] == "rerank",
                "Der lokale Cross-Encoder-Reranker wurde nicht angewendet",
            )
            combined = " ".join(result["content"] for result in results)
            _check(
                "notice period" in combined.lower(),
                f"Kein Treffer zur Kündigungsfrist in den Top-3: {combined[:200]}",
            )
            print(f"  Retrieval OK – Top-Score {results[0]['score']:.4f} "
                  f"[{results[0]['chunk_type']}] {' > '.join(results[0]['hierarchy'])}")

            table_results = await rag.retrieve("notice period for tenure over 5 years", limit=5)
            _check(
                any(result["chunk_type"] == "table" for result in table_results),
                "Die Fristen-Tabelle wurde vom lokalen Stack nicht gefunden",
            )
            print("  Tabellen-Treffer über Hybrid-Suche OK")
    finally:
        await rag.close()
    print("\nLOKALER STACK VERIFIZIERT ✔")


if __name__ == "__main__":
    asyncio.run(main())
