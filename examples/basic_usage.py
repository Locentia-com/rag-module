"""Minimales Nutzungsbeispiel für das RAG-Modul (lokaler Default-Stack, keine API-Keys).

Voraussetzungen:
    - Laufender Qdrant (docker run -p 6333:6333 qdrant/qdrant)
    - Beim ersten Start werden die lokalen Modellgewichte (Dense + Reranker)
      automatisch heruntergeladen und gecacht.

Optionale Andockstellen (Env-Vars):
    - RAG_EXPANSION_BACKEND=ollama              (lokale Query Expansion via Ollama)
    - RAG_DENSE_BACKEND=cohere + RAG_COHERE_API_KEY
    - RAG_RERANK_BACKEND=cohere + RAG_COHERE_API_KEY
    - RAG_EXPANSION_BACKEND=anthropic + RAG_ANTHROPIC_API_KEY

Start: python examples/basic_usage.py <pfad/zur/datei> <document_type>
"""

from __future__ import annotations

import asyncio
import logging
import sys

from rag_module import AdvancedRAGModule, RAGSettings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


async def main(file_path: str, document_type: str) -> None:
    settings = RAGSettings()  # liest RAG_*-Env-Vars bzw. .env
    rag = AdvancedRAGModule(settings=settings)
    try:
        # --- Ingestion (Version 1 bzw. neue Version bei gleicher document_id) ---
        document_id = await rag.ingest_document(
            file_path,
            document_type=document_type,
            metadata={"document_id": "demo-dokument", "tenant": "demo"},
        )
        print(f"Ingestiert als document_id={document_id!r}")
        for version in await rag.get_document_history(document_id):
            print(
                f"  v{version['version']}: aktiv={version['is_active']} "
                f"({version['chunk_count']} Chunks, gültig ab {version['valid_from']})"
            )

        # --- Retrieval: aktueller Stand ---
        results = await rag.retrieve(
            "Welche Fristen und Bedingungen gelten?",
            limit=5,
            metadata_filter={"tenant": "demo"},
        )
        for rank, result in enumerate(results, start=1):
            breadcrumb = " > ".join(result["hierarchy"]) or "(ohne Hierarchie)"
            print(f"\n#{rank} score={result['score']:.4f} [{result['chunk_type']}] {breadcrumb}")
            print(result["content"][:300])

        # --- Retrieval: historischer Stand (Beispiel) ---
        historical = await rag.retrieve(
            "Welche Fristen und Bedingungen gelten?",
            limit=3,
            temporal_filter={"version": 1},
        )
        print(f"\nTreffer im Stand v1: {len(historical)}")
    finally:
        await rag.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Aufruf: python examples/basic_usage.py <datei> <document_type>")
    asyncio.run(main(sys.argv[1], sys.argv[2]))
