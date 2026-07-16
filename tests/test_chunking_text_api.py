"""Tests für die chunk_text-API der ChunkingEngine."""

from __future__ import annotations

import pytest

from rag_module.chunking import ChunkingEngine
from rag_module.config import RAGSettings
from rag_module.exceptions import ChunkingError
from rag_module.models import ChunkType, DocumentType


@pytest.fixture
def engine() -> ChunkingEngine:
    return ChunkingEngine(RAGSettings())


def test_chunk_text_sync_markdown_with_table(engine: ChunkingEngine) -> None:
    content = (
        "# Preise\n\n"
        "Unsere aktuelle Preisliste als Übersicht.\n\n"
        "| Produkt | Preis |\n"
        "| --- | --- |\n"
        "| Widget | 5 € |\n"
        "| Gadget | 9 € |\n"
    )
    chunks = engine.chunk_text_sync(content, DocumentType.MARKDOWN, source_name="p.md")
    table_chunks = [c for c in chunks if c.chunk_type is ChunkType.TABLE]
    assert len(table_chunks) == 1
    assert "| Widget | 5 € |" in table_chunks[0].content
    assert "| Gadget | 9 € |" in table_chunks[0].content


def test_chunk_text_sync_rejects_pdf(engine: ChunkingEngine) -> None:
    with pytest.raises(ChunkingError, match="PDF"):
        engine.chunk_text_sync("egal", DocumentType.PDF)


def test_chunk_text_sync_empty_content_raises(engine: ChunkingEngine) -> None:
    with pytest.raises(ChunkingError, match="keine Chunks"):
        engine.chunk_text_sync("   \n  ", DocumentType.TEXT)


def test_chunk_text_sequences_are_contiguous(engine: ChunkingEngine) -> None:
    content = "\n\n".join(
        f"## Abschnitt {i}\n\nInhalt des Abschnitts {i} mit etwas Fließtext."
        for i in range(5)
    )
    chunks = engine.chunk_text_sync(content, DocumentType.MARKDOWN)
    assert [chunk.sequence for chunk in chunks] == list(range(len(chunks)))


async def test_chunk_text_async_wrapper(engine: ChunkingEngine) -> None:
    chunks = await engine.chunk_text(
        "Ein einfacher Absatz mit genug Inhalt für einen Chunk.",
        DocumentType.TEXT,
        source_name="a.txt",
    )
    assert chunks
