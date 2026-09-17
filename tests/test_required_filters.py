"""Tests für erzwungene Pflicht-Filter (z. B. Mandanten-Isolation)."""

from __future__ import annotations

import pytest

from rag_module import RAGSettings


class TestRequiredFilterKeys:
    @pytest.fixture
    def offline_settings(self) -> RAGSettings:
        return RAGSettings(
            qdrant_url=":memory:",
            rerank_backend="none",
            expansion_backend="none",
            required_filter_keys=["tenant"],
        )

    async def test_retrieve_without_required_key_raises(self, offline_module) -> None:
        with pytest.raises(ValueError, match="Pflicht-Schlüssel"):
            await offline_module.retrieve("egal welche query")
        with pytest.raises(ValueError, match="tenant"):
            await offline_module.retrieve("query", metadata_filter={"other": "x"})
        with pytest.raises(ValueError, match="tenant"):
            await offline_module.retrieve("query", metadata_filter={"tenant": None})

    async def test_retrieve_with_required_key_enforces_isolation(self, offline_module) -> None:
        await offline_module.ingest_text(
            "Der Rahmenvertrag von ACME regelt die Kündigungsfrist von drei Monaten.",
            document_type="text",
            metadata={"document_id": "doc-acme", "tenant": "acme"},
        )
        await offline_module.ingest_text(
            "Der Rahmenvertrag von Globex regelt die Kündigungsfrist von sechs Monaten.",
            document_type="text",
            metadata={"document_id": "doc-globex", "tenant": "globex"},
        )

        results = await offline_module.retrieve(
            "Rahmenvertrag Kündigungsfrist", metadata_filter={"tenant": "acme"}
        )
        assert results
        assert {r["document_id"] for r in results} == {"doc-acme"}


async def test_default_settings_do_not_require_filters(offline_module) -> None:
    await offline_module.ingest_text(
        "Freier Text ohne Mandanten-Metadaten für die Suche.",
        document_type="text",
        metadata={"document_id": "doc-free"},
    )
    results = await offline_module.retrieve("Text Suche")
    assert results
