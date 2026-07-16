"""Tests für erzwungene Pflicht-Filter (z. B. Mandanten-Isolation)."""

from __future__ import annotations

import pytest

from rag_module import RAGSettings
from tests.conftest import build_offline_module


@pytest.fixture
def tenant_settings() -> RAGSettings:
    return RAGSettings(
        qdrant_url=":memory:",
        rerank_backend="none",
        expansion_backend="none",
        required_filter_keys=["tenant"],
    )


async def test_retrieve_without_required_key_raises(
    tenant_settings: RAGSettings,
) -> None:
    module = build_offline_module(tenant_settings)
    try:
        with pytest.raises(ValueError, match="Pflicht-Schlüssel"):
            await module.retrieve("egal welche query")
        with pytest.raises(ValueError, match="tenant"):
            await module.retrieve("query", metadata_filter={"other": "x"})
        with pytest.raises(ValueError, match="tenant"):
            await module.retrieve("query", metadata_filter={"tenant": None})
    finally:
        await module.close()


async def test_retrieve_with_required_key_enforces_isolation(
    tenant_settings: RAGSettings,
) -> None:
    module = build_offline_module(tenant_settings)
    try:
        await module.ingest_text(
            "Der Rahmenvertrag von ACME regelt die Kündigungsfrist von drei Monaten.",
            document_type="text",
            metadata={"document_id": "doc-acme", "tenant": "acme"},
        )
        await module.ingest_text(
            "Der Rahmenvertrag von Globex regelt die Kündigungsfrist von sechs Monaten.",
            document_type="text",
            metadata={"document_id": "doc-globex", "tenant": "globex"},
        )

        results = await module.retrieve(
            "Rahmenvertrag Kündigungsfrist", metadata_filter={"tenant": "acme"}
        )
        assert results
        assert {r["document_id"] for r in results} == {"doc-acme"}
    finally:
        await module.close()


async def test_default_settings_do_not_require_filters(offline_module) -> None:
    await offline_module.ingest_text(
        "Freier Text ohne Mandanten-Metadaten für die Suche.",
        document_type="text",
        metadata={"document_id": "doc-free"},
    )
    results = await offline_module.retrieve("Text Suche")
    assert results
