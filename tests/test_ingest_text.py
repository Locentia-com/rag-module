"""Tests für ingest_text: Text-Ingestion ohne Datei, inkl. Versionierung."""

from __future__ import annotations

import pytest


async def test_ingest_text_roundtrip(offline_module) -> None:
    document_id = await offline_module.ingest_text(
        "# Handbuch\n\nDie Wartung der Anlage erfolgt quartalsweise durch Fachpersonal.",
        document_type="markdown",
        metadata={"document_id": "handbuch"},
    )
    assert document_id == "handbuch"

    results = await offline_module.retrieve("Wartung der Anlage")
    assert results
    assert results[0]["document_id"] == "handbuch"
    assert "quartalsweise" in results[0]["content"]


async def test_ingest_text_versioning_with_same_document_id(offline_module) -> None:
    await offline_module.ingest_text(
        "Version eins des Inhalts über Kaffeemaschinen.",
        document_type="text",
        metadata={"document_id": "doc"},
    )
    await offline_module.ingest_text(
        "Version zwei des Inhalts über Kaffeemaschinen.",
        document_type="text",
        metadata={"document_id": "doc"},
    )

    history = await offline_module.get_document_history("doc")
    assert [entry["version"] for entry in history] == [1, 2]
    assert history[0]["is_active"] is False
    assert history[1]["is_active"] is True

    # Default-Retrieve sieht nur den aktiven Stand …
    results = await offline_module.retrieve("Inhalts über Kaffeemaschinen")
    assert {r["version"] for r in results} == {2}

    # … die alte Version bleibt über den Temporal-Filter erreichbar.
    old = await offline_module.retrieve(
        "Inhalts über Kaffeemaschinen", temporal_filter={"version": 1}
    )
    assert {r["version"] for r in old} == {1}


async def test_ingest_text_derives_id_from_source_name(offline_module) -> None:
    document_id = await offline_module.ingest_text(
        "Inhalt aus einer benannten Quelle mit genug Text.",
        document_type="text",
        metadata={},
        source_name="Quartalsbericht Q3.txt",
    )
    assert document_id == "quartalsbericht-q3"


async def test_ingest_text_without_id_or_source_gets_random_id(offline_module) -> None:
    first = await offline_module.ingest_text(
        "Anonymer Inhalt eins mit ausreichend Textmenge.",
        document_type="text",
        metadata={},
    )
    second = await offline_module.ingest_text(
        "Anonymer Inhalt zwei mit ausreichend Textmenge.",
        document_type="text",
        metadata={},
    )
    # Keine stillschweigende Versionierung unter einer geteilten Fallback-ID.
    assert first != second


async def test_ingest_text_rejects_pdf(offline_module) -> None:
    with pytest.raises(ValueError, match="PDF"):
        await offline_module.ingest_text(
            "irrelevant", document_type="pdf", metadata={}
        )


async def test_ingest_text_rejects_empty_content(offline_module) -> None:
    with pytest.raises(ValueError, match="content"):
        await offline_module.ingest_text("   ", document_type="text", metadata={})


async def test_ingest_text_rejects_unserializable_metadata(offline_module) -> None:
    with pytest.raises(ValueError, match="JSON-serialisierbar"):
        await offline_module.ingest_text(
            "Inhalt mit kaputten Metadaten und genug Text.",
            document_type="text",
            metadata={"bad": object()},
        )
