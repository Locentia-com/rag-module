"""Tests für die konfigurierbare Token-Zählung (Heuristik + HF-Tokenizer)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from rag_module.chunking import ChunkingEngine
from rag_module.config import RAGSettings
from rag_module.exceptions import ConfigurationError
from rag_module.utils import (
    configure_token_counter,
    estimate_tokens,
    heuristic_token_count,
    hf_token_counter,
)


def test_heuristic_counter_is_ceil_of_quarter_chars() -> None:
    assert heuristic_token_count("") == 1
    assert heuristic_token_count("abcd") == 1
    assert heuristic_token_count("abcde") == 2


def test_estimate_tokens_uses_configured_counter() -> None:
    configure_token_counter(lambda text: 3)
    assert estimate_tokens("beliebiger text") == 3


def _install_fake_tokenizers(
    monkeypatch: pytest.MonkeyPatch, *, raise_on_load: bool = False
) -> None:
    """Fake-``tokenizers``: zählt Wörter statt Subwords; optional Ladefehler."""

    def from_pretrained(model_name: str) -> SimpleNamespace:
        if raise_on_load:
            raise OSError(f"model '{model_name}' not found")
        return SimpleNamespace(
            encode=lambda text, add_special_tokens=True: SimpleNamespace(ids=text.split())
        )

    monkeypatch.setitem(
        sys.modules,
        "tokenizers",
        SimpleNamespace(Tokenizer=SimpleNamespace(from_pretrained=from_pretrained)),
    )


def test_hf_counter_counts_via_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tokenizers(monkeypatch)
    count = hf_token_counter("some/model")
    assert count("drei kleine tokens") == 3
    assert count("") == 1


def test_hf_counter_fails_closed_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tokenizers(monkeypatch, raise_on_load=True)
    with pytest.raises(ConfigurationError, match="konnte nicht geladen werden"):
        hf_token_counter("does/not-exist")


def test_engine_configures_hf_counter_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tokenizers(monkeypatch)
    ChunkingEngine(RAGSettings(tokenizer_backend="hf", tokenizer_model="some/model"))
    # Der aktive Counter zählt jetzt wortweise (Fake-HF), nicht zeichenweise.
    assert estimate_tokens("eins zwei drei vier") == 4


def test_engine_defaults_to_heuristic() -> None:
    ChunkingEngine(RAGSettings(tokenizer_backend="heuristic"))
    # 20 Zeichen -> 5 Tokens bei 4 Zeichen/Token
    assert estimate_tokens("a" * 20) == 5
