"""Tests für die konfigurierbare Token-Zählung (Heuristik + HF-Tokenizer)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from rag_module.chunking import ChunkingEngine
from rag_module.config import RAGSettings
from rag_module.exceptions import ConfigurationError
from rag_module.utils import (
    HeuristicTokenCounter,
    HFTokenCounter,
    configure_token_counter,
    estimate_tokens,
)


def test_heuristic_counter_is_ceil_of_quarter_chars() -> None:
    counter = HeuristicTokenCounter()
    assert counter.count("") == 1
    assert counter.count("abcd") == 1
    assert counter.count("abcde") == 2


def test_estimate_tokens_uses_configured_counter() -> None:
    class TripleCounter(HeuristicTokenCounter):
        def count(self, text: str) -> int:  # noqa: ARG002
            return 3

    configure_token_counter(TripleCounter())
    assert estimate_tokens("beliebiger text") == 3


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeHFTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> _FakeEncoding:  # noqa: ARG002
        return _FakeEncoding(list(range(len(text.split()))))


def _install_fake_tokenizers(
    monkeypatch: pytest.MonkeyPatch, *, raise_on_load: bool = False
) -> None:
    def from_pretrained(model_name: str) -> _FakeHFTokenizer:
        if raise_on_load:
            raise OSError(f"model '{model_name}' not found")
        return _FakeHFTokenizer()

    fake_module = SimpleNamespace(
        Tokenizer=SimpleNamespace(from_pretrained=from_pretrained)
    )
    monkeypatch.setitem(sys.modules, "tokenizers", fake_module)


def test_hf_counter_counts_via_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tokenizers(monkeypatch)
    counter = HFTokenCounter("some/model")
    assert counter.count("drei kleine tokens") == 3
    assert counter.count("") == 1


def test_hf_counter_fails_closed_on_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_tokenizers(monkeypatch, raise_on_load=True)
    with pytest.raises(ConfigurationError, match="konnte nicht geladen werden"):
        HFTokenCounter("does/not-exist")


def test_engine_configures_hf_counter_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_tokenizers(monkeypatch)
    settings = RAGSettings(tokenizer_backend="hf", tokenizer_model="some/model")
    ChunkingEngine(settings)
    # Der aktive Counter zählt jetzt wortweise (Fake-HF), nicht zeichenweise.
    assert estimate_tokens("eins zwei drei vier") == 4


def test_engine_defaults_to_heuristic() -> None:
    ChunkingEngine(RAGSettings(tokenizer_backend="heuristic"))
    # 20 Zeichen -> 5 Tokens bei 4 Zeichen/Token
    assert estimate_tokens("a" * 20) == 5
