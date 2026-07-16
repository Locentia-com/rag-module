"""Querschnitts-Utilities: Retry-Logik mit Exponential Backoff, Token-Zählung, Zeit-Helfer."""

from __future__ import annotations

import asyncio
import importlib
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Grobe Heuristik: ~4 Zeichen pro Token (funktioniert für DE/EN-Mischtexte
#: ausreichend genau, um Chunk-Budgets modellagnostisch einzuhalten).
CHARS_PER_TOKEN: int = 4


# ---------------------------------------------------------------------------
# Token-Zählung
# ---------------------------------------------------------------------------


class BaseTokenCounter(ABC):
    """Interface für Token-Zählung; steuert alle Chunk-Budgets."""

    @abstractmethod
    def count(self, text: str) -> int:
        """Anzahl der Tokens in ``text`` (mindestens 1)."""


class HeuristicTokenCounter(BaseTokenCounter):
    """Zeichenbasierte Näherung (~4 Zeichen/Token) — modellagnostisch, ohne Download."""

    def count(self, text: str) -> int:
        return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


class HFTokenCounter(BaseTokenCounter):
    """Exakte Token-Zählung über einen HuggingFace-Tokenizer.

    Nutzt das ``tokenizers``-Paket (transitiv über fastembed installiert).
    Der Tokenizer wird beim Konstruieren geladen (HF-Hub bzw. lokaler Cache);
    ``encode`` selbst ist thread-sicher (immutabler Rust-Core).
    """

    def __init__(self, model_name: str) -> None:
        tokenizers = require_module(
            "tokenizers",
            hint="Installation: pip install tokenizers (kommt transitiv mit fastembed).",
        )
        try:
            self._tokenizer = tokenizers.Tokenizer.from_pretrained(model_name)
        except Exception as exc:
            raise ConfigurationError(
                f"HF-Tokenizer '{model_name}' konnte nicht geladen werden: {exc}. "
                "Für air-gapped Betrieb den HF-Cache vorbefüllen oder "
                "RAG_TOKENIZER_BACKEND=heuristic setzen."
            ) from exc
        self._model_name = model_name

    def count(self, text: str) -> int:
        if not text:
            return 1
        return max(1, len(self._tokenizer.encode(text, add_special_tokens=False).ids))


#: Prozessweiter aktiver Token-Counter. Bewusst modul-global, weil die
#: Chunking-Strategien Token-Budgets über Modul-Funktionen prüfen; die
#: ChunkingEngine konfiguriert ihn aus den Settings. Bei mehreren
#: AdvancedRAGModule-Instanzen mit unterschiedlichem tokenizer_backend im
#: selben Prozess gewinnt die zuletzt konstruierte (dokumentierte Grenze).
_active_token_counter: BaseTokenCounter = HeuristicTokenCounter()


def configure_token_counter(counter: BaseTokenCounter) -> None:
    """Setzt den prozessweiten Token-Counter (siehe Hinweis oben)."""
    global _active_token_counter
    _active_token_counter = counter
    logger.info("Token-Counter konfiguriert: %s", type(counter).__name__)


def estimate_tokens(text: str) -> int:
    """Token-Zählung über den aktiven Counter (Default: Zeichen-Heuristik)."""
    return _active_token_counter.count(text)


def utc_now() -> datetime:
    """Aktuelle Zeit als timezone-aware UTC-Datetime."""
    return datetime.now(UTC)


def require_module(name: str, hint: str) -> Any:
    """Importiert ein optionales Paket oder wirft eine verständliche ConfigurationError."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ConfigurationError(
            f"Das Python-Paket '{name}' ist nicht installiert, wird aber für diese "
            f"Funktion benötigt. {hint}"
        ) from exc


def is_retryable_error(exc: BaseException) -> bool:
    """Heuristik: Ist ein Fehler transient (Netzwerk, Timeout, 5xx, 429) und damit retry-würdig?

    Funktioniert SDK-übergreifend (qdrant-client, cohere, anthropic, httpx), indem
    zuerst ein etwaiger HTTP-Statuscode geprüft wird und andernfalls auf
    Exception-Typ bzw. -Namen zurückgegriffen wird.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 425, 429) or status >= 500
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    name = type(exc).__name__.lower()
    return any(
        marker in name
        for marker in (
            "timeout",
            "connection",
            "unavailable",
            "toomanyrequests",
            "ratelimit",
            "responsehandling",
            "internalserver",
            "overloaded",
        )
    )


async def retry_async(
    factory: Callable[[], Awaitable[T]],
    *,
    op_name: str,
    attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 20.0,
    timeout: float | None = None,
    should_retry: Callable[[BaseException], bool] = is_retryable_error,
) -> T:
    """Führt eine asynchrone Operation mit Exponential Backoff + Full Jitter aus.

    Args:
        factory: Muss bei JEDEM Aufruf eine neue Coroutine erzeugen
            (Coroutinen sind nicht wiederverwendbar).
        op_name: Name der Operation für Logging.
        attempts: Maximale Gesamtzahl an Versuchen (inkl. Erstversuch).
        base_delay: Startverzögerung in Sekunden; verdoppelt sich pro Versuch.
        max_delay: Obergrenze der Verzögerung in Sekunden.
        timeout: Optionales Timeout pro Versuch in Sekunden.
        should_retry: Prädikat, das entscheidet, ob ein Fehler transient ist.

    Raises:
        Die zuletzt aufgetretene Exception, wenn alle Versuche erschöpft sind
        oder der Fehler als nicht-transient eingestuft wurde.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            coro = factory()
            if timeout is not None:
                return await asyncio.wait_for(coro, timeout=timeout)
            return await coro
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 – Filterung erfolgt über should_retry
            last_exc = exc
            if attempt >= attempts or not should_retry(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random() / 2  # Full Jitter: 50–100 % des Backoffs
            logger.warning(
                "Operation '%s' fehlgeschlagen (Versuch %d/%d): %s – neuer Versuch in %.2fs",
                op_name,
                attempt,
                attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None  # unerreichbar; beruhigt den Type-Checker
    raise last_exc
