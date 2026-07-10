"""Intelligente, layout-bewusste Chunking-Engine.

Kern-Prinzipien:
- KEIN starres Zeichen-Chunking: Jede Dokumentklasse (Markdown, PDF, Verträge/
  Gesetze, Code, JSON/API-Payloads, SQL-Dumps) hat eine strukturerhaltende Strategie.
- Tabellen werden als vollständige Markdown-/HTML-Strukturen erkannt und als EIN
  Chunk isoliert – inklusive übergeordnetem Textkontext (Breadcrumb + letzter Absatz).
- Verträge/Gesetze werden anhand von §-/Artikel-/Überschriften-Hierarchien
  segmentiert (Parent-Child-Chunking: Parent = ganze Sektion, Children = Absätze).
- Code wird per AST (Python) bzw. Klammer-Analyse (C-artige Sprachen) nach
  Klassen/Funktionen zerlegt.

Alle Chunker sind reine, zustandslose Funktionen über Strings und damit
thread-sicher; die Engine führt sie über ``asyncio.to_thread`` aus, um die
Event-Loop nicht zu blockieren.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .config import RAGSettings
from .exceptions import ChunkingError, DocumentLoadError
from .models import Chunk, ChunkRole, ChunkType, DocumentType
from .utils import CHARS_PER_TOKEN, estimate_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rekursiver Text-Splitter (Fallback-Baustein aller Strategien)
# ---------------------------------------------------------------------------

#: Trennmuster in absteigender Priorität: Absatz -> Zeile -> Satzende -> Whitespace
_BOUNDARY_PATTERNS: list[str] = [r"\n{2,}", r"\n", r"(?<=[.!?;:])\s", r"\s+"]


def _split_keep(text: str, pattern: str) -> list[str]:
    """Splittet ``text`` an ``pattern``; der Separator verbleibt am linken Teilstück."""
    parts = re.split(f"({pattern})", text)
    out: list[str] = []
    for i in range(0, len(parts), 2):
        segment = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        if segment:
            out.append(segment)
    return out


def _split_atomic(text: str, max_chars: int, _level: int = 0) -> list[str]:
    """Zerlegt Text rekursiv entlang natürlicher Grenzen in Stücke <= max_chars."""
    if len(text) <= max_chars:
        return [text]
    if _level >= len(_BOUNDARY_PATTERNS):
        # Harte Notbremse: Zeichenweise schneiden (z. B. eine einzige Riesenzeile).
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    out: list[str] = []
    for part in _split_keep(text, _BOUNDARY_PATTERNS[_level]):
        if len(part) <= max_chars:
            out.append(part)
        else:
            out.extend(_split_atomic(part, max_chars, _level + 1))
    return out


def split_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    *,
    min_tokens: int = 1,
) -> list[str]:
    """Teilt Text in Fenster <= ``max_tokens`` mit Überlappung an natürlichen Grenzen.

    Ein zu kleines letztes Fenster (< ``min_tokens``) wird mit dem vorherigen
    verschmolzen, damit keine kontextlosen Mini-Chunks entstehen.
    """
    text = text.strip()
    if not text:
        return []
    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = min(overlap_tokens * CHARS_PER_TOKEN, max_chars // 3)
    if len(text) <= max_chars:
        return [text]

    atoms = _split_atomic(text, max_chars)
    windows: list[str] = []
    current = ""
    for atom in atoms:
        if len(current) + len(atom) <= max_chars:
            current += atom
            continue
        if current.strip():
            windows.append(current.strip())
        tail = current[-overlap_chars:] if overlap_chars else ""
        # Überlappung an Wortgrenze beginnen lassen
        cut = tail.find(" ")
        if 0 <= cut < len(tail) - 1:
            tail = tail[cut + 1 :]
        current = (tail + atom) if len(tail) + len(atom) <= max_chars else atom
    if current.strip():
        windows.append(current.strip())

    if len(windows) >= 2 and estimate_tokens(windows[-1]) < min_tokens:
        last = windows.pop()
        windows[-1] = windows[-1] + "\n" + last
    return windows


def _cap_text(text: str, max_tokens: int) -> str:
    """Kürzt Text hart auf ein Token-Budget (für Parent-Chunks) mit Markierung."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    boundary = cut.rfind(" ")
    if boundary > max_chars // 2:
        cut = cut[:boundary]
    return cut + "\n… [Inhalt für Parent-Kontext gekürzt]"


# ---------------------------------------------------------------------------
# Basis-Interface
# ---------------------------------------------------------------------------


class BaseChunker(ABC):
    """Interface aller Chunking-Strategien."""

    @abstractmethod
    def chunk(self, content: str, *, source_name: str = "") -> list[Chunk]:
        """Zerlegt ``content`` in strukturerhaltende Chunks."""


# ---------------------------------------------------------------------------
# Block-Parser für strukturierte Texte (Markdown, Verträge, Plain Text)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Block:
    kind: str  # "heading" | "table" | "code" | "text"
    text: str
    level: int = 0
    title: str = ""


@dataclass(slots=True)
class _Section:
    hierarchy: list[str]
    heading_line: str
    blocks: list[_Block] = field(default_factory=list)


_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")


def _parse_blocks(text: str, detect_heading) -> list[_Block]:
    """Zerlegt Text zeilenweise in Blöcke: Überschriften, Tabellen, Code-Fences, Fließtext.

    Tabellen (Markdown- und HTML-Syntax) und Code-Fences werden atomar erfasst,
    sodass sie später niemals mitten in der Struktur zerschnitten werden.
    """
    lines = text.split("\n")
    blocks: list[_Block] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraph = "\n".join(buffer).strip("\n")
            if paragraph.strip():
                blocks.append(_Block("text", paragraph))
            buffer.clear()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(2)[0] * 3
            collected = [line]
            i += 1
            while i < n:
                collected.append(lines[i])
                if lines[i].strip().startswith(marker):
                    i += 1
                    break
                i += 1
            flush()
            blocks.append(_Block("code", "\n".join(collected)))
            continue

        if "<table" in line.lower():
            collected = []
            while i < n:
                collected.append(lines[i])
                if "</table>" in lines[i].lower():
                    i += 1
                    break
                i += 1
            flush()
            blocks.append(_Block("table", "\n".join(collected)))
            continue

        if (
            _MD_TABLE_ROW.match(line)
            and i + 1 < n
            and _MD_TABLE_SEP.match(lines[i + 1])
            and "|" in lines[i + 1]
        ):
            collected = []
            while i < n and (_MD_TABLE_ROW.match(lines[i]) or _MD_TABLE_SEP.match(lines[i])):
                collected.append(lines[i])
                i += 1
            flush()
            blocks.append(_Block("table", "\n".join(collected)))
            continue

        heading = detect_heading(line)
        if heading is not None:
            flush()
            level, title = heading
            blocks.append(_Block("heading", line.strip(), level=level, title=title))
            i += 1
            continue

        buffer.append(line)
        i += 1

    flush()
    return blocks


def _build_sections(blocks: Sequence[_Block]) -> list[_Section]:
    """Ordnet Blöcke ihren Überschriften-Hierarchien zu (Stack-basierte Gliederung)."""
    stack: list[tuple[int, str]] = []
    sections: list[_Section] = [_Section(hierarchy=[], heading_line="")]
    for block in blocks:
        if block.kind == "heading":
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.title))
            sections.append(
                _Section(hierarchy=[title for _, title in stack], heading_line=block.text)
            )
        else:
            sections[-1].blocks.append(block)
    return [s for s in sections if s.blocks or s.heading_line]


def _tail_context(text_parts: Sequence[str], fallback: str, max_chars: int = 400) -> str:
    """Letzter Absatz vor einer Tabelle als übergeordneter Textkontext."""
    for part in reversed(text_parts):
        cleaned = " ".join(part.split())
        if cleaned:
            return cleaned[-max_chars:]
    return " ".join(fallback.split())[:max_chars]


class _StructuredTextChunker(BaseChunker):
    """Gemeinsame Basis für hierarchische Text-Chunker (Markdown, Legal, Plain Text).

    Erzeugt pro Sektion:
    - einen (gekappten) Parent-Chunk, sobald die Sektion mehr als eine
      Retrieval-Einheit enthält,
    - Child-Chunks für den Fließtext (mit Breadcrumb-Kontextheader),
    - isolierte Chunks für Tabellen und große Code-Fences – inklusive
      übergeordnetem Textkontext.
    """

    chunk_type_text: ChunkType = ChunkType.TEXT

    def __init__(
        self,
        *,
        max_tokens: int,
        overlap_tokens: int,
        parent_max_tokens: int,
        min_tokens: int = 24,
    ) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._parent_max_tokens = parent_max_tokens
        self._min_tokens = min_tokens

    # -- Hooks für Subklassen ------------------------------------------------

    @abstractmethod
    def detect_heading(self, line: str) -> Optional[tuple[int, str]]:
        """Gibt (Level, Titel) zurück, wenn die Zeile eine Überschrift ist."""

    def split_body(self, body: str) -> list[str]:
        """Zerlegt den Fließtext einer Sektion; Subklassen können Marker nutzen."""
        return split_text(
            body, self._max_tokens, self._overlap_tokens, min_tokens=self._min_tokens
        )

    # -- Kern-Logik ------------------------------------------------------------

    def chunk(self, content: str, *, source_name: str = "") -> list[Chunk]:
        blocks = _parse_blocks(content, self.detect_heading)
        sections = _build_sections(blocks)
        chunks: list[Chunk] = []
        for section in sections:
            self._emit_section(section, chunks)
        return chunks

    def _emit_section(self, section: _Section, out: list[Chunk]) -> None:
        breadcrumb = " > ".join(section.hierarchy)
        header = f"[Kontext: {breadcrumb}]\n\n" if breadcrumb else ""

        text_parts: list[str] = []
        specials: list[tuple[str, str, ChunkType]] = []  # (Inhalt, Kontext, Typ)
        for block in section.blocks:
            if block.kind == "table":
                specials.append(
                    (block.text, _tail_context(text_parts, section.heading_line), ChunkType.TABLE)
                )
            elif block.kind == "code" and estimate_tokens(block.text) > self._max_tokens:
                specials.append(
                    (
                        block.text,
                        _tail_context(text_parts, section.heading_line),
                        ChunkType.CODE_UNIT,
                    )
                )
            else:
                text_parts.append(block.text)

        body = "\n\n".join(text_parts).strip()
        full = "\n\n".join(part for part in (section.heading_line, body) if part).strip()
        pieces = self.split_body(full) if full else []
        total_units = len(pieces) + len(specials)
        if total_units == 0:
            return

        parent: Optional[Chunk] = None
        if total_units > 1:
            parent_source = full if full else "\n\n".join(s[0] for s in specials)
            parent = Chunk(
                content=header + _cap_text(parent_source, self._parent_max_tokens),
                chunk_type=ChunkType.SECTION,
                role=ChunkRole.PARENT,
                hierarchy=list(section.hierarchy),
                searchable=False,
            )
            out.append(parent)

        role = ChunkRole.CHILD if parent else ChunkRole.STANDALONE
        parent_id = parent.chunk_id if parent else None
        for piece in pieces:
            out.append(
                Chunk(
                    content=header + piece,
                    chunk_type=self.chunk_type_text,
                    role=role,
                    parent_id=parent_id,
                    hierarchy=list(section.hierarchy),
                )
            )
        for special_content, context, ctype in specials:
            context_line = f"[Übergeordneter Textkontext: {context}]\n\n" if context else ""
            out.append(
                Chunk(
                    content=header + context_line + special_content,
                    chunk_type=ctype,
                    role=role,
                    parent_id=parent_id,
                    hierarchy=list(section.hierarchy),
                    extra={"isolated_structure": True},
                )
            )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


class MarkdownChunker(_StructuredTextChunker):
    """Markdown-Chunker: ATX-Überschriften-Hierarchie, Tabellen- und Fence-Isolation.

    Hinweis: Setext-Überschriften (``===``/``---``-Unterstreichungen) werden
    bewusst nicht unterstützt, da sie mit Tabellen-Separatoren kollidieren;
    ATX-Stil (``#``) ist der De-facto-Standard.
    """

    def detect_heading(self, line: str) -> Optional[tuple[int, str]]:
        match = _MD_HEADING_RE.match(line)
        if not match:
            return None
        return len(match.group(1)), match.group(2).strip()


# ---------------------------------------------------------------------------
# Verträge / Gesetze (Legal)
# ---------------------------------------------------------------------------

_LEGAL_LEVEL_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^\s*(Teil|Buch)\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE), 1),
    (re.compile(r"^\s*(Kapitel|Abschnitt|Titel|Unterabschnitt)\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE), 2),
    (re.compile(r"^\s*(Anlage|Anhang|Appendix|Annex)(\s+[\w.]+)?\s*:?\s*", re.IGNORECASE), 2),
    (re.compile(r"^\s*(Präambel|Preamble)\s*:?\s*$", re.IGNORECASE), 2),
    (re.compile(r"^\s*§{1,2}\s*\d+\s*[a-z]?\b"), 3),
    (re.compile(r"^\s*Art(?:ikel|\.)\s*\d+\s*[a-z]?\b", re.IGNORECASE), 3),
]

_LEGAL_NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+(?=[A-ZÄÖÜ§\"'(])")
_ABSATZ_SPLIT_RE = re.compile(r"(?m)^(?=\(\d+\)\s)")


class LegalChunker(_StructuredTextChunker):
    """Chunker für Verträge und Gesetze.

    Segmentiert anhand von §-, Artikel-, Abschnitts- und Klausel-Nummerierungen
    (z. B. ``§ 5``, ``Artikel 12``, ``3.2 Haftung``) sowie Markdown-Überschriften.
    Innerhalb eines Paragraphen dienen Absatz-Marker ``(1)``, ``(2)`` … als
    bevorzugte Schnittgrenzen, sodass juristische Sinneinheiten intakt bleiben.
    """

    chunk_type_text = ChunkType.LEGAL_CLAUSE

    def detect_heading(self, line: str) -> Optional[tuple[int, str]]:
        stripped = line.strip()
        if not stripped or len(stripped) > 160:
            return None

        md = _MD_HEADING_RE.match(line)
        if md:
            return len(md.group(1)), md.group(2).strip()

        for pattern, level in _LEGAL_LEVEL_PATTERNS:
            if pattern.match(stripped):
                return level, stripped

        # Nummerierte Vertragsklauseln ("3.2 Haftung") – nur kurze Zeilen ohne
        # Satz-Schlusszeichen, um Aufzählungs-Sätze nicht als Überschrift zu deuten.
        numbered = _LEGAL_NUMBERED_RE.match(stripped)
        if numbered and len(stripped) <= 90 and not re.search(r"[.;,]\s*$", stripped):
            depth = numbered.group(1).count(".")
            return min(2 + depth, 6), stripped
        return None

    def split_body(self, body: str) -> list[str]:
        segments = [s for s in _ABSATZ_SPLIT_RE.split(body) if s.strip()]
        if len(segments) <= 1:
            return super().split_body(body)

        # Absätze greedy zu Fenstern <= max_tokens zusammenfassen;
        # überlange Einzelabsätze rekursiv weiterteilen.
        windows: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for segment in segments:
            seg_tokens = estimate_tokens(segment)
            if seg_tokens > self._max_tokens:
                if current:
                    windows.append("\n".join(current).strip())
                    current, current_tokens = [], 0
                windows.extend(
                    split_text(
                        segment,
                        self._max_tokens,
                        self._overlap_tokens,
                        min_tokens=self._min_tokens,
                    )
                )
                continue
            if current_tokens + seg_tokens > self._max_tokens and current:
                windows.append("\n".join(current).strip())
                current, current_tokens = [], 0
            current.append(segment.rstrip())
            current_tokens += seg_tokens
        if current:
            windows.append("\n".join(current).strip())
        return [w for w in windows if w]


# ---------------------------------------------------------------------------
# Plain Text / PDF-Fließtext (heuristische Überschriften-Erkennung)
# ---------------------------------------------------------------------------

_PLAIN_NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+\S")


class PlainTextChunker(_StructuredTextChunker):
    """Generischer Text-Chunker mit konservativer Überschriften-Heuristik.

    Erkennt nummerierte Gliederungen ("2.1 Systemarchitektur") und
    GROSSBUCHSTABEN-Zeilen als Überschriften; alles andere wird als Fließtext
    an natürlichen Grenzen segmentiert.
    """

    def detect_heading(self, line: str) -> Optional[tuple[int, str]]:
        stripped = line.strip()
        if not 4 <= len(stripped) <= 90 or stripped.endswith((".", ",", ";")):
            return None
        numbered = _PLAIN_NUMBERED_RE.match(stripped)
        if numbered:
            rest = stripped[numbered.end(1) :].lstrip(".) \t")
            if rest and rest[0].isupper() and len(stripped) <= 80:
                return min(2 + numbered.group(1).count("."), 6), stripped
        letters = [c for c in stripped if c.isalpha()]
        if len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.85:
            return 2, stripped
        return None


# ---------------------------------------------------------------------------
# Code (AST für Python, Klammer-Analyse für C-artige Sprachen)
# ---------------------------------------------------------------------------

_BRACE_HEADER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)?\s*\("),
    re.compile(r"^\s*(?:export\s+)?(?:public\s+|abstract\s+)?class\s+([A-Za-z_$][\w$]*)"),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"),
    re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\("),  # Go
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)"),  # Rust
    re.compile(r"^\s*(?:public|private|protected|internal|static)[\w<>\[\],\s]*\s([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$"),  # Java/C#
]

_PY_HINT_RE = re.compile(r"^\s*(def |class |import |from \w)", re.MULTILINE)


class CodeChunker(BaseChunker):
    """Code-Chunker: zerlegt Quelldateien nach Klassen und Funktionen.

    - Python: exakte Segmentierung per ``ast`` (inkl. Dekoratoren); große Klassen
      werden per Parent-Child-Chunking in Klassenkopf + Einzelmethoden zerlegt.
    - C-artige Sprachen (JS/TS, Go, Rust, Java, C#): Signatur-Erkennung per Regex
      plus string-/kommentar-bewusstes Brace-Matching.
    - Fallback: rekursive Segmentierung an Leerzeilen-Grenzen.
    """

    def __init__(self, *, max_tokens: int, overlap_tokens: int, parent_max_tokens: int) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._parent_max_tokens = parent_max_tokens

    def chunk(self, content: str, *, source_name: str = "") -> list[Chunk]:
        suffix = Path(source_name).suffix.lower() if source_name else ""
        if suffix in {".py", ".pyi"} or (not suffix and len(_PY_HINT_RE.findall(content)) >= 2):
            try:
                return self._chunk_python(content, source_name)
            except SyntaxError:
                logger.warning(
                    "Python-AST-Parsing für '%s' fehlgeschlagen – Fallback auf Klammer-Analyse.",
                    source_name,
                )
        return self._chunk_braces(content, source_name)

    # -- Python (AST) ---------------------------------------------------------

    def _chunk_python(self, source: str, source_name: str) -> list[Chunk]:
        tree = ast.parse(source)
        lines = source.split("\n")
        file_header = f"# Datei: {source_name}\n" if source_name else ""

        def node_span(node: ast.stmt) -> tuple[int, int]:
            start = node.lineno
            for decorator in getattr(node, "decorator_list", []):
                start = min(start, decorator.lineno)
            return start, node.end_lineno or node.lineno

        def segment(start: int, end: int) -> str:
            return "\n".join(lines[start - 1 : end])

        chunks: list[Chunk] = []
        module_spans: list[tuple[int, int]] = []

        def flush_module() -> None:
            if not module_spans:
                return
            text = "\n".join(segment(s, e) for s, e in module_spans).strip("\n")
            module_spans.clear()
            if not text.strip():
                return
            for piece in split_text(text, self._max_tokens, self._overlap_tokens):
                chunks.append(
                    Chunk(
                        content=file_header + piece,
                        chunk_type=ChunkType.CODE_UNIT,
                        extra={"language": "python", "kind": "module"},
                    )
                )

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                flush_module()
                chunks.append(
                    Chunk(
                        content=file_header + segment(*node_span(node)),
                        chunk_type=ChunkType.CODE_UNIT,
                        extra={"language": "python", "kind": "function", "qualname": node.name},
                    )
                )
            elif isinstance(node, ast.ClassDef):
                flush_module()
                self._emit_python_class(node, node_span, segment, file_header, chunks)
            else:
                module_spans.append(node_span(node))
        flush_module()

        if not chunks:
            raise ChunkingError(f"Python-Datei '{source_name}' enthält keinen chunkbaren Code.")
        return chunks

    def _emit_python_class(
        self,
        node: ast.ClassDef,
        node_span,
        segment,
        file_header: str,
        out: list[Chunk],
    ) -> None:
        start, end = node_span(node)
        full = segment(start, end)
        qualname = node.name

        if estimate_tokens(full) <= self._max_tokens * 2:
            out.append(
                Chunk(
                    content=file_header + full,
                    chunk_type=ChunkType.CODE_UNIT,
                    extra={"language": "python", "kind": "class", "qualname": qualname},
                )
            )
            return

        # Große Klasse: Parent = ganze Klasse (gekappt), Children = Kopf + Methoden.
        parent = Chunk(
            content=file_header + _cap_text(full, self._parent_max_tokens),
            chunk_type=ChunkType.CODE_UNIT,
            role=ChunkRole.PARENT,
            searchable=False,
            extra={"language": "python", "kind": "class", "qualname": qualname},
        )
        out.append(parent)
        class_signature = segment(node.lineno, node.lineno).strip()

        methods = [
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        method_lines = {
            line
            for method in methods
            for line in range(node_span(method)[0], node_span(method)[1] + 1)
        }
        head_body_lines = [
            line
            for line in range(start, end + 1)
            if line not in method_lines
        ]
        if head_body_lines:
            head_text = "\n".join(
                segment(line, line) for line in head_body_lines
            ).strip("\n")
            if head_text.strip():
                out.append(
                    Chunk(
                        content=file_header + head_text,
                        chunk_type=ChunkType.CODE_UNIT,
                        role=ChunkRole.CHILD,
                        parent_id=parent.chunk_id,
                        extra={
                            "language": "python",
                            "kind": "class_header",
                            "qualname": qualname,
                        },
                    )
                )
        for method in methods:
            method_src = segment(*node_span(method))
            out.append(
                Chunk(
                    content=f"{file_header}# Kontext: {class_signature}\n{method_src}",
                    chunk_type=ChunkType.CODE_UNIT,
                    role=ChunkRole.CHILD,
                    parent_id=parent.chunk_id,
                    extra={
                        "language": "python",
                        "kind": "method",
                        "qualname": f"{qualname}.{method.name}",
                    },
                )
            )

    # -- C-artige Sprachen (Brace-Matching) ------------------------------------

    def _chunk_braces(self, source: str, source_name: str) -> list[Chunk]:
        lines = source.split("\n")
        file_header = f"// Datei: {source_name}\n" if source_name else ""
        consumed = [False] * len(lines)
        units: list[tuple[int, int, str]] = []

        i = 0
        while i < len(lines):
            name: Optional[str] = None
            matched = False
            for pattern in _BRACE_HEADER_PATTERNS:
                match = pattern.match(lines[i])
                if match:
                    matched = True
                    name = match.group(1) or "<anonym>"
                    break
            if matched:
                end = self._find_block_end(lines, i)
                if end is not None:
                    units.append((i, end, name or "<anonym>"))
                    for j in range(i, end + 1):
                        consumed[j] = True
                    i = end + 1
                    continue
            i += 1

        chunks: list[Chunk] = []
        for start, end, name in units:
            unit_src = "\n".join(lines[start : end + 1])
            chunks.append(
                Chunk(
                    content=file_header + unit_src,
                    chunk_type=ChunkType.CODE_UNIT,
                    extra={"kind": "unit", "qualname": name},
                )
            )

        # Zwischenräume (Imports, Konstanten, freistehender Code) einsammeln.
        leftovers: list[str] = []
        run: list[str] = []
        for idx, line in enumerate(lines):
            if consumed[idx]:
                if run:
                    leftovers.append("\n".join(run))
                    run = []
            else:
                run.append(line)
        if run:
            leftovers.append("\n".join(run))
        leftover_text = "\n\n".join(part for part in leftovers if part.strip())
        if leftover_text.strip():
            for piece in split_text(leftover_text, self._max_tokens, self._overlap_tokens):
                chunks.append(
                    Chunk(
                        content=file_header + piece,
                        chunk_type=ChunkType.CODE_UNIT,
                        extra={"kind": "module"},
                    )
                )

        if not chunks:
            for piece in split_text(source, self._max_tokens, self._overlap_tokens):
                chunks.append(
                    Chunk(
                        content=file_header + piece,
                        chunk_type=ChunkType.CODE_UNIT,
                        extra={"kind": "raw"},
                    )
                )
        return chunks

    @staticmethod
    def _find_block_end(lines: list[str], start: int) -> Optional[int]:
        """Findet die Zeile, in der der ``{...}``-Block einer Signatur schließt.

        String- und kommentar-bewusster Zustandsautomat; bricht ab, wenn in den
        ersten drei Zeilen keine öffnende Klammer auftaucht (z. B. Arrow-Function
        ohne Block) oder der Block nicht innerhalb von 4000 Zeilen schließt.
        """
        has_opening = any("{" in lines[j] for j in range(start, min(start + 3, len(lines))))
        if not has_opening:
            return None

        depth = 0
        started = False
        in_string: Optional[str] = None
        in_block_comment = False
        for idx in range(start, min(start + 4000, len(lines))):
            line = lines[idx]
            pos = 0
            length = len(line)
            while pos < length:
                char = line[pos]
                nxt = line[pos + 1] if pos + 1 < length else ""
                if in_block_comment:
                    if char == "*" and nxt == "/":
                        in_block_comment = False
                        pos += 2
                        continue
                    pos += 1
                    continue
                if in_string is not None:
                    if char == "\\":
                        pos += 2
                        continue
                    if char == in_string:
                        in_string = None
                    pos += 1
                    continue
                if char == "/" and nxt == "/":
                    break  # Zeilenkommentar: Rest der Zeile ignorieren
                if char == "/" and nxt == "*":
                    in_block_comment = True
                    pos += 2
                    continue
                if char in ("'", '"', "`"):
                    in_string = char
                    pos += 1
                    continue
                if char == "{":
                    depth += 1
                    started = True
                elif char == "}":
                    depth -= 1
                    if started and depth == 0:
                        return idx
                pos += 1
            # Einzeilige Strings gelten am Zeilenende als beendet (defensiv
            # gegen unbalancierte Quotes in Kommentaren/Regex-Literalen).
            if in_string is not None and in_string != "`":
                in_string = None
        return None


# ---------------------------------------------------------------------------
# JSON / API-Payloads
# ---------------------------------------------------------------------------


class JSONChunker(BaseChunker):
    """JSON-/API-Payload-Chunker: strukturerhaltende Fragmentierung mit JSON-Pfaden.

    Jeder Chunk ist ein VALIDES JSON-Fragment (Objekt-Teilmenge oder Array-Slice)
    mit seinem JSON-Pfad als Kontextzeile. Für große Dokumente entsteht ein
    Parent-Chunk mit einer Strukturübersicht. NDJSON (eine JSON-Zeile pro
    Datensatz) wird automatisch erkannt.
    """

    def __init__(self, *, max_tokens: int, overlap_tokens: int, parent_max_tokens: int) -> None:
        self._max_tokens = max_tokens
        self._max_chars = max_tokens * CHARS_PER_TOKEN
        self._overlap_tokens = overlap_tokens
        self._parent_max_tokens = parent_max_tokens

    def chunk(self, content: str, *, source_name: str = "") -> list[Chunk]:
        data = self._parse(content, source_name)
        fragments = self._fragments("$", data)
        if not fragments:
            raise ChunkingError(f"JSON-Dokument '{source_name}' ergab keine Fragmente.")

        chunks: list[Chunk] = []
        parent: Optional[Chunk] = None
        if len(fragments) > 1:
            label = f" '{source_name}'" if source_name else ""
            parent = Chunk(
                content=_cap_text(
                    f"JSON-Dokument{label} – Strukturübersicht:\n{self._shape_summary(data)}",
                    self._parent_max_tokens,
                ),
                chunk_type=ChunkType.SECTION,
                role=ChunkRole.PARENT,
                searchable=False,
                extra={"json_path": "$"},
            )
            chunks.append(parent)

        role = ChunkRole.CHILD if parent else ChunkRole.STANDALONE
        for path, body in fragments:
            chunks.append(
                Chunk(
                    content=f"JSON-Pfad: {path}\n{body}",
                    chunk_type=ChunkType.JSON_FRAGMENT,
                    role=role,
                    parent_id=parent.chunk_id if parent else None,
                    extra={"json_path": path},
                )
            )
        return chunks

    def _parse(self, content: str, source_name: str) -> Any:
        try:
            return json.loads(content)
        except json.JSONDecodeError as primary_error:
            # NDJSON-Fallback: eine JSON-Struktur pro Zeile (typisch für API-Logs).
            records: list[Any] = []
            invalid = 0
            for line in content.splitlines():
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    invalid += 1
            if records and invalid <= max(1, len(records) // 5):
                if invalid:
                    logger.warning(
                        "NDJSON '%s': %d ungültige Zeilen übersprungen.", source_name, invalid
                    )
                return records
            raise ChunkingError(
                f"'{source_name}' ist weder valides JSON noch NDJSON: {primary_error}"
            ) from primary_error

    def _dumps(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)

    def _fits(self, serialized: str) -> bool:
        return len(serialized) <= self._max_chars

    def _fragments(self, path: str, value: Any) -> list[tuple[str, str]]:
        serialized = self._dumps(value)
        if self._fits(serialized):
            return [(path, serialized)]

        if isinstance(value, dict):
            out: list[tuple[str, str]] = []
            group: dict[str, Any] = {}
            group_size = 2
            def flush_group() -> None:
                nonlocal group, group_size
                if group:
                    keys = ",".join(group)
                    out.append((f"{path}.{{{keys}}}", self._dumps(group)))
                group = {}
                group_size = 2
            for key, val in value.items():
                entry = self._dumps({key: val})
                if not self._fits(entry):
                    flush_group()
                    out.extend(self._fragments(f"{path}.{key}", val))
                    continue
                if group_size + len(entry) > self._max_chars:
                    flush_group()
                group[key] = val
                group_size += len(entry)
            flush_group()
            return out

        if isinstance(value, list):
            out = []
            buffer: list[Any] = []
            buffer_size = 2
            start = 0
            for index, item in enumerate(value):
                item_serialized = self._dumps(item)
                if not self._fits(item_serialized):
                    if buffer:
                        out.append((f"{path}[{start}:{index}]", self._dumps(buffer)))
                        buffer, buffer_size = [], 2
                    out.extend(self._fragments(f"{path}[{index}]", item))
                    start = index + 1
                    continue
                if buffer_size + len(item_serialized) > self._max_chars and buffer:
                    out.append((f"{path}[{start}:{index}]", self._dumps(buffer)))
                    buffer, buffer_size = [], 2
                    start = index
                buffer.append(item)
                buffer_size += len(item_serialized) + 2
            if buffer:
                out.append((f"{path}[{start}:{len(value)}]", self._dumps(buffer)))
            return out

        # Überlanger Skalar (z. B. eingebetteter Fließtext in einem Feld).
        parts = split_text(str(value), self._max_tokens, self._overlap_tokens)
        return [(f"{path} (Teil {i + 1}/{len(parts)})", part) for i, part in enumerate(parts)]

    def _shape_summary(self, value: Any, depth: int = 0) -> str:
        indent = "  " * depth
        if isinstance(value, dict):
            lines: list[str] = []
            for key, val in list(value.items())[:60]:
                if isinstance(val, (dict, list)) and depth < 2:
                    lines.append(f"{indent}{key}:")
                    lines.append(self._shape_summary(val, depth + 1))
                else:
                    lines.append(f"{indent}{key}: {type(val).__name__}")
            if len(value) > 60:
                lines.append(f"{indent}… ({len(value) - 60} weitere Schlüssel)")
            return "\n".join(lines)
        if isinstance(value, list):
            element_type = type(value[0]).__name__ if value else "leer"
            summary = f"{indent}Array mit {len(value)} Elementen (Typ: {element_type})"
            if value and isinstance(value[0], dict) and depth < 2:
                summary += "\n" + self._shape_summary(value[0], depth + 1)
            return summary
        return f"{indent}{type(value).__name__}"


# ---------------------------------------------------------------------------
# SQL-Dumps
# ---------------------------------------------------------------------------

_SQL_CREATE_TABLE_RE = re.compile(
    r"^\s*CREATE\s+(?:TEMPORARY\s+|UNLOGGED\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\"'`\[\]\w.]+)",
    re.IGNORECASE,
)
_SQL_INSERT_RE = re.compile(r"^\s*INSERT\s+INTO\s+([\"'`\[\]\w.]+)", re.IGNORECASE)
_SQL_ALTER_RE = re.compile(r"^\s*ALTER\s+TABLE\s+(?:ONLY\s+)?([\"'`\[\]\w.]+)", re.IGNORECASE)
_SQL_INDEX_RE = re.compile(
    r"^\s*CREATE\s+(?:UNIQUE\s+)?INDEX\s+.*?\s+ON\s+(?:ONLY\s+)?([\"'`\[\]\w.]+)",
    re.IGNORECASE | re.DOTALL,
)
_SQL_COMMENT_RE = re.compile(
    r"^\s*COMMENT\s+ON\s+(?:TABLE|COLUMN)\s+([\"'`\[\]\w.]+)", re.IGNORECASE
)
_SQL_COPY_RE = re.compile(r"^\s*COPY\s+([\"'`\[\]\w.]+)", re.IGNORECASE)
_SQL_COPY_HEADER_RE = re.compile(r"^\s*COPY\s+.*FROM\s+stdin;?\s*$", re.IGNORECASE)


def _normalize_table_name(raw: str) -> str:
    name = raw.strip("\"'`[]")
    return name.split(".")[-1].strip("\"'`[]")


class SQLDumpChunker(BaseChunker):
    """SQL-Dump-Chunker: hält Tabellenschemata zusammen und gruppiert Daten pro Tabelle.

    - ``CREATE TABLE`` + zugehörige ``ALTER``/``INDEX``/``COMMENT``-Statements einer
      Tabelle bilden einen Schema-Chunk (zugleich Parent der Daten-Chunks).
    - ``INSERT``-Statements werden pro Tabelle zu Batches <= max_tokens gebündelt.
    - PostgreSQL-``COPY … FROM stdin``-Blöcke werden erkannt; ihre Datenzeilen
      werden mit wiederholtem COPY-Header gebatcht, sodass jeder Chunk
      selbsterklärend bleibt.
    - Der String-/Kommentar-bewusste Statement-Splitter respektiert einfache
      Quotes, Dollar-Quoting (``$tag$``), ``--``- und ``/* */``-Kommentare.
    """

    def __init__(self, *, max_tokens: int, overlap_tokens: int, parent_max_tokens: int) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._parent_max_tokens = parent_max_tokens

    def chunk(self, content: str, *, source_name: str = "") -> list[Chunk]:
        statements = self._split_statements(content)
        if not statements:
            raise ChunkingError(f"SQL-Dump '{source_name}' enthält keine Statements.")

        schema_by_table: dict[str, list[str]] = {}
        inserts_by_table: dict[str, list[str]] = {}
        copy_blocks: list[tuple[str, str]] = []  # (Tabelle, Block)
        misc: list[str] = []

        for statement in statements:
            stripped = statement.strip()
            if not stripped:
                continue
            if _SQL_COPY_HEADER_RE.match(stripped.split("\n", 1)[0]):
                match = _SQL_COPY_RE.match(stripped)
                table = _normalize_table_name(match.group(1)) if match else "unbekannt"
                copy_blocks.append((table, stripped))
                continue
            classified = False
            for pattern, target in (
                (_SQL_CREATE_TABLE_RE, schema_by_table),
                (_SQL_ALTER_RE, schema_by_table),
                (_SQL_INDEX_RE, schema_by_table),
                (_SQL_COMMENT_RE, schema_by_table),
                (_SQL_INSERT_RE, inserts_by_table),
            ):
                match = pattern.match(stripped)
                if match:
                    target.setdefault(_normalize_table_name(match.group(1)), []).append(stripped)
                    classified = True
                    break
            if not classified:
                misc.append(stripped)

        chunks: list[Chunk] = []
        schema_chunk_ids: dict[str, str] = {}

        for table, schema_statements in schema_by_table.items():
            schema_text = f"-- Schema für Tabelle: {table}\n" + "\n\n".join(schema_statements)
            has_data = table in inserts_by_table or any(t == table for t, _ in copy_blocks)
            schema_chunk = Chunk(
                content=schema_text,
                chunk_type=ChunkType.SQL_STATEMENT,
                role=ChunkRole.PARENT if has_data else ChunkRole.STANDALONE,
                # Schemata sind auch als Parents durchsuchbar – sie beantworten
                # Struktur-Fragen ("Welche Spalten hat X?") direkt.
                searchable=True,
                extra={"table": table, "kind": "schema"},
            )
            chunks.append(schema_chunk)
            schema_chunk_ids[table] = schema_chunk.chunk_id

        for table, insert_statements in inserts_by_table.items():
            parent_id = schema_chunk_ids.get(table)
            for batch_index, batch in enumerate(
                self._batch_statements(insert_statements), start=1
            ):
                header = f"-- Daten für Tabelle: {table} (Batch {batch_index})\n"
                chunks.append(
                    Chunk(
                        content=header + "\n".join(batch),
                        chunk_type=ChunkType.SQL_STATEMENT,
                        role=ChunkRole.CHILD if parent_id else ChunkRole.STANDALONE,
                        parent_id=parent_id,
                        extra={
                            "table": table,
                            "kind": "insert_batch",
                            "statement_count": len(batch),
                        },
                    )
                )

        for table, block in copy_blocks:
            parent_id = schema_chunk_ids.get(table)
            for part in self._split_copy_block(block):
                chunks.append(
                    Chunk(
                        content=part,
                        chunk_type=ChunkType.SQL_STATEMENT,
                        role=ChunkRole.CHILD if parent_id else ChunkRole.STANDALONE,
                        parent_id=parent_id,
                        extra={"table": table, "kind": "copy_data"},
                    )
                )

        if misc:
            for batch in self._batch_statements(misc):
                chunks.append(
                    Chunk(
                        content="-- Allgemeine Dump-Anweisungen\n" + "\n".join(batch),
                        chunk_type=ChunkType.SQL_STATEMENT,
                        extra={"kind": "misc"},
                    )
                )
        return chunks

    def _batch_statements(self, statements: list[str]) -> list[list[str]]:
        """Bündelt Statements greedy zu Batches <= max_tokens; Riesen-Statements einzeln."""
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for statement in statements:
            tokens = estimate_tokens(statement)
            if current and current_tokens + tokens > self._max_tokens:
                batches.append(current)
                current, current_tokens = [], 0
            current.append(statement)
            current_tokens += tokens
            if tokens > self._max_tokens:
                batches.append(current)
                current, current_tokens = [], 0
        if current:
            batches.append(current)
        return batches

    def _split_copy_block(self, block: str) -> list[str]:
        lines = block.split("\n")
        header = lines[0]
        data_lines = [line for line in lines[1:] if line.strip() != "\\."]
        max_chars = self._max_tokens * CHARS_PER_TOKEN
        parts: list[str] = []
        current: list[str] = []
        current_size = len(header)
        for line in data_lines:
            if current and current_size + len(line) > max_chars:
                parts.append("\n".join([header, *current, "\\."]))
                current, current_size = [], len(header)
            current.append(line)
            current_size += len(line) + 1
        if current or not parts:
            parts.append("\n".join([header, *current, "\\."]))
        return parts

    def _split_statements(self, content: str) -> list[str]:
        """Zerlegt einen Dump in Statements; COPY-Datenblöcke werden atomar erfasst."""
        statements: list[str] = []
        pending: list[str] = []
        lines = content.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            if _SQL_COPY_HEADER_RE.match(line):
                if pending:
                    statements.extend(self._char_split("\n".join(pending)))
                    pending = []
                block = [line]
                i += 1
                while i < len(lines) and lines[i].strip() != "\\.":
                    block.append(lines[i])
                    i += 1
                if i < len(lines):
                    block.append(lines[i])
                    i += 1
                statements.append("\n".join(block))
                continue
            pending.append(line)
            i += 1
        if pending:
            statements.extend(self._char_split("\n".join(pending)))
        return [s for s in statements if s.strip()]

    @staticmethod
    def _char_split(text: str) -> list[str]:
        statements: list[str] = []
        buffer: list[str] = []
        i = 0
        length = len(text)
        in_single = in_double = in_backtick = False
        dollar_tag: Optional[str] = None
        while i < length:
            char = text[i]
            nxt = text[i + 1] if i + 1 < length else ""

            if dollar_tag is not None:
                buffer.append(char)
                if char == "$" and text[i : i + len(dollar_tag)] == dollar_tag:
                    buffer.extend(dollar_tag[1:])
                    i += len(dollar_tag)
                    dollar_tag = None
                    continue
                i += 1
                continue
            if in_single:
                buffer.append(char)
                if char == "'" and nxt == "'":
                    buffer.append(nxt)
                    i += 2
                    continue
                if char == "'":
                    in_single = False
                i += 1
                continue
            if in_double:
                buffer.append(char)
                if char == '"':
                    in_double = False
                i += 1
                continue
            if in_backtick:
                buffer.append(char)
                if char == "`":
                    in_backtick = False
                i += 1
                continue
            if char == "-" and nxt == "-":
                # Zeilenkommentar bis Zeilenende übernehmen
                end = text.find("\n", i)
                end = length if end == -1 else end
                buffer.append(text[i:end])
                i = end
                continue
            if char == "/" and nxt == "*":
                end = text.find("*/", i + 2)
                end = length if end == -1 else end + 2
                buffer.append(text[i:end])
                i = end
                continue
            if char == "'":
                in_single = True
            elif char == '"':
                in_double = True
            elif char == "`":
                in_backtick = True
            elif char == "$":
                tag_match = re.match(r"\$[A-Za-z_]*\$", text[i:])
                if tag_match:
                    dollar_tag = tag_match.group(0)
                    buffer.append(dollar_tag)
                    i += len(dollar_tag)
                    continue
            elif char == ";":
                buffer.append(char)
                statements.append("".join(buffer).strip())
                buffer = []
                i += 1
                continue
            buffer.append(char)
            i += 1
        trailing = "".join(buffer).strip()
        if trailing:
            statements.append(trailing)
        return statements


# ---------------------------------------------------------------------------
# PDF-Loading (pdfplumber) – Text + Tabellen-Extraktion
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PdfTable:
    page: int
    markdown: str
    context: str


def _table_to_markdown(rows: Sequence[Sequence[Any]]) -> str:
    """Konvertiert eine extrahierte PDF-Tabelle in eine Markdown-Tabelle."""
    cleaned: list[list[str]] = []
    for row in rows:
        if row is None:
            continue
        cleaned.append(
            [
                ("" if cell is None else str(cell).replace("\n", " ").replace("|", "\\|").strip())
                for cell in row
            ]
        )
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if len(cleaned) < 2:
        return ""
    width = max(len(row) for row in cleaned)
    for row in cleaned:
        row.extend([""] * (width - len(row)))
    lines = ["| " + " | ".join(cleaned[0]) + " |"]
    lines.append("|" + "|".join([" --- "] * width) + "|")
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _load_pdf(path: Path) -> tuple[str, list[_PdfTable]]:
    """Extrahiert Fließtext (mit Seitenmarkern) und Tabellen aus einem PDF."""
    try:
        import pdfplumber  # type: ignore[import-untyped]
    except ImportError as exc:
        raise DocumentLoadError(
            "PDF-Verarbeitung benötigt das Paket 'pdfplumber' "
            "(Installation: pip install 'rag-module[pdf]')."
        ) from exc

    page_texts: list[str] = []
    tables: list[_PdfTable] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                page_texts.append(f"[Seite {page_number}]\n{text}")
                try:
                    raw_tables = page.extract_tables() or []
                except Exception as table_error:  # noqa: BLE001 – Tabellen sind best effort
                    logger.warning(
                        "Tabellen-Extraktion auf Seite %d von '%s' fehlgeschlagen: %s",
                        page_number,
                        path.name,
                        table_error,
                    )
                    raw_tables = []
                for raw_table in raw_tables:
                    markdown = _table_to_markdown(raw_table)
                    if markdown:
                        context = " ".join(text.split())[:300]
                        tables.append(_PdfTable(page_number, markdown, context))
    except DocumentLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 – kaputte PDFs sauber melden
        raise DocumentLoadError(f"PDF '{path}' konnte nicht gelesen werden: {exc}") from exc
    return "\n\n".join(page_texts), tables


# ---------------------------------------------------------------------------
# Engine: Dispatch + Nachbearbeitung
# ---------------------------------------------------------------------------


class ChunkingEngine:
    """Fassade über alle Chunking-Strategien inkl. Datei-Loading und Nachbearbeitung.

    Die Nachbearbeitung stellt Invarianten sicher:
    - Kein Nicht-Tabellen-Chunk überschreitet sein Token-Budget wesentlich
      (Code-Einheiten dürfen 2x, Text 1,5x groß sein, bevor nachgeteilt wird).
    - Überharte Tabellen (> ``table_hard_max_tokens``) werden zeilenweise geteilt,
      wobei Kopfzeilen in jedem Teil wiederholt werden.
    - Sequenznummern sind fortlaufend und eindeutig (Basis für deterministische
      Punkt-IDs in der Vektordatenbank).
    """

    def __init__(self, settings: RAGSettings) -> None:
        self._settings = settings
        common = dict(
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            parent_max_tokens=settings.parent_max_tokens,
        )
        structured = dict(common, min_tokens=settings.min_chunk_tokens)
        self._markdown = MarkdownChunker(**structured)
        self._legal = LegalChunker(**structured)
        self._plain = PlainTextChunker(**structured)
        self._code = CodeChunker(**common)
        self._json = JSONChunker(**common)
        self._sql = SQLDumpChunker(**common)

    async def chunk_file(
        self,
        file_path: str | Path,
        document_type: DocumentType,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Asynchroner Einstiegspunkt; CPU-Arbeit läuft in einem Worker-Thread."""
        return await asyncio.to_thread(self.chunk_file_sync, file_path, document_type, metadata)

    def chunk_file_sync(
        self,
        file_path: str | Path,
        document_type: DocumentType,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        path = Path(file_path)
        if not path.is_file():
            raise DocumentLoadError(f"Datei nicht gefunden: {path}")

        is_pdf_file = path.suffix.lower() == ".pdf"
        legal_domain = str(metadata.get("domain", "")).lower() in {
            "legal",
            "contract",
            "law",
            "vertrag",
            "gesetz",
        }

        if document_type is DocumentType.PDF or (
            is_pdf_file and document_type in (DocumentType.LEGAL, DocumentType.TEXT)
        ):
            text, pdf_tables = _load_pdf(path)
            inner = (
                self._legal
                if document_type is DocumentType.LEGAL or legal_domain
                else self._plain
            )
            chunks = inner.chunk(text, source_name=path.name)
            for table in pdf_tables:
                chunks.append(
                    Chunk(
                        content=(
                            f"[Kontext: Seite {table.page}"
                            + (f" – {table.context}" if table.context else "")
                            + f"]\n\n{table.markdown}"
                        ),
                        chunk_type=ChunkType.TABLE,
                        hierarchy=[f"Seite {table.page}"],
                        extra={"page": table.page, "isolated_structure": True},
                    )
                )
        else:
            content = self._read_text(path)
            chunker = self._select_chunker(document_type)
            chunks = chunker.chunk(content, source_name=path.name)

        chunks = self._post_process(chunks)
        if not chunks:
            raise ChunkingError(f"Aus '{path.name}' konnten keine Chunks erzeugt werden.")
        return chunks

    def _select_chunker(self, document_type: DocumentType) -> BaseChunker:
        mapping: dict[DocumentType, BaseChunker] = {
            DocumentType.MARKDOWN: self._markdown,
            DocumentType.LEGAL: self._legal,
            DocumentType.JSON: self._json,
            DocumentType.API_PAYLOAD: self._json,
            DocumentType.CODE: self._code,
            DocumentType.SQL_DUMP: self._sql,
            DocumentType.TEXT: self._plain,
        }
        return mapping[document_type]

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("'%s' ist kein UTF-8 – Fallback auf latin-1.", path.name)
            return path.read_text(encoding="latin-1")
        except OSError as exc:
            raise DocumentLoadError(f"Datei '{path}' konnte nicht gelesen werden: {exc}") from exc

    def _post_process(self, chunks: list[Chunk]) -> list[Chunk]:
        settings = self._settings
        processed: list[Chunk] = []
        for chunk in chunks:
            chunk.content = chunk.content.strip()
            if not chunk.content:
                continue
            chunk.token_estimate = estimate_tokens(chunk.content)
            if (
                chunk.chunk_type is ChunkType.TABLE
                and chunk.token_estimate > settings.table_hard_max_tokens
            ):
                processed.extend(self._split_table_chunk(chunk))
            elif (
                chunk.role is not ChunkRole.PARENT
                and chunk.token_estimate > self._oversize_limit(chunk)
            ):
                processed.extend(self._resplit_chunk(chunk))
            else:
                processed.append(chunk)
        for sequence, chunk in enumerate(processed):
            chunk.sequence = sequence
        return processed

    def _oversize_limit(self, chunk: Chunk) -> int:
        factor = 2.0 if chunk.chunk_type is ChunkType.CODE_UNIT else 1.5
        return int(self._settings.chunk_max_tokens * factor)

    def _resplit_chunk(self, chunk: Chunk) -> list[Chunk]:
        pieces = split_text(
            chunk.content,
            self._settings.chunk_max_tokens,
            self._settings.chunk_overlap_tokens,
            min_tokens=self._settings.min_chunk_tokens,
        )
        out: list[Chunk] = []
        for index, piece in enumerate(pieces, start=1):
            out.append(
                Chunk(
                    content=piece,
                    chunk_type=chunk.chunk_type,
                    role=chunk.role,
                    parent_id=chunk.parent_id,
                    hierarchy=list(chunk.hierarchy),
                    searchable=chunk.searchable,
                    token_estimate=estimate_tokens(piece),
                    extra={**chunk.extra, "split_part": index, "split_total": len(pieces)},
                )
            )
        return out

    def _split_table_chunk(self, chunk: Chunk) -> list[Chunk]:
        """Teilt eine überharte Markdown-Tabelle zeilenweise; Header wird wiederholt."""
        lines = chunk.content.split("\n")
        table_start = next(
            (i for i, line in enumerate(lines) if _MD_TABLE_ROW.match(line)), None
        )
        if table_start is None or table_start + 1 >= len(lines):
            logger.warning(
                "Tabelle (Chunk %s) überschreitet das Limit, ist aber nicht zeilenweise "
                "teilbar (HTML/unbekanntes Format) – Fallback auf Text-Split.",
                chunk.chunk_id,
            )
            return self._resplit_chunk(chunk)

        preamble = "\n".join(lines[:table_start]).strip()
        header_lines = lines[table_start : table_start + 2]
        data_rows = lines[table_start + 2 :]
        max_chars = self._settings.table_hard_max_tokens * CHARS_PER_TOKEN
        base = (preamble + "\n\n" if preamble else "") + "\n".join(header_lines) + "\n"

        parts: list[str] = []
        current_rows: list[str] = []
        current_size = len(base)
        for row in data_rows:
            if current_rows and current_size + len(row) > max_chars:
                parts.append(base + "\n".join(current_rows))
                current_rows, current_size = [], len(base)
            current_rows.append(row)
            current_size += len(row) + 1
        if current_rows:
            parts.append(base + "\n".join(current_rows))

        out: list[Chunk] = []
        for index, part in enumerate(parts, start=1):
            out.append(
                Chunk(
                    content=f"[Tabellen-Teil {index}/{len(parts)}]\n{part}",
                    chunk_type=ChunkType.TABLE,
                    role=chunk.role,
                    parent_id=chunk.parent_id,
                    hierarchy=list(chunk.hierarchy),
                    token_estimate=estimate_tokens(part),
                    extra={**chunk.extra, "table_part": index, "table_parts_total": len(parts)},
                )
            )
        return out
