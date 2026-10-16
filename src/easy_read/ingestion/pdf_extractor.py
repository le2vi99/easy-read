"""Extract structured text from PDF using PyMuPDF.

Heading detection strategy:
- Collect all font sizes across the document → frequency map
- Body text = most common font size
- H1 (chapter title) = largest font size appearing rarely (< 5% of spans)
- H2 (section) = second distinct large font size
- Bold flag (flags & 16) used as tiebreaker for same-size spans
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass(frozen=True)
class Span:
    text: str
    size: float
    bold: bool
    page_num: int


@dataclass(frozen=True)
class PageContent:
    page_num: int
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class HeadingThresholds:
    body_size: float
    h2_size: float
    h1_size: float


def _is_bold(flags: int) -> bool:
    return bool(flags & 16)


def _collect_spans(doc: fitz.Document) -> list[Span]:
    spans: list[Span] = []
    for page in doc:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:  # skip image blocks
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    spans.append(
                        Span(
                            text=text,
                            size=round(span["size"], 1),
                            bold=_is_bold(span["flags"]),
                            page_num=page.number,
                        )
                    )
    return spans


def _compute_thresholds(spans: list[Span]) -> HeadingThresholds:
    """Derive H1/H2/body font-size thresholds from span frequency.

    Candidates must:
    - Be larger than body text
    - Appear at least MIN_HEADING_COUNT times (filters decorative/cover fonts)
    - Appear in less than 15% of all spans (filters body-level text)
    """
    MIN_HEADING_COUNT = 5

    size_counts: dict[float, int] = {}
    for s in spans:
        size_counts[s.size] = size_counts.get(s.size, 0) + 1

    total = sum(size_counts.values())
    # Body = most common size
    body_size = max(size_counts, key=lambda k: size_counts[k])

    # Heading candidates: larger than body, frequent enough to be real headings,
    # but not so frequent they are body text
    candidates = sorted(
        [
            sz
            for sz, cnt in size_counts.items()
            if sz > body_size
            and cnt >= MIN_HEADING_COUNT
            and cnt / total < 0.15
        ],
        reverse=True,
    )

    h1_size = candidates[0] if candidates else body_size + 4
    h2_size = candidates[1] if len(candidates) > 1 else body_size + 2

    return HeadingThresholds(body_size=body_size, h2_size=h2_size, h1_size=h1_size)


def _classify_span(span: Span, thresholds: HeadingThresholds) -> str:
    """Return 'h1', 'h2', or 'body'."""
    if span.size >= thresholds.h1_size:
        return "h1"
    if span.size >= thresholds.h2_size and (span.bold or span.size > thresholds.body_size + 1):
        return "h2"
    return "body"


@dataclass
class StructuredSpan:
    text: str
    kind: str  # 'h1', 'h2', 'body'
    page_num: int


# H1 titles that should not start a chapter (front/back matter)
_SKIP_H1_TITLES = frozenset({
    "table of contents",
    "preface",
    "index",
    "glossary",
    "colophon",
    "about the author",
    "copyright",
    "references",
    "bibliography",
})

# H1 titles that are part markers, not chapter titles (e.g. "Foundations of Data Systems")
_PART_KEYWORDS = ("part i", "part ii", "part iii", "part iv", "part v")


def _is_frontmatter(text: str) -> bool:
    lower = text.strip().lower()
    return lower in _SKIP_H1_TITLES or lower.startswith(_PART_KEYWORDS)


def extract(pdf_path: Path) -> tuple[list[StructuredSpan], int]:
    """Extract structured spans from a PDF, preserving paragraph structure.

    Emits one StructuredSpan per PDF text block so that paragraph boundaries
    are preserved as '\\n\\n' in body text (required for large-chunk splitting).

    Returns:
        (structured_spans, total_pages)
    """
    doc = fitz.open(str(pdf_path))
    raw_spans = _collect_spans(doc)
    thresholds = _compute_thresholds(raw_spans)

    result: list[StructuredSpan] = []

    for page in doc:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue

            # Collect all spans in this block, classified
            block_lines: list[str] = []
            block_kind = "body"
            heading_text: str | None = None

            for line in block.get("lines", []):
                line_parts: list[str] = []
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    raw = Span(
                        text=text,
                        size=round(span["size"], 1),
                        bold=_is_bold(span["flags"]),
                        page_num=page.number,
                    )
                    kind = _classify_span(raw, thresholds)
                    if kind in ("h1", "h2"):
                        # Emit any accumulated body lines first
                        if block_lines:
                            body_text = "\n".join(block_lines).strip()
                            if body_text:
                                result.append(
                                    StructuredSpan(text=body_text, kind="body", page_num=page.number)
                                )
                            block_lines = []
                        # Emit heading span directly
                        if not (kind == "h1" and _is_frontmatter(text)):
                            result.append(StructuredSpan(text=text, kind=kind, page_num=page.number))
                    else:
                        line_parts.append(text)

                if line_parts:
                    block_lines.append(" ".join(line_parts))

            # Emit remaining body text as one block (preserves paragraph boundary)
            if block_lines:
                body_text = "\n".join(block_lines).strip()
                if body_text:
                    result.append(
                        StructuredSpan(text=body_text, kind="body", page_num=page.number)
                    )

    return result, doc.page_count
