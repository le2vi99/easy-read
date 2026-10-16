"""Unit tests for the chunker module."""

from __future__ import annotations

from easy_read.ingestion.chunker import Chunk, build_chunks
from easy_read.ingestion.pdf_extractor import StructuredSpan


def _span(text: str, kind: str, page: int = 0) -> StructuredSpan:
    return StructuredSpan(text=text, kind=kind, page_num=page)


def _make_spans(structure: list[tuple[str, str]]) -> list[StructuredSpan]:
    return [_span(text, kind) for kind, text in structure]


# --- Basic structure ---

def test_single_chapter_single_section():
    spans = _make_spans([
        ("h1", "Chapter One"),
        ("h2", "Introduction"),
        ("body", "word " * 300),
    ])
    chunks = build_chunks(spans, min_words=50, max_words=1500)
    assert len(chunks) == 1
    assert chunks[0].chapter == 1
    assert chunks[0].section == 1
    assert chunks[0].section_title == "Introduction"


def test_two_sections_same_chapter():
    spans = _make_spans([
        ("h1", "Chapter One"),
        ("h2", "Section A"),
        ("body", "word " * 300),
        ("h2", "Section B"),
        ("body", "word " * 300),
    ])
    chunks = build_chunks(spans, min_words=50, max_words=1500)
    assert len(chunks) == 2
    assert chunks[0].section_title == "Section A"
    assert chunks[1].section_title == "Section B"
    assert chunks[0].chapter == chunks[1].chapter == 1


def test_two_chapters():
    spans = _make_spans([
        ("h1", "Chapter One"),
        ("h2", "Sec 1.1"),
        ("body", "word " * 300),
        ("h1", "Chapter Two"),
        ("h2", "Sec 2.1"),
        ("body", "word " * 300),
    ])
    chunks = build_chunks(spans, min_words=50, max_words=1500)
    assert len(chunks) == 2
    assert chunks[0].chapter == 1
    assert chunks[1].chapter == 2
    # Section counter resets per chapter
    assert chunks[0].section == 1
    assert chunks[1].section == 1


def test_body_before_first_h2_is_discarded():
    """Body text before any H1 should not produce a chunk."""
    spans = _make_spans([
        ("body", "Copyright blurb " * 50),
        ("h1", "Chapter One"),
        ("h2", "Real Section"),
        ("body", "real content " * 300),
    ])
    chunks = build_chunks(spans, min_words=50, max_words=1500)
    assert all(c.section_title == "Real Section" for c in chunks)


# --- Merge short chunks ---

def test_short_chunk_merged_into_previous():
    spans = _make_spans([
        ("h1", "Chapter One"),
        ("h2", "Big Section"),
        ("body", "word " * 300),
        ("h2", "Tiny Section"),
        ("body", "word " * 50),   # below min_words=200
    ])
    chunks = build_chunks(spans, min_words=200, max_words=1500)
    assert len(chunks) == 1
    assert "Big Section" in chunks[0].section_title


def test_short_chunk_not_merged_across_chapters():
    spans = _make_spans([
        ("h1", "Chapter One"),
        ("h2", "Section A"),
        ("body", "word " * 300),
        ("h1", "Chapter Two"),
        ("h2", "Tiny Section"),
        ("body", "word " * 50),
    ])
    chunks = build_chunks(spans, min_words=200, max_words=1500)
    # Tiny section can't merge into ch1 section — different chapter
    assert len(chunks) == 2


# --- Split large chunks ---

def test_large_chunk_is_split():
    long_para = ("word " * 400 + "\n\n") * 5  # 2000 words across 5 paragraphs
    spans = _make_spans([
        ("h1", "Chapter One"),
        ("h2", "Big Section"),
        ("body", long_para),
    ])
    chunks = build_chunks(spans, min_words=50, max_words=1500)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.word_count <= 1500


# --- Slug generation ---

def test_chunk_slug_format():
    spans = _make_spans([
        ("h1", "Chapter One"),
        ("h2", "Hash Indexes & B-Trees!"),
        ("body", "word " * 300),
    ])
    chunks = build_chunks(spans, min_words=50, max_words=1500)
    assert chunks[0].slug.startswith("ch01_s01_")
    assert "hash-indexes" in chunks[0].slug


# --- Word count ---

def test_word_count_accuracy():
    spans = _make_spans([
        ("h1", "Chapter One"),
        ("h2", "Section"),
        ("body", "one two three four five"),
    ])
    chunks = build_chunks(spans, min_words=1, max_words=10000)
    assert chunks[0].word_count == 5
