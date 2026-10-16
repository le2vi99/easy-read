"""Persist chunks as markdown files and an index.json."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from easy_read.ingestion.chunker import Chunk


def _slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"[\s_]+", "-", s)[:60].rstrip("-")


def _book_slug(title: str) -> str:
    return _slugify(title)


def _frontmatter(chunk: Chunk, book_title: str) -> str:
    return (
        "---\n"
        f"book: {book_title}\n"
        f"chapter: {chunk.chapter}\n"
        f"chapter_title: {chunk.chapter_title}\n"
        f"section: {chunk.section}\n"
        f"section_title: {chunk.section_title}\n"
        f"start_page: {chunk.start_page + 1}\n"  # 1-indexed for display
        f"end_page: {chunk.end_page + 1}\n"
        f"word_count: {chunk.word_count}\n"
        "---\n\n"
    )


def save_chunks(
    chunks: list[Chunk],
    book_title: str,
    books_dir: Path,
) -> tuple[Path, list[dict]]:
    """Write chunks to disk and return (book_dir, index_entries).

    Args:
        chunks: Ordered list of Chunk objects.
        book_title: Human-readable book title (used for slug + frontmatter).
        books_dir: Parent directory, e.g. ~/.easy-read/books/

    Returns:
        (book_dir, index) where index is a list of dicts with chunk metadata.
    """
    slug = _book_slug(book_title)
    book_dir = books_dir / slug
    book_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict] = []

    for chunk in chunks:
        filename = f"{chunk.slug}.md"
        file_path = book_dir / filename
        content = _frontmatter(chunk, book_title) + chunk.text
        file_path.write_text(content, encoding="utf-8")

        index.append(
            {
                "slug": chunk.slug,
                "file": filename,
                "chapter": chunk.chapter,
                "chapter_title": chunk.chapter_title,
                "section": chunk.section,
                "section_title": chunk.section_title,
                "start_page": chunk.start_page + 1,
                "end_page": chunk.end_page + 1,
                "word_count": chunk.word_count,
            }
        )

    index_path = book_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    return book_dir, index
