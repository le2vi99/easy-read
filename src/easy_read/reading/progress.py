"""SQLite-based progress tracking.

All functions return new objects (immutable pattern) — no in-place mutation.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Book:
    id: int
    title: str
    slug: str
    pdf_path: str
    total_chunks: int
    ingested_at: str


@dataclass(frozen=True)
class ChunkRecord:
    id: int
    book_id: int
    chapter: int
    section: int
    title: str
    file_path: str
    word_count: int


@dataclass(frozen=True)
class Session:
    id: int
    chunk_id: int
    read_at: str
    quiz_score: float | None
    answers_json: str | None


@dataclass(frozen=True)
class BookProgress:
    book: Book
    total_chunks: int
    read_chunks: int
    avg_quiz_score: float | None

    @property
    def percent_complete(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return round(self.read_chunks / self.total_chunks * 100, 1)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT    NOT NULL,
    slug         TEXT    NOT NULL UNIQUE,
    pdf_path     TEXT    NOT NULL,
    total_chunks INTEGER NOT NULL,
    ingested_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id      INTEGER NOT NULL REFERENCES books(id),
    chapter      INTEGER NOT NULL,
    section      INTEGER NOT NULL,
    title        TEXT    NOT NULL,
    file_path    TEXT    NOT NULL,
    word_count   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id     INTEGER NOT NULL REFERENCES chunks(id),
    read_at      TEXT    NOT NULL,
    quiz_score   REAL,
    answers_json TEXT
);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


@contextmanager
def _connect(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def init_db(db_path: Path) -> None:
    """Create tables if they don't exist."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


# ---------------------------------------------------------------------------
# Books
# ---------------------------------------------------------------------------


def register_book(
    db_path: Path,
    title: str,
    slug: str,
    pdf_path: str,
    total_chunks: int,
) -> Book:
    """Insert or replace a book record. Returns the Book."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        # Remove stale chunks/sessions for this slug if re-ingesting
        existing = conn.execute(
            "SELECT id FROM books WHERE slug = ?", (slug,)
        ).fetchone()
        if existing:
            book_id = existing["id"]
            chunk_ids = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM chunks WHERE book_id = ?", (book_id,)
                ).fetchall()
            ]
            if chunk_ids:
                placeholders = ",".join("?" * len(chunk_ids))
                conn.execute(
                    f"DELETE FROM sessions WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                )
            conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
            conn.execute(
                "UPDATE books SET title=?, pdf_path=?, total_chunks=?, ingested_at=? WHERE id=?",
                (title, pdf_path, total_chunks, now, book_id),
            )
            row_id = book_id
        else:
            cur = conn.execute(
                "INSERT INTO books (title, slug, pdf_path, total_chunks, ingested_at) VALUES (?,?,?,?,?)",
                (title, slug, pdf_path, total_chunks, now),
            )
            row_id = cur.lastrowid

    return Book(
        id=row_id,
        title=title,
        slug=slug,
        pdf_path=pdf_path,
        total_chunks=total_chunks,
        ingested_at=now,
    )


def get_book_by_slug(db_path: Path, slug: str) -> Book | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM books WHERE slug = ?", (slug,)
        ).fetchone()
    if row is None:
        return None
    return Book(**dict(row))


def list_books(db_path: Path) -> list[Book]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM books ORDER BY ingested_at DESC").fetchall()
    return [Book(**dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


def register_chunks(
    db_path: Path,
    book_id: int,
    index_entries: list[dict],
) -> list[ChunkRecord]:
    """Bulk-insert chunk records from an index.json entry list."""
    with _connect(db_path) as conn:
        records: list[ChunkRecord] = []
        for entry in index_entries:
            cur = conn.execute(
                "INSERT INTO chunks (book_id, chapter, section, title, file_path, word_count) "
                "VALUES (?,?,?,?,?,?)",
                (
                    book_id,
                    entry["chapter"],
                    entry["section"],
                    entry["section_title"],
                    entry["file"],
                    entry["word_count"],
                ),
            )
            records.append(
                ChunkRecord(
                    id=cur.lastrowid,
                    book_id=book_id,
                    chapter=entry["chapter"],
                    section=entry["section"],
                    title=entry["section_title"],
                    file_path=entry["file"],
                    word_count=entry["word_count"],
                )
            )
    return records


def get_chunks_for_book(db_path: Path, book_id: int) -> list[ChunkRecord]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE book_id = ? ORDER BY chapter, section",
            (book_id,),
        ).fetchall()
    return [ChunkRecord(**dict(r)) for r in rows]


def next_unread_chunk(db_path: Path, book_id: int) -> ChunkRecord | None:
    """Return the first chunk for this book with no session record."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT c.*
            FROM chunks c
            LEFT JOIN sessions s ON s.chunk_id = c.id
            WHERE c.book_id = ? AND s.id IS NULL
            ORDER BY c.chapter, c.section
            LIMIT 1
            """,
            (book_id,),
        ).fetchone()
    return ChunkRecord(**dict(row)) if row else None


def get_chunk_by_id(db_path: Path, chunk_id: int) -> ChunkRecord | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
    return ChunkRecord(**dict(row)) if row else None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def record_session(
    db_path: Path,
    chunk_id: int,
    quiz_score: float | None = None,
    answers: list[dict] | None = None,
) -> Session:
    """Mark a chunk as read, optionally with quiz results."""
    now = datetime.now(timezone.utc).isoformat()
    answers_json = json.dumps(answers) if answers else None
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO sessions (chunk_id, read_at, quiz_score, answers_json) VALUES (?,?,?,?)",
            (chunk_id, now, quiz_score, answers_json),
        )
        session_id = cur.lastrowid
    return Session(
        id=session_id,
        chunk_id=chunk_id,
        read_at=now,
        quiz_score=quiz_score,
        answers_json=answers_json,
    )


def get_session_for_chunk(db_path: Path, chunk_id: int) -> Session | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE chunk_id = ? ORDER BY read_at DESC LIMIT 1",
            (chunk_id,),
        ).fetchone()
    return Session(**dict(row)) if row else None


# ---------------------------------------------------------------------------
# Progress summary
# ---------------------------------------------------------------------------


def find_book(db_path: Path, query: str) -> Book | None:
    """Find a book by exact slug, or by partial match against slug/title.

    Priority:
      1. Exact slug match
      2. Slug contains query
      3. Title contains query (case-insensitive)
    Returns None if no match or multiple ambiguous matches.
    """
    books = list_books(db_path)
    if not books:
        return None

    q = query.strip().lower()

    # 1. Exact slug
    for b in books:
        if b.slug == q:
            return b

    # 2. Slug contains query
    slug_matches = [b for b in books if q in b.slug]
    if len(slug_matches) == 1:
        return slug_matches[0]

    # 3. Title contains query
    title_matches = [b for b in books if q in b.title.lower()]
    if len(title_matches) == 1:
        return title_matches[0]

    # Ambiguous or not found
    return None


def get_read_chunk_ids(db_path: Path, book_id: int) -> set[int]:
    """Return IDs of all chunks for this book that have at least one session."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT c.id
            FROM chunks c
            JOIN sessions s ON s.chunk_id = c.id
            WHERE c.book_id = ?
            """,
            (book_id,),
        ).fetchall()
    return {r["id"] for r in rows}


def get_book_progress(db_path: Path, book_id: int) -> BookProgress:
    with _connect(db_path) as conn:
        book_row = conn.execute(
            "SELECT * FROM books WHERE id = ?", (book_id,)
        ).fetchone()
        stats = conn.execute(
            """
            SELECT
                COUNT(DISTINCT s.chunk_id)          AS read_count,
                AVG(s.quiz_score)                   AS avg_score
            FROM chunks c
            LEFT JOIN sessions s ON s.chunk_id = c.id
            WHERE c.book_id = ?
            """,
            (book_id,),
        ).fetchone()

    book = Book(**dict(book_row))
    return BookProgress(
        book=book,
        total_chunks=book.total_chunks,
        read_chunks=stats["read_count"] or 0,
        avg_quiz_score=stats["avg_score"],
    )
