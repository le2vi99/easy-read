"""Unit tests for the progress (SQLite) module."""

from __future__ import annotations

import pytest
from pathlib import Path

from easy_read.reading import progress as prog


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test_progress.db"
    prog.init_db(db_path)
    return db_path


@pytest.fixture()
def sample_book(db: Path) -> prog.Book:
    return prog.register_book(db, "Test Book", "test-book", "/fake/path.pdf", 10)


@pytest.fixture()
def sample_index() -> list[dict]:
    return [
        {
            "chapter": 1, "section": 1,
            "section_title": "Introduction", "chapter_title": "Chapter One",
            "file": "ch01_s01_introduction.md", "word_count": 500,
        },
        {
            "chapter": 1, "section": 2,
            "section_title": "Deep Dive", "chapter_title": "Chapter One",
            "file": "ch01_s02_deep-dive.md", "word_count": 800,
        },
        {
            "chapter": 2, "section": 1,
            "section_title": "Next Chapter", "chapter_title": "Chapter Two",
            "file": "ch02_s01_next-chapter.md", "word_count": 600,
        },
    ]


# --- Schema / init ---

def test_init_db_creates_tables(db: Path) -> None:
    import sqlite3
    conn = sqlite3.connect(str(db))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert {"books", "chunks", "sessions"} <= tables


# --- Books ---

def test_register_book_returns_book(db: Path) -> None:
    book = prog.register_book(db, "My Book", "my-book", "/path/book.pdf", 5)
    assert book.title == "My Book"
    assert book.slug == "my-book"
    assert book.total_chunks == 5
    assert book.id > 0


def test_register_book_idempotent_on_reingest(db: Path) -> None:
    b1 = prog.register_book(db, "My Book", "my-book", "/path/book.pdf", 5)
    b2 = prog.register_book(db, "My Book Updated", "my-book", "/path/book.pdf", 7)
    assert b1.id == b2.id
    assert b2.total_chunks == 7


def test_get_book_by_slug(db: Path, sample_book: prog.Book) -> None:
    fetched = prog.get_book_by_slug(db, "test-book")
    assert fetched is not None
    assert fetched.id == sample_book.id


def test_get_book_by_slug_missing(db: Path) -> None:
    assert prog.get_book_by_slug(db, "nonexistent") is None


def test_list_books_empty(db: Path) -> None:
    assert prog.list_books(db) == []


def test_list_books_returns_all(db: Path) -> None:
    prog.register_book(db, "Book A", "book-a", "/a.pdf", 3)
    prog.register_book(db, "Book B", "book-b", "/b.pdf", 5)
    books = prog.list_books(db)
    assert len(books) == 2


# --- Chunks ---

def test_register_chunks(db: Path, sample_book: prog.Book, sample_index: list[dict]) -> None:
    records = prog.register_chunks(db, sample_book.id, sample_index)
    assert len(records) == 3
    assert all(r.book_id == sample_book.id for r in records)


def test_get_chunks_for_book(db: Path, sample_book: prog.Book, sample_index: list[dict]) -> None:
    prog.register_chunks(db, sample_book.id, sample_index)
    chunks = prog.get_chunks_for_book(db, sample_book.id)
    assert len(chunks) == 3
    # Ordered by chapter, section
    assert chunks[0].chapter == 1 and chunks[0].section == 1
    assert chunks[1].chapter == 1 and chunks[1].section == 2
    assert chunks[2].chapter == 2


def test_next_unread_chunk_returns_first(db: Path, sample_book: prog.Book, sample_index: list[dict]) -> None:
    prog.register_chunks(db, sample_book.id, sample_index)
    chunk = prog.next_unread_chunk(db, sample_book.id)
    assert chunk is not None
    assert chunk.chapter == 1
    assert chunk.section == 1


def test_next_unread_advances_after_session(db: Path, sample_book: prog.Book, sample_index: list[dict]) -> None:
    records = prog.register_chunks(db, sample_book.id, sample_index)
    first = records[0]
    prog.record_session(db, first.id, quiz_score=0.8)

    nxt = prog.next_unread_chunk(db, sample_book.id)
    assert nxt is not None
    assert nxt.id == records[1].id


def test_next_unread_none_when_all_read(db: Path, sample_book: prog.Book, sample_index: list[dict]) -> None:
    records = prog.register_chunks(db, sample_book.id, sample_index)
    for r in records:
        prog.record_session(db, r.id)
    assert prog.next_unread_chunk(db, sample_book.id) is None


# --- Sessions ---

def test_record_session_without_quiz(db: Path, sample_book: prog.Book, sample_index: list[dict]) -> None:
    records = prog.register_chunks(db, sample_book.id, sample_index)
    session = prog.record_session(db, records[0].id)
    assert session.chunk_id == records[0].id
    assert session.quiz_score is None
    assert session.answers_json is None


def test_record_session_with_quiz(db: Path, sample_book: prog.Book, sample_index: list[dict]) -> None:
    records = prog.register_chunks(db, sample_book.id, sample_index)
    answers = [{"question": "What is X?", "answer": "It is Y", "rating": "CORRECT", "feedback": "Good."}]
    session = prog.record_session(db, records[0].id, quiz_score=1.0, answers=answers)
    assert session.quiz_score == 1.0
    assert session.answers_json is not None
    import json
    assert json.loads(session.answers_json)[0]["rating"] == "CORRECT"


def test_get_session_for_chunk(db: Path, sample_book: prog.Book, sample_index: list[dict]) -> None:
    records = prog.register_chunks(db, sample_book.id, sample_index)
    prog.record_session(db, records[0].id, quiz_score=0.5)
    session = prog.get_session_for_chunk(db, records[0].id)
    assert session is not None
    assert session.quiz_score == 0.5


def test_get_session_for_unread_chunk(db: Path, sample_book: prog.Book, sample_index: list[dict]) -> None:
    records = prog.register_chunks(db, sample_book.id, sample_index)
    assert prog.get_session_for_chunk(db, records[0].id) is None


# --- Progress summary ---

def test_progress_zero_at_start(db: Path, sample_book: prog.Book, sample_index: list[dict]) -> None:
    prog.register_chunks(db, sample_book.id, sample_index)
    bp = prog.get_book_progress(db, sample_book.id)
    assert bp.read_chunks == 0
    assert bp.percent_complete == 0.0
    assert bp.avg_quiz_score is None


def test_progress_updates_after_sessions(db: Path, sample_book: prog.Book, sample_index: list[dict]) -> None:
    records = prog.register_chunks(db, sample_book.id, sample_index)
    prog.record_session(db, records[0].id, quiz_score=1.0)
    prog.record_session(db, records[1].id, quiz_score=0.5)

    bp = prog.get_book_progress(db, sample_book.id)
    assert bp.read_chunks == 2
    assert bp.avg_quiz_score == pytest.approx(0.75)


def test_progress_percent_complete(db: Path) -> None:
    # Register book with accurate total_chunks=3
    book = prog.register_book(db, "Test Book", "test-book", "/fake/path.pdf", 3)
    index = [
        {"chapter": 1, "section": 1, "section_title": "A", "chapter_title": "Ch1",
         "file": "ch01_s01_a.md", "word_count": 300},
        {"chapter": 1, "section": 2, "section_title": "B", "chapter_title": "Ch1",
         "file": "ch01_s02_b.md", "word_count": 300},
        {"chapter": 2, "section": 1, "section_title": "C", "chapter_title": "Ch2",
         "file": "ch02_s01_c.md", "word_count": 300},
    ]
    records = prog.register_chunks(db, book.id, index)
    prog.record_session(db, records[0].id)

    bp = prog.get_book_progress(db, book.id)
    assert bp.read_chunks == 1
    assert bp.percent_complete == pytest.approx(33.3, abs=0.1)
