# easy-read: Implementation Plan

## Overview

A Python CLI tool for reading technical books (PDF) in digestible chunks, with AI-powered comprehension questions using a local Ollama model.

---

## Goals

- Ingest a PDF book → split by section headings → store as markdown chunks
- Present one section at a time in the terminal
- After each section, generate 3 comprehension questions via local Ollama LLM
- Evaluate user answers and give feedback
- Track reading progress in SQLite (resume where you left off)

---

## Tech Stack

| Concern | Choice | Reason |
|---------|--------|--------|
| Language | Python 3.11+ | Best ecosystem for PDF + LLM + CLI |
| CLI framework | Typer + Rich | Clean UX, colors, panels |
| PDF parsing | PyMuPDF (fitz) | Gives font size/bold metadata for heading detection |
| AI backend | Ollama (HTTP) | Local LLM, no API key needed |
| HTTP client | httpx | Async-capable, clean API |
| Storage | Markdown files | Human-readable chunks, easy to inspect |
| Progress DB | SQLite (stdlib) | No extra deps, simple CRUD |
| Build | Hatchling + pyproject.toml | Modern Python packaging |

---

## Project Structure

```
easy-read/
├── src/
│   └── easy_read/
│       ├── __init__.py
│       ├── cli.py                      # Typer app, all commands
│       ├── config.py                   # Config loading (TOML → dataclass)
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── pdf_extractor.py        # PyMuPDF: page text + heading detection
│       │   ├── chunker.py              # Group pages into Chunk objects by section
│       │   └── storage.py             # Write .md files + index.json
│       ├── reading/
│       │   ├── __init__.py
│       │   ├── session.py              # Read → quiz → save loop
│       │   └── progress.py            # SQLite CRUD (immutable patterns)
│       ├── ai/
│       │   ├── __init__.py
│       │   └── ollama_client.py        # POST /api/generate, health check
│       └── quiz/
│           ├── __init__.py
│           └── quiz_engine.py          # Prompt templates, response parsing
├── tests/
│   ├── conftest.py
│   ├── test_chunker.py
│   ├── test_quiz_engine.py
│   └── test_progress.py
├── config_template.toml                # Copied to ~/.easy-read/config.toml on first run
├── pyproject.toml
└── PLAN.md                             # This file
```

---

## CLI Commands

```bash
easy-read ingest <pdf_path>         # Extract + store chunks into ~/.easy-read/books/
easy-read books                     # List all ingested books + progress
easy-read read <book>               # Start/resume reading session
  --chapter N                       # Jump to chapter N
  --no-quiz                         # Skip comprehension questions
easy-read status [book]             # Show detailed progress (chapters, % complete)
easy-read quiz <book>               # Quiz-only mode on already-read chunks
easy-read config                    # Print current config path + values
```

---

## Data Layout on Disk

```
~/.easy-read/
├── config.toml
├── progress.db                     # SQLite
└── books/
    └── designing-data-intensive-applications/
        ├── index.json              # Ordered chunk metadata
        ├── ch01_s01_thinking-about-data-systems.md
        ├── ch01_s02_reliability.md
        ├── ch01_s03_hardware-faults.md
        └── ...
```

**Chunk markdown format:**
```markdown
---
book: Designing Data-Intensive Applications
chapter: 1
chapter_title: Reliable, Scalable, and Maintainable Applications
section: 2
section_title: Reliability
start_page: 6
end_page: 9
word_count: 612
---

<section text>
```

---

## SQLite Schema

```sql
CREATE TABLE books (
  id           INTEGER PRIMARY KEY,
  title        TEXT NOT NULL,
  slug         TEXT NOT NULL UNIQUE,
  pdf_path     TEXT NOT NULL,
  total_chunks INTEGER NOT NULL,
  ingested_at  TIMESTAMP NOT NULL
);

CREATE TABLE chunks (
  id           INTEGER PRIMARY KEY,
  book_id      INTEGER NOT NULL REFERENCES books(id),
  chapter      INTEGER NOT NULL,
  section      INTEGER NOT NULL,
  title        TEXT NOT NULL,
  file_path    TEXT NOT NULL,
  word_count   INTEGER NOT NULL
);

CREATE TABLE sessions (
  id           INTEGER PRIMARY KEY,
  chunk_id     INTEGER NOT NULL REFERENCES chunks(id),
  read_at      TIMESTAMP NOT NULL,
  quiz_score   REAL,           -- NULL if quiz was skipped
  answers_json TEXT            -- JSON array of {question, answer, rating, feedback}
);
```

---

## Core Algorithms

### Heading Detection (pdf_extractor.py)

PyMuPDF returns text spans with font size and flags. Strategy:
1. Collect all font sizes across the document → build a frequency map
2. Body text = most common font size
3. H1 (chapter title) = largest font size that appears rarely (once per chapter)
4. H2 (section) = second largest, appears ~5–15 times per chapter
5. Flag 16 (bold) used as a tiebreaker

**Known risk**: Some PDFs use images for headings, or headings may not differ in font size. Detection may need tuning per book. → See [Open Issues](#open-issues).

### Chunking Strategy (chunker.py)

1. Walk pages in order, accumulate body text until an H2 heading is encountered
2. H2 heading starts a new chunk
3. H1 heading starts a new chapter and resets section counter
4. Post-process:
   - Merge chunks < 200 words into the previous chunk
   - Split chunks > 1500 words at paragraph boundaries (double newline)

### Quiz Prompt Design (quiz_engine.py)

**Generate questions:**
```
You are a reading tutor helping someone study "{book_title}".

Section: "{section_title}"

Generate exactly 3 comprehension questions about the key concepts below.
Focus on understanding and application, not memorization of facts.
Output only a numbered list (1. 2. 3.), no preamble.

TEXT:
{chunk_text}
```

**Evaluate answer:**
```
You are a reading tutor evaluating a student's understanding.

Book: "{book_title}"
Section: "{section_title}"
Question: {question}
Student's answer: {user_answer}

Evaluate the answer. Respond in exactly this format:
RATING: CORRECT | PARTIAL | INCORRECT
FEEDBACK: <one sentence, encouraging but honest>
```

---

## Configuration (`~/.easy-read/config.toml`)

```toml
[ai]
model = "llama3.2"
base_url = "http://localhost:11434"
temperature = 0.7
timeout_seconds = 60

[reading]
words_per_minute = 200
quiz_questions = 3
min_chunk_words = 200
max_chunk_words = 1500

[display]
theme = "dark"               # dark | light
show_page_numbers = true
```

---

## Implementation Phases

### Phase 1 — Scaffold
- [ ] `pyproject.toml` (deps, entry point `easy-read`)
- [ ] `src/easy_read/__init__.py`, `config.py`
- [ ] `config_template.toml`
- [ ] Bootstrap `~/.easy-read/` dir on first run

### Phase 2 — Ingestion
- [ ] `pdf_extractor.py`: heading detection + text extraction
- [ ] `chunker.py`: section splitting with merge/split rules
- [ ] `storage.py`: write .md chunks + index.json
- [ ] `easy-read ingest` command (with Rich progress bar)

### Phase 3 — Progress Tracking
- [ ] `progress.py`: SQLite schema creation + CRUD (immutable returns)
- [ ] Register book + chunks on ingest
- [ ] `easy-read books` and `easy-read status` commands

### Phase 4 — Ollama Client
- [ ] `ollama_client.py`: health check + `generate()`
- [ ] Config-driven model/base_url
- [ ] Clear error if Ollama not running

### Phase 5 — Quiz Engine
- [ ] `quiz_engine.py`: prompt templates + response parser
- [ ] Parse RATING/FEEDBACK from LLM output (with fallback if malformed)

### Phase 6 — Reading Session
- [ ] `session.py`: full read → quiz → save loop
- [ ] `easy-read read` command
- [ ] Graceful Ctrl+C (save progress before exit)

### Phase 7 — Polish
- [x] `easy-read quiz` standalone command
- [x] `easy-read config` display command
- [x] Batch read_ids query (no N+1 in status/quiz/session)
- [ ] End-to-end smoke test with DDIA PDF

---

## Open Issues

> Track problems discovered during implementation here.

| # | Issue | Status | Resolution |
|---|-------|--------|------------|
| 1 | Heading detection may fail on PDFs with non-standard fonts or image-based headings | Open | May need per-book config override: `heading_h1_size`, `heading_h2_size` |
| 2 | Very long sections (e.g. 3000+ word sections) may overwhelm LLM context | Open | Truncate chunk text to first 1000 words for quiz prompt; full text still saved |
| 3 | Ollama response may not follow RATING/FEEDBACK format exactly | Open | Add regex fallback; if parse fails, show raw response |
| 4 | Some PDF pages are mostly images (diagrams) with little text | Open | Skip chunks with <50 words; log skipped pages |
| 5 | `easy-read read` resumption: what if chunk file was deleted? | Open | Detect missing file in session.py, warn user + skip |

---

## Verification Checklist

- [ ] `easy-read ingest ddia.pdf` completes without error; markdown files exist in `~/.easy-read/books/`
- [ ] Spot-check 3 chunks: heading names match actual PDF sections
- [ ] `easy-read books` lists the ingested book with correct chunk count
- [ ] `easy-read read ddia` displays first unread chunk with Rich formatting
- [ ] After pressing Enter, 3 questions are generated by Ollama
- [ ] Answering questions produces RATING + FEEDBACK for each
- [ ] `easy-read status ddia` shows updated progress after session
- [ ] Ctrl+C during session resumes correctly on next `easy-read read`
- [ ] `pytest tests/` passes (chunker edge cases, quiz parser, progress DB)
