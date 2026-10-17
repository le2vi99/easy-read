# easy-read

Read technical books in small pieces, then answer a few AI-written questions to check you understood. Runs in the terminal. The AI is a local Ollama model, so no API key and no data leaves your machine.

## How it works

1. `ingest` a PDF. The book is split by section headings into markdown chunks.
2. `read` shows one section at a time.
3. After each section, the model asks 3 questions and grades your answers.
4. Progress is saved in SQLite, so you resume where you stopped.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running on `localhost:11434` with a model pulled (default: `gemma4`)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
easy-read ingest path/to/book.pdf   # split a PDF into chunks
easy-read books                     # list books and progress
easy-read read <book>               # start or resume reading
easy-read read <book> --chapter 3   # jump to a chapter
easy-read read <book> --no-quiz     # read without questions
easy-read quiz <book>               # quiz on sections already read
easy-read status [book]             # detailed progress
easy-read config                    # show config path and values
```

## Config

Copy `config_template.toml` to `~/.easy-read/config.toml` (done on first run). Main options:

| Section | Key | Default | Meaning |
|---|---|---|---|
| `ai` | `model` | `gemma4` | Ollama model name |
| `ai` | `base_url` | `http://localhost:11434` | Ollama address |
| `reading` | `quiz_questions` | `3` | Questions per section |
| `reading` | `min_chunk_words` / `max_chunk_words` | `200` / `1500` | Chunk size limits |
| `display` | `theme` | `dark` | Terminal theme |

Data lives in `~/.easy-read/` (config, `progress.db`, and `books/`).

## Test

```bash
pytest
```

## Notes

Heading detection uses font size and boldness. PDFs with image headings or flat typography may split badly. See [PLAN.md](PLAN.md) for the design.
