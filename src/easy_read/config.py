"""Configuration loading and management."""

from __future__ import annotations

import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path.home() / ".easy-read"
CONFIG_PATH = DATA_DIR / "config.toml"
BOOKS_DIR = DATA_DIR / "books"
DB_PATH = DATA_DIR / "progress.db"
_TEMPLATE = Path(__file__).parent.parent.parent / "config_template.toml"


@dataclass(frozen=True)
class AiConfig:
    model: str
    base_url: str
    temperature: float
    timeout_seconds: int


@dataclass(frozen=True)
class ReadingConfig:
    words_per_minute: int
    quiz_questions: int
    min_chunk_words: int
    max_chunk_words: int


@dataclass(frozen=True)
class DisplayConfig:
    theme: str
    show_page_numbers: bool


@dataclass(frozen=True)
class Config:
    ai: AiConfig
    reading: ReadingConfig
    display: DisplayConfig


def bootstrap() -> None:
    """Create ~/.easy-read/ structure on first run."""
    DATA_DIR.mkdir(exist_ok=True)
    BOOKS_DIR.mkdir(exist_ok=True)
    if not CONFIG_PATH.exists():
        if _TEMPLATE.exists():
            shutil.copy(_TEMPLATE, CONFIG_PATH)
        else:
            # Write defaults inline if template not found (installed package)
            CONFIG_PATH.write_text(
                '[ai]\nmodel = "llama3.2"\nbase_url = "http://localhost:11434"\n'
                "temperature = 0.7\ntimeout_seconds = 60\n\n"
                "[reading]\nwords_per_minute = 200\nquiz_questions = 3\n"
                "min_chunk_words = 200\nmax_chunk_words = 1500\n\n"
                '[display]\ntheme = "dark"\nshow_page_numbers = true\n'
            )


def load() -> Config:
    """Load config from ~/.easy-read/config.toml, bootstrapping if needed."""
    bootstrap()
    raw = tomllib.loads(CONFIG_PATH.read_text())
    ai = raw.get("ai", {})
    reading = raw.get("reading", {})
    display = raw.get("display", {})
    return Config(
        ai=AiConfig(
            model=ai.get("model", "llama3.2"),
            base_url=ai.get("base_url", "http://localhost:11434"),
            temperature=float(ai.get("temperature", 0.7)),
            timeout_seconds=int(ai.get("timeout_seconds", 60)),
        ),
        reading=ReadingConfig(
            words_per_minute=int(reading.get("words_per_minute", 200)),
            quiz_questions=int(reading.get("quiz_questions", 3)),
            min_chunk_words=int(reading.get("min_chunk_words", 200)),
            max_chunk_words=int(reading.get("max_chunk_words", 1500)),
        ),
        display=DisplayConfig(
            theme=display.get("theme", "dark"),
            show_page_numbers=bool(display.get("show_page_numbers", True)),
        ),
    )
