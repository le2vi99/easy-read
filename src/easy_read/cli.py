"""CLI entry point for easy-read."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from easy_read import __version__
from easy_read import config as cfg

app = typer.Typer(
    name="easy-read",
    help="Read technical books in digestible chunks with AI comprehension questions.",
    add_completion=True,
)
console = Console()


def _complete_book_slug(incomplete: str) -> list[str]:
    """Tab-completion callback: returns matching book slugs."""
    try:
        from easy_read.reading import progress as prog
        cfg.bootstrap()
        prog.init_db(cfg.DB_PATH)
        books = prog.list_books(cfg.DB_PATH)
        return [b.slug for b in books if b.slug.startswith(incomplete)]
    except Exception:
        return []


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"easy-read [bold]{__version__}[/bold]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", callback=_version_callback, is_eager=True
    ),
) -> None:
    pass


@app.command()
def config() -> None:
    """Show current configuration."""
    c = cfg.load()
    table = Table(title=f"Config: {cfg.CONFIG_PATH}", show_header=True)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("ai.model", c.ai.model)
    table.add_row("ai.base_url", c.ai.base_url)
    table.add_row("ai.temperature", str(c.ai.temperature))
    table.add_row("ai.timeout_seconds", str(c.ai.timeout_seconds))
    table.add_row("reading.words_per_minute", str(c.reading.words_per_minute))
    table.add_row("reading.quiz_questions", str(c.reading.quiz_questions))
    table.add_row("reading.min_chunk_words", str(c.reading.min_chunk_words))
    table.add_row("reading.max_chunk_words", str(c.reading.max_chunk_words))
    table.add_row("display.theme", c.display.theme)
    table.add_row("display.show_page_numbers", str(c.display.show_page_numbers))

    console.print(table)


@app.command()
def ingest(
    pdf_path: Path = typer.Argument(..., help="Path to the PDF book file."),
    title: str = typer.Option("", "--title", "-t", help="Override book title (auto-detected if omitted)."),
) -> None:
    """Extract a PDF book into digestible section chunks."""
    from easy_read.ingestion import chunker, pdf_extractor, storage
    from easy_read.reading import progress as prog

    if not pdf_path.exists():
        console.print(f"[red]File not found:[/red] {pdf_path}")
        raise typer.Exit(1)

    c = cfg.load()
    prog.init_db(cfg.DB_PATH)

    # Step 1: Extract
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task("Extracting text from PDF…", total=None)
        spans, total_pages = pdf_extractor.extract(pdf_path)
        p.update(task, description=f"Extracted {len(spans)} spans from {total_pages} pages.")

    # Step 2: Chunk
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        task = p.add_task("Splitting into sections…", total=None)
        chunks = chunker.build_chunks(
            spans,
            min_words=c.reading.min_chunk_words,
            max_words=c.reading.max_chunk_words,
        )
        p.update(task, description=f"Created {len(chunks)} chunks.")

    if not chunks:
        console.print("[red]No chunks extracted.[/red] The PDF heading detection may need tuning.")
        raise typer.Exit(1)

    book_title = title or chunks[0].chapter_title or pdf_path.stem

    # Step 3: Save to disk
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as p:
        task = p.add_task("Saving chunks…", total=len(chunks))
        book_dir, index = storage.save_chunks(chunks, book_title, cfg.BOOKS_DIR)
        p.update(task, advance=len(chunks))

    # Step 4: Register in DB
    from easy_read.ingestion.storage import _book_slug
    slug = _book_slug(book_title)
    book = prog.register_book(cfg.DB_PATH, book_title, slug, str(pdf_path), len(chunks))
    prog.register_chunks(cfg.DB_PATH, book.id, index)

    console.print(f"\n[green]Done![/green] [bold]{book_title}[/bold]")
    console.print(f"  Chunks  : [cyan]{len(chunks)}[/cyan]")
    console.print(f"  Location: [dim]{book_dir}[/dim]")

    table = Table(title="Chapter Summary", show_header=True)
    table.add_column("Ch", style="cyan", justify="right")
    table.add_column("Title", style="white")
    table.add_column("Sections", justify="right")
    chapter_sections: dict[int, tuple[str, int]] = {}
    for entry in index:
        ch = entry["chapter"]
        ct = entry["chapter_title"]
        prev = chapter_sections.get(ch, (ct, 0))
        chapter_sections[ch] = (ct, prev[1] + 1)
    for ch_num in sorted(chapter_sections):
        ch_title, sec_count = chapter_sections[ch_num]
        table.add_row(str(ch_num), ch_title, str(sec_count))
    console.print(table)


@app.command()
def read(
    book_slug: str | None = typer.Argument(None, help="Book slug or partial name. Omit to pick interactively.", autocompletion=_complete_book_slug),
    chapter: int = typer.Option(None, "--chapter", "-c", help="Jump to a specific chapter number."),
    no_quiz: bool = typer.Option(False, "--no-quiz", help="Skip comprehension questions."),
    show_answers: bool = typer.Option(False, "--show-answers", "-a", help="Always reveal the correct answer after each question."),
) -> None:
    """Start or resume a reading session."""
    from easy_read.ai.ollama_client import OllamaNotRunningError, from_config
    from easy_read.reading.session import SessionConfig, run

    slug = _resolve_book_slug(book_slug, "Which book do you want to read?")
    if not slug:
        raise typer.Exit(0)
    book_slug = slug

    c = cfg.load()

    # Validate Ollama unless quiz is disabled
    client = None
    if not no_quiz:
        client = from_config(c.ai)
        try:
            client.health_check()
        except OllamaNotRunningError as exc:
            console.print(f"[red]Ollama error:[/red] {exc}")
            console.print("[dim]Tip: run with --no-quiz to read without AI questions.[/dim]")
            raise typer.Exit(1)

    session_cfg = SessionConfig(
        db_path=cfg.DB_PATH,
        books_dir=cfg.BOOKS_DIR,
        wpm=c.reading.words_per_minute,
        n_questions=c.reading.quiz_questions,
        show_pages=c.display.show_page_numbers,
        quiz_enabled=not no_quiz,
        show_answers=show_answers,
    )

    run(
        book_slug=book_slug,
        client=client,
        config=session_cfg,
        chapter=chapter,
    )


@app.command()
def books() -> None:
    """List all ingested books with reading progress."""
    from easy_read.reading import progress as prog

    prog.init_db(cfg.DB_PATH)
    book_list = prog.list_books(cfg.DB_PATH)

    if not book_list:
        console.print("[dim]No books ingested yet. Run [bold]easy-read ingest <pdf>[/bold] to get started.[/dim]")
        return

    table = Table(title="Your Books", show_header=True)
    table.add_column("Slug", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Progress", justify="right")
    table.add_column("Avg Score", justify="right")

    for book in book_list:
        bp = prog.get_book_progress(cfg.DB_PATH, book.id)
        pct = f"{bp.read_chunks}/{bp.total_chunks} ({bp.percent_complete}%)"
        score = f"{bp.avg_quiz_score:.0%}" if bp.avg_quiz_score is not None else "—"
        table.add_row(book.slug, book.title, pct, score)

    console.print(table)


@app.command()
def status(
    book_slug: str | None = typer.Argument(None, help="Book slug or partial name. Omit to pick interactively.", autocompletion=_complete_book_slug),
) -> None:
    """Show detailed reading progress for a book."""
    from easy_read.reading import progress as prog

    slug = _resolve_book_slug(book_slug, "Which book?")
    if not slug:
        raise typer.Exit(0)

    prog.init_db(cfg.DB_PATH)
    book = prog.get_book_by_slug(cfg.DB_PATH, slug)
    if not book:
        console.print(f"[red]Book not found:[/red] {slug}")
        raise typer.Exit(1)

    bp = prog.get_book_progress(cfg.DB_PATH, book.id)
    chunks = prog.get_chunks_for_book(cfg.DB_PATH, book.id)

    console.print(f"\n[bold]{book.title}[/bold]")
    console.print(f"  Progress : [cyan]{bp.read_chunks}[/cyan] / {bp.total_chunks} chunks "
                  f"([green]{bp.percent_complete}%[/green])")
    if bp.avg_quiz_score is not None:
        console.print(f"  Avg Score: [yellow]{bp.avg_quiz_score:.0%}[/yellow]")
    console.print()

    # Chapter-level breakdown
    table = Table(title="Chapter Progress", show_header=True)
    table.add_column("Ch", style="cyan", justify="right")
    table.add_column("Title", style="white")
    table.add_column("Read", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Status", justify="center")

    # Group by chapter
    from itertools import groupby
    chapters: dict[int, list] = {}
    for chunk in chunks:
        chapters.setdefault(chunk.chapter, []).append(chunk)

    read_ids = prog.get_read_chunk_ids(cfg.DB_PATH, book.id)

    for ch_num in sorted(chapters):
        ch_chunks = chapters[ch_num]
        read = sum(1 for c in ch_chunks if c.id in read_ids)
        total = len(ch_chunks)
        ch_display = _chapter_title_for(ch_chunks)
        if read == 0:
            status_icon = "[dim]○ not started[/dim]"
        elif read == total:
            status_icon = "[green]✓ complete[/green]"
        else:
            status_icon = "[yellow]◐ in progress[/yellow]"
        table.add_row(str(ch_num), ch_display, str(read), str(total), status_icon)

    console.print(table)


@app.command()
def quiz(
    book_slug: str | None = typer.Argument(None, help="Book slug or partial name. Omit to pick interactively.", autocompletion=_complete_book_slug),
    chapter: int = typer.Option(None, "--chapter", "-c", help="Quiz a specific chapter only."),
) -> None:
    """Re-quiz yourself on already-read chunks."""
    from easy_read.ai.ollama_client import OllamaNotRunningError, from_config
    from easy_read.reading import progress as prog
    from easy_read.quiz import quiz_engine

    slug = _resolve_book_slug(book_slug, "Which book do you want to quiz?")
    if not slug:
        raise typer.Exit(0)

    c = cfg.load()
    prog.init_db(cfg.DB_PATH)

    book = prog.get_book_by_slug(cfg.DB_PATH, slug)
    if not book:
        console.print(f"[red]Book not found:[/red] {slug}")
        raise typer.Exit(1)

    client = from_config(c.ai)
    try:
        client.health_check()
    except OllamaNotRunningError as exc:
        console.print(f"[red]Ollama error:[/red] {exc}")
        raise typer.Exit(1)

    chunks = prog.get_chunks_for_book(cfg.DB_PATH, book.id)
    read_ids = prog.get_read_chunk_ids(cfg.DB_PATH, book.id)
    read_chunks = [
        ch for ch in chunks
        if ch.id in read_ids
        and (chapter is None or ch.chapter == chapter)
    ]

    if not read_chunks:
        scope = f"chapter {chapter}" if chapter else "book"
        console.print(f"[yellow]No read chunks found for {scope}.[/yellow]")
        console.print("Read some chunks first with [bold]easy-read read[/bold].")
        return

    book_dir = cfg.BOOKS_DIR / book_slug
    scores: list[float] = []

    import random
    import signal

    _stop = False
    def _handle_sigint(sig, frame):  # noqa: ANN001
        nonlocal _stop
        _stop = True
        console.print("\n[dim]Stopping quiz…[/dim]")
    old = signal.signal(signal.SIGINT, _handle_sigint)

    try:
        for chunk in random.sample(read_chunks, len(read_chunks)):
            if _stop:
                break

            console.print(f"\n[bold cyan]Ch {chunk.chapter} · {chunk.title}[/bold cyan]")

            file_path = book_dir / chunk.file_path
            if not file_path.exists():
                console.print("[dim]Chunk file missing, skipping.[/dim]")
                continue

            raw = file_path.read_text(encoding="utf-8")
            if raw.startswith("---"):
                end = raw.find("---", 3)
                chunk_text = raw[end + 3:].strip() if end != -1 else raw
            else:
                chunk_text = raw

            with console.status("[yellow]Generating questions…[/yellow]"):
                try:
                    questions = quiz_engine.generate_questions(
                        client, book.title, chunk.title, chunk_text, n=c.reading.quiz_questions
                    )
                except OllamaNotRunningError as exc:
                    console.print(f"[red]AI error:[/red] {exc}")
                    break

            for q in questions:
                if _stop:
                    break
                console.print(f"\n[bold yellow]Q{q.number}.[/bold yellow] {q.text}")
                try:
                    answer = console.input("[dim]Answer:[/dim] ").strip()
                except (EOFError, KeyboardInterrupt):
                    _stop = True
                    break

                if not answer:
                    continue

                with console.status("[yellow]Evaluating…[/yellow]"):
                    try:
                        result = quiz_engine.evaluate_answer(
                            client, book.title, chunk.title, q.text, answer
                        )
                    except OllamaNotRunningError as exc:
                        console.print(f"[red]AI error:[/red] {exc}")
                        _stop = True
                        break

                color = {"CORRECT": "green", "PARTIAL": "yellow", "INCORRECT": "red"}.get(result.rating, "white")
                console.print(f"  [{color}]{result.rating}[/{color}]  {result.feedback}")
                scores.append(result.score)

            if not _stop:
                try:
                    cont = console.input("\n[dim]Next chunk? [bold]Enter[/bold] · [bold]q[/bold] to quit:[/dim] ").strip().lower()
                    if cont == "q":
                        break
                except (EOFError, KeyboardInterrupt):
                    break
    finally:
        signal.signal(signal.SIGINT, old)

    if scores:
        avg = sum(scores) / len(scores)
        color = "green" if avg >= 0.7 else "yellow" if avg >= 0.4 else "red"
        console.print(f"\n[bold]Session score:[/bold] [{color}]{avg:.0%}[/{color}] over {len(scores)} answers")


def _pick_book(prompt_text: str = "Select a book") -> str | None:
    """Interactive book picker. Returns slug or None if no books / user cancels."""
    from easy_read.reading import progress as prog

    prog.init_db(cfg.DB_PATH)
    book_list = prog.list_books(cfg.DB_PATH)

    if not book_list:
        console.print("[dim]No books ingested yet. Run [bold]easy-read ingest <pdf>[/bold] first.[/dim]")
        return None

    console.print(f"\n[bold]{prompt_text}[/bold]")
    for i, book in enumerate(book_list, 1):
        bp = prog.get_book_progress(cfg.DB_PATH, book.id)
        pct = f"{bp.percent_complete}%" if bp.read_chunks > 0 else "not started"
        console.print(f"  [cyan]{i}[/cyan]  {book.title}  [dim]({pct})[/dim]")

    console.print()
    try:
        raw = console.input("[dim]Enter number (or q to quit):[/dim] ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if raw.lower() == "q":
        return None
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(book_list):
            return book_list[idx].slug
    except ValueError:
        pass

    console.print("[red]Invalid selection.[/red]")
    return None


def _resolve_book_slug(slug: str | None, prompt_text: str = "Select a book") -> str | None:
    """Return a valid slug: resolve partial match, or show picker if slug is None."""
    from easy_read.reading import progress as prog

    if slug is None:
        return _pick_book(prompt_text)

    prog.init_db(cfg.DB_PATH)
    # Try exact match first
    book = prog.get_book_by_slug(cfg.DB_PATH, slug)
    if book:
        return book.slug

    # Try fuzzy match
    book = prog.find_book(cfg.DB_PATH, slug)
    if book:
        console.print(f"[dim]Matched:[/dim] [cyan]{book.title}[/cyan]")
        return book.slug

    console.print(f"[red]No book found matching:[/red] {slug}")
    console.print("Run [bold]easy-read books[/bold] to see available slugs.")
    return None


def _chapter_title_for(chunks: list) -> str:
    """Derive a display chapter title from the chunk file path."""
    # chunk.file_path looks like ch02_s01_reliable-scalable-and-maintainable-...md
    # Use the section_title of the chunk with section==1 (intro) or fallback
    intro = next((c for c in chunks if c.section == 1), chunks[0])
    # Convert slug back to title-case for display
    parts = intro.file_path.replace(".md", "").split("_")
    if len(parts) >= 3:
        return parts[2].replace("-", " ").title()
    return intro.title
