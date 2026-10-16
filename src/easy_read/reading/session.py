"""Reading session: present a chunk, run quiz, save progress."""

from __future__ import annotations

import os
import select
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from easy_read.ai.ollama_client import OllamaClient, OllamaNotRunningError
from easy_read.quiz.quiz_engine import (
    AnswerResult, Question, REVEAL_PHRASES,
    evaluate_answer, generate_questions, reveal_answer,
)
from easy_read.reading import progress as prog

console = Console()


# ---------------------------------------------------------------------------
# Chunk display
# ---------------------------------------------------------------------------

def _estimate_read_minutes(word_count: int, wpm: int) -> str:
    minutes = max(1, round(word_count / wpm))
    return f"{minutes} min read"


def _display_chunk(
    chunk: prog.ChunkRecord,
    book: prog.Book,
    book_dir: Path,
    wpm: int,
    show_pages: bool,
) -> bool:
    """Render the chunk text in the terminal. Returns False if file missing."""
    file_path = book_dir / chunk.file_path
    if not file_path.exists():
        console.print(f"[red]Chunk file not found:[/red] {file_path}")
        console.print("[dim]Skipping this chunk.[/dim]")
        return False

    content = file_path.read_text(encoding="utf-8")
    # Strip YAML frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:].strip()

    subtitle_parts = [f"Ch {chunk.chapter}  ·  {chunk.title}"]
    if show_pages:
        # Re-read frontmatter for page numbers
        raw = file_path.read_text(encoding="utf-8")
        start_page = end_page = None
        for line in raw.splitlines():
            if line.startswith("start_page:"):
                start_page = line.split(":", 1)[1].strip()
            elif line.startswith("end_page:"):
                end_page = line.split(":", 1)[1].strip()
        if start_page and end_page:
            subtitle_parts.append(f"pp. {start_page}–{end_page}")
    subtitle_parts.append(_estimate_read_minutes(chunk.word_count, wpm))

    console.print()
    console.print(Panel(
        Markdown(content),
        title=f"[bold cyan]{book.title}[/bold cyan]",
        subtitle="  ·  ".join(subtitle_parts),
        border_style="cyan",
        padding=(1, 2),
    ))
    return True


# ---------------------------------------------------------------------------
# Quiz loop
# ---------------------------------------------------------------------------

def _run_quiz(
    client: OllamaClient,
    book: prog.Book,
    chunk: prog.ChunkRecord,
    chunk_text: str,
    n_questions: int,
    show_answers: bool = False,
) -> list[AnswerResult] | None:
    """Generate questions, prompt user answers, evaluate each.

    Returns:
        list[AnswerResult] on normal completion (may be empty).
        None if the user typed 'q' to quit mid-quiz (session should not be saved).
    """
    console.print()
    console.print(Rule(
        f"[bold yellow]Comprehension Check[/bold yellow]  [dim]·  Ch {chunk.chapter}: {chunk.title}[/dim]",
        style="yellow",
    ))

    with console.status("[yellow]Generating questions…[/yellow]"):
        try:
            questions = generate_questions(
                client,
                book_title=book.title,
                section_title=chunk.title,
                chunk_text=chunk_text,
                n=n_questions,
            )
        except OllamaNotRunningError as exc:
            console.print(f"[red]AI error:[/red] {exc}")
            return []

    if not questions:
        console.print("[dim]Could not generate questions. Skipping quiz.[/dim]")
        return []

    console.print("[dim]Type [bold]q[/bold] to exit the quiz without saving.[/dim]")

    results: list[AnswerResult] = []
    for q in questions:
        console.print()
        console.print(f"[bold yellow]Q{q.number}.[/bold yellow] {q.text}")
        try:
            user_answer = console.input("[dim]Your answer (or q to quit):[/dim] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Quiz interrupted.[/dim]")
            break

        if user_answer.lower() == "q":
            console.print("[dim]Quiz exited. Progress not saved for this chunk.[/dim]")
            return None

        if not user_answer:
            console.print("[dim]Skipped.[/dim]")
            continue

        # "I don't know" / reveal shortcut — skip evaluation, just show the answer
        if user_answer.lower().strip() in REVEAL_PHRASES:
            with console.status("[yellow]Fetching answer…[/yellow]"):
                try:
                    answer_text = reveal_answer(
                        client,
                        book_title=book.title,
                        section_title=chunk.title,
                        question=q.text,
                    )
                except OllamaNotRunningError as exc:
                    console.print(f"[red]AI error:[/red] {exc}")
                    continue
            console.print(f"  [dim]Answer:[/dim] {answer_text}")
            results.append(AnswerResult(
                question=q.text,
                user_answer=user_answer,
                rating="INCORRECT",
                feedback="Answer revealed.",
                score=0.0,
                correct_answer=answer_text,
            ))
            continue

        with console.status("[yellow]Evaluating…[/yellow]"):
            try:
                result = evaluate_answer(
                    client,
                    book_title=book.title,
                    section_title=chunk.title,
                    question=q.text,
                    user_answer=user_answer,
                )
            except OllamaNotRunningError as exc:
                console.print(f"[red]AI error:[/red] {exc}")
                continue

        _rating_color = {"CORRECT": "green", "PARTIAL": "yellow", "INCORRECT": "red"}
        color = _rating_color.get(result.rating, "white")
        console.print(f"  [{color}]{result.rating}[/{color}]  {result.feedback}")
        if result.correct_answer and (show_answers or result.rating != "CORRECT"):
            console.print(f"  [dim]Answer:[/dim] {result.correct_answer}")
        results.append(result)

    return results


def _show_quiz_summary(results: list[AnswerResult]) -> float | None:
    """Print a score summary table. Returns score as 0.0-1.0 or None."""
    if not results:
        return None

    score = sum(r.score for r in results) / len(results)

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Question")
    table.add_column("Rating", justify="center", width=10)

    for i, r in enumerate(results, 1):
        color = {"CORRECT": "green", "PARTIAL": "yellow", "INCORRECT": "red"}.get(r.rating, "white")
        table.add_row(str(i), r.question[:60], f"[{color}]{r.rating}[/{color}]")

    console.print()
    console.print(Rule("[bold]Quiz Summary[/bold]"))
    console.print(table)

    pct = int(score * 100)
    color = "green" if pct >= 70 else "yellow" if pct >= 40 else "red"
    console.print(f"  Score: [{color}]{pct}%[/{color}]")
    return score


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def _read_nav_key() -> str:
    """Read a single keypress for chunk navigation.

    Returns one of: 'left', 'right', 'enter', 'jump', 'quit'.
    Falls back to line input on non-Unix systems.
    """
    try:
        import tty
        import termios
    except ImportError:
        # Non-Unix fallback
        try:
            raw = sys.stdin.readline().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "quit"
        if raw in ("q", "quit"):
            return "quit"
        if raw in ("h", "b", "back", "prev", "left", "<"):
            return "left"
        if raw in ("j", "jump"):
            return "jump"
        return "enter"

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    ch = b""
    try:
        tty.setraw(fd)
        # Use os.read() to bypass Python's buffered IO — select.select() checks
        # the OS-level fd, which BufferedReader.read() can silently drain ahead,
        # making select think no data is available even when bytes are buffered.
        ch = os.read(fd, 1)
        if ch == b"\x1b":
            ready, _, _ = select.select([sys.stdin], [], [], 0.15)
            if ready:
                ch2 = os.read(fd, 1)
                if ch2 == b"[":
                    ready2, _, _ = select.select([sys.stdin], [], [], 0.15)
                    if ready2:
                        ch3 = os.read(fd, 1)
                        if ch3 == b"D":
                            return "left"
                        if ch3 == b"C":
                            return "right"
            return "enter"  # bare Escape → treat as enter
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    if ch in (b"\x03", b"\x04"):   # Ctrl+C / Ctrl+D
        return "quit"
    if ch in (b"\r", b"\n"):
        return "enter"
    if ch in (b"q", b"Q"):
        return "quit"
    if ch in (b"s", b"S"):
        return "skip"
    if ch in (b"j", b"J"):
        return "jump"
    if ch == b"h":                  # vim-style left
        return "left"
    if ch in (b"l", b" "):          # vim-style right / space
        return "right"
    return "enter"


def _show_nav_bar(current_idx: int, total: int) -> None:
    """Print the post-read navigation hint line."""
    console.print(
        f"\n[dim]Chunk [bold]{current_idx + 1}[/bold] / {total}  ·  "
        "[bold]←[/bold]/[bold]h[/bold] prev  "
        "[bold]→[/bold]/[bold]l[/bold]/[bold]Enter[/bold] next  "
        "[bold]j[/bold] jump  "
        "[bold]q[/bold] quit[/dim]"
    )


def _show_pre_quiz_bar(current_idx: int, total: int, already_read: bool) -> None:
    """Print the pre-quiz navigation hint line."""
    re_tag = "[dim]Re-reading · [/dim]" if already_read else ""
    console.print(
        f"\n{re_tag}[dim]Chunk [bold]{current_idx + 1}[/bold] / {total}  ·  "
        "[bold]←[/bold]/[bold]h[/bold] prev  "
        "[bold]→[/bold]/[bold]Enter[/bold] quiz  "
        "[bold]s[/bold] skip  "
        "[bold]j[/bold] jump  "
        "[bold]q[/bold] quit[/dim]"
    )


def _prompt_jump(
    all_chunks: list[prog.ChunkRecord],
    current_idx: int,
) -> int | None:
    """Show a chunk picker and return the selected index, or None to cancel."""
    console.print()
    console.print(Rule("[bold]Jump to Chunk[/bold]"))

    prev_ch: int | None = None
    for i, c in enumerate(all_chunks):
        if c.chapter != prev_ch:
            console.print(f"\n  [bold cyan]Chapter {c.chapter}[/bold cyan]")
            prev_ch = c.chapter
        marker = "  [green]◀ current[/green]" if i == current_idx else ""
        console.print(f"    [dim]{i + 1:>3}.[/dim]  {c.title}{marker}")

    console.print()
    try:
        raw = console.input(
            "[dim]Enter chunk number or search text (blank to cancel):[/dim] "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not raw:
        return None

    # Try as integer index
    try:
        n = int(raw)
        if 1 <= n <= len(all_chunks):
            return n - 1
        console.print(f"[red]Out of range (1–{len(all_chunks)})[/red]")
        return None
    except ValueError:
        pass

    # Try as title substring search
    query = raw.lower()
    matches = [i for i, c in enumerate(all_chunks) if query in c.title.lower()]
    if not matches:
        console.print(f"[red]No chunks matching '{raw}'[/red]")
        return None
    if len(matches) == 1:
        return matches[0]

    # Multiple matches — show them and ask for a number
    console.print(f"[yellow]{len(matches)} matches:[/yellow]")
    for idx in matches[:8]:
        c = all_chunks[idx]
        console.print(f"  [dim]{idx + 1:>3}.[/dim]  Ch {c.chapter} · {c.title}")
    if len(matches) > 8:
        console.print(f"  [dim]…and {len(matches) - 8} more[/dim]")
    try:
        num_raw = console.input("[dim]Enter chunk number:[/dim] ").strip()
        n = int(num_raw)
        if 1 <= n <= len(all_chunks):
            return n - 1
    except (ValueError, EOFError, KeyboardInterrupt):
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionConfig:
    db_path: Path
    books_dir: Path
    wpm: int
    n_questions: int
    show_pages: bool
    quiz_enabled: bool
    show_answers: bool = False


def run(
    book_slug: str,
    client: OllamaClient | None,
    config: SessionConfig,
    chapter: int | None = None,
) -> None:
    """Main reading session loop with chunk navigation.

    Starts at the first unread chunk. Arrow keys (or h/l) move between chunks;
    j opens a jump picker; Enter/→ advances; ← goes back.
    """
    prog.init_db(config.db_path)

    book = prog.get_book_by_slug(config.db_path, book_slug)
    if not book:
        console.print(f"[red]Book not found:[/red] {book_slug}")
        console.print("Run [bold]easy-read books[/bold] to see available books.")
        return

    book_dir = config.books_dir / book_slug
    all_chunks = prog.get_chunks_for_book(config.db_path, book.id)

    if chapter is not None:
        all_chunks = [c for c in all_chunks if c.chapter == chapter]
        if not all_chunks:
            console.print(f"[red]Chapter {chapter} not found in '{book.title}'.[/red]")
            return

    if not all_chunks:
        console.print("[red]No chunks found.[/red]")
        return

    read_ids = prog.get_read_chunk_ids(config.db_path, book.id)

    # Start at first unread chunk; if all read, start at 0 for review
    start_idx = next(
        (i for i, c in enumerate(all_chunks) if c.id not in read_ids),
        0,
    )
    all_read = all(c.id in read_ids for c in all_chunks)
    if all_read:
        scope = f"chapter {chapter}" if chapter else "book"
        console.print(f"[green]You've finished the {scope}![/green] 🎉")
        bp = prog.get_book_progress(config.db_path, book.id)
        console.print(f"Total progress: [cyan]{bp.read_chunks}[/cyan] / {bp.total_chunks} chunks")
        console.print("[dim]You can still navigate back through any chunk to review.[/dim]")

    _interrupted = False

    def _handle_sigint(sig, frame):  # noqa: ANN001
        nonlocal _interrupted
        _interrupted = True
        console.print("\n\n[dim]Saving progress and exiting…[/dim]")

    old_handler = signal.signal(signal.SIGINT, _handle_sigint)

    current_idx = start_idx

    try:
        while 0 <= current_idx < len(all_chunks):
            if _interrupted:
                break

            chunk = all_chunks[current_idx]
            already_read = chunk.id in read_ids

            ok = _display_chunk(chunk, book, book_dir, config.wpm, config.show_pages)
            if not ok:
                prog.record_session(config.db_path, chunk.id)
                read_ids = read_ids | {chunk.id}
                current_idx += 1
                continue

            # Pre-quiz navigation bar — navigate without recording progress
            _show_pre_quiz_bar(current_idx, len(all_chunks), already_read)
            pre_key = _read_nav_key()
            console.print()  # newline after raw keypress

            if pre_key == "quit":
                break
            elif pre_key == "left":
                current_idx = max(0, current_idx - 1)
                continue
            elif pre_key == "jump":
                target = _prompt_jump(all_chunks, current_idx)
                if target is not None:
                    current_idx = target
                continue

            # enter / right → quiz; skip → record without quiz
            skip_quiz = pre_key == "skip"

            quiz_score: float | None = None
            answer_dicts: list[dict] = []
            quiz_aborted = False

            if config.quiz_enabled and not skip_quiz and client is not None:
                file_path = book_dir / chunk.file_path
                chunk_text = ""
                if file_path.exists():
                    raw = file_path.read_text(encoding="utf-8")
                    if raw.startswith("---"):
                        end = raw.find("---", 3)
                        chunk_text = raw[end + 3:].strip() if end != -1 else raw
                    else:
                        chunk_text = raw

                best_score: float | None = None
                best_answer_dicts: list[dict] = []

                while True:
                    results = _run_quiz(client, book, chunk, chunk_text, config.n_questions, config.show_answers)
                    if results is None:
                        quiz_aborted = True
                        break

                    attempt_score = _show_quiz_summary(results)
                    attempt_dicts = [
                        {
                            "question": r.question,
                            "answer": r.user_answer,
                            "rating": r.rating,
                            "feedback": r.feedback,
                        }
                        for r in results
                    ]

                    if attempt_score is not None and (best_score is None or attempt_score > best_score):
                        best_score = attempt_score
                        best_answer_dicts = attempt_dicts

                    passed = best_score is not None and best_score >= 0.7
                    if not passed and best_score is not None and not _interrupted:
                        console.print("[dim]Score below 70%. Type [bold]r[/bold] to retry · [bold]Enter[/bold] to continue.[/dim]")
                        try:
                            retry_choice = console.input("").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            break
                        if retry_choice == "r":
                            continue
                    break

                if not quiz_aborted:
                    quiz_score = best_score
                    answer_dicts = best_answer_dicts

            if quiz_aborted:
                break

            prog.record_session(
                config.db_path,
                chunk.id,
                quiz_score=quiz_score,
                answers=answer_dicts if answer_dicts else None,
            )
            read_ids = read_ids | {chunk.id}

            bp = prog.get_book_progress(config.db_path, book.id)
            console.print(
                f"\n[dim]Progress: {bp.read_chunks}/{bp.total_chunks} chunks "
                f"({bp.percent_complete}%)[/dim]"
            )

            if _interrupted:
                break

            # Navigation bar — single keypress
            _show_nav_bar(current_idx, len(all_chunks))
            key = _read_nav_key()
            console.print()  # newline after raw keypress

            if key == "quit":
                break
            elif key == "left":
                current_idx = max(0, current_idx - 1)
            elif key == "jump":
                target = _prompt_jump(all_chunks, current_idx)
                if target is not None:
                    current_idx = target
                # Cancelled → stay; loop will redisplay current chunk
            else:  # enter / right / space / anything else → advance
                current_idx += 1

    finally:
        signal.signal(signal.SIGINT, old_handler)
