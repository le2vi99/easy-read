"""Split structured spans into digestible section chunks.

Rules:
- H1 span → new chapter, reset section counter
- H2 span → new section chunk within current chapter
- Body spans accumulate into the current chunk's text
- Post-processing:
    - Filter out front-matter chapters (no H2 content or clearly decorative)
    - Merge chunks < min_words into previous
    - Split chunks > max_words at paragraph boundaries
    - Re-number sections sequentially to avoid collisions after splitting
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from easy_read.ingestion.pdf_extractor import StructuredSpan


@dataclass(frozen=True)
class Chunk:
    chapter: int
    chapter_title: str
    section: int
    section_title: str
    text: str
    start_page: int
    end_page: int

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def slug(self) -> str:
        def _slugify(s: str) -> str:
            s = s.lower().strip()
            s = re.sub(r"[^\w\s-]", "", s)
            return re.sub(r"[\s_]+", "-", s)[:50].rstrip("-")

        return f"ch{self.chapter:02d}_s{self.section:02d}_{_slugify(self.section_title)}"


def _merge_short(chunks: list[Chunk], min_words: int) -> list[Chunk]:
    """Merge chunks below min_words into the previous chunk."""
    if not chunks:
        return chunks
    result: list[Chunk] = [chunks[0]]
    for chunk in chunks[1:]:
        prev = result[-1]
        if chunk.word_count < min_words and chunk.chapter == prev.chapter:
            merged = Chunk(
                chapter=prev.chapter,
                chapter_title=prev.chapter_title,
                section=prev.section,
                section_title=prev.section_title,
                text=prev.text + "\n\n" + chunk.text,
                start_page=prev.start_page,
                end_page=chunk.end_page,
            )
            result[-1] = merged
        else:
            result.append(chunk)
    return result


def _split_large(chunk: Chunk, max_words: int) -> list[Chunk]:
    """Split a chunk exceeding max_words at paragraph boundaries."""
    if chunk.word_count <= max_words:
        return [chunk]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", chunk.text) if p.strip()]
    parts: list[Chunk] = []
    current_paras: list[str] = []
    current_words = 0
    sub_idx = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > max_words and current_paras:
            parts.append(
                Chunk(
                    chapter=chunk.chapter,
                    chapter_title=chunk.chapter_title,
                    section=chunk.section + sub_idx,
                    section_title=chunk.section_title
                    + (f" (part {sub_idx + 1})" if sub_idx > 0 else ""),
                    text="\n\n".join(current_paras),
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                )
            )
            sub_idx += 1
            current_paras = [para]
            current_words = para_words
        else:
            current_paras.append(para)
            current_words += para_words

    if current_paras:
        parts.append(
            Chunk(
                chapter=chunk.chapter,
                chapter_title=chunk.chapter_title,
                section=chunk.section + sub_idx,
                section_title=chunk.section_title
                + (f" (part {sub_idx + 1})" if sub_idx > 0 else ""),
                text="\n\n".join(current_paras),
                start_page=chunk.start_page,
                end_page=chunk.end_page,
            )
        )
    return parts if parts else [chunk]


_TOC_LINE_RE = re.compile(r"(\.\s){3,}|\.\.\.\s*\d+\s*$|\d{1,4}\s*$")


def _is_toc_chunk(chunk: Chunk, threshold: float = 0.35) -> bool:
    """Return True if this chunk looks like a table of contents page.

    Heuristic: if more than `threshold` fraction of non-empty lines match
    the dot-leader / trailing-page-number pattern, it's a ToC chunk.
    """
    lines = [ln.strip() for ln in chunk.text.splitlines() if ln.strip()]
    if not lines:
        return False
    toc_lines = sum(1 for ln in lines if _TOC_LINE_RE.search(ln))
    return toc_lines / len(lines) >= threshold


def build_chunks(
    spans: list[StructuredSpan],
    min_words: int = 200,
    max_words: int = 1500,
) -> list[Chunk]:
    """Convert structured spans into a list of Chunks."""
    chapter = 0
    chapter_title = ""
    section = 0
    current_section_title = ""
    current_text_parts: list[str] = []
    current_start_page = 0
    current_end_page = 0
    raw_chunks: list[Chunk] = []

    def _flush() -> None:
        nonlocal section, current_text_parts, current_start_page, current_end_page
        if not current_text_parts or chapter == 0:
            return
        # Join blocks with paragraph separator; blocks already have internal newlines
        text = "\n\n".join(current_text_parts).strip()
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if text:
            raw_chunks.append(
                Chunk(
                    chapter=chapter,
                    chapter_title=chapter_title,
                    section=section,
                    section_title=current_section_title or chapter_title,
                    text=text,
                    start_page=current_start_page,
                    end_page=current_end_page,
                )
            )
        current_text_parts.clear()

    last_was_h1 = False

    for span in spans:
        if span.kind == "h1":
            if last_was_h1:
                # Merge wrapped title (e.g. "Reliable, Scalable, and" + "Maintainable Applications")
                chapter_title = chapter_title + " " + span.text
                current_section_title = chapter_title
            else:
                _flush()
                # Discard any body text accumulated before the first H1
                current_text_parts.clear()
                chapter += 1
                chapter_title = span.text
                section = 0
                current_section_title = span.text
                current_start_page = span.page_num
                current_end_page = span.page_num
            last_was_h1 = True
        elif span.kind == "h2":
            _flush()
            section += 1
            current_section_title = span.text
            current_start_page = span.page_num
            current_end_page = span.page_num
            last_was_h1 = False
        else:
            current_text_parts.append(span.text)
            current_end_page = span.page_num
            last_was_h1 = False

    _flush()

    # Filter front-matter/back-matter: chapters with no H2 sections
    # (section == 0 means the chapter had only body text, no named sections)
    chapter_has_sections = {
        c.chapter for c in raw_chunks if c.section > 0
    }
    raw_chunks = [c for c in raw_chunks if c.chapter in chapter_has_sections]

    # Filter table-of-contents chunks: pages dominated by dot-leader lines
    # (e.g. "Linearizability . . . . . . 324")
    raw_chunks = [c for c in raw_chunks if not _is_toc_chunk(c)]

    # Re-number chapters sequentially after filtering
    old_to_new: dict[int, int] = {}
    new_ch = 0
    for c in raw_chunks:
        if c.chapter not in old_to_new:
            new_ch += 1
            old_to_new[c.chapter] = new_ch
    raw_chunks = [
        Chunk(
            chapter=old_to_new[c.chapter],
            chapter_title=c.chapter_title,
            section=c.section,
            section_title=c.section_title,
            text=c.text,
            start_page=c.start_page,
            end_page=c.end_page,
        )
        for c in raw_chunks
    ]

    # Post-process
    merged = _merge_short(raw_chunks, min_words)
    split: list[Chunk] = []
    for chunk in merged:
        split.extend(_split_large(chunk, max_words))

    # Re-number sections sequentially per chapter to avoid collisions after splitting
    final: list[Chunk] = []
    chapter_section_counter: dict[int, int] = {}
    for chunk in split:
        ch = chunk.chapter
        chapter_section_counter[ch] = chapter_section_counter.get(ch, 0) + 1
        sec = chapter_section_counter[ch]
        final.append(
            Chunk(
                chapter=chunk.chapter,
                chapter_title=chunk.chapter_title,
                section=sec,
                section_title=chunk.section_title,
                text=chunk.text,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
            )
        )

    return final
