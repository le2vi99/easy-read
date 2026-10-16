"""Quiz engine: generate comprehension questions and evaluate answers.

Uses two prompt templates:
  1. generate_questions  → numbered list of N questions
  2. evaluate_answer     → RATING + FEEDBACK for a single answer
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from easy_read.ai.ollama_client import OllamaClient


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    number: int
    text: str


@dataclass(frozen=True)
class AnswerResult:
    question: str
    user_answer: str
    rating: str          # "CORRECT" | "PARTIAL" | "INCORRECT"
    feedback: str
    score: float         # 1.0 | 0.5 | 0.0
    correct_answer: str  # model answer from the text


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_QUESTION_PROMPT = """\
You are a reading tutor. A student just finished reading the TEXT below.

TEXT:
{chunk_text}

---
Your task has two steps:

Step 1 — List the key concepts, terms, or ideas that are EXPLICITLY explained in the TEXT above (not just mentioned). Write each as a short phrase.

Step 2 — Using ONLY those concepts from Step 1, write exactly {n} comprehension questions a student should be able to answer after reading the TEXT. Do not use any knowledge beyond what is in the TEXT.

Output format (no extra text):
CONCEPTS:
- <concept 1>
- <concept 2>
...

QUESTIONS:
1. <question>
2. <question>
...
"""

_EVALUATE_PROMPT = """\
You are a reading tutor evaluating a student's understanding.

Book: "{book_title}"
Section: "{section_title}"
Question: {question}
Student's answer: {user_answer}

Evaluate the answer. Respond in EXACTLY this format with no extra text:
RATING: CORRECT | PARTIAL | INCORRECT
FEEDBACK: <one encouraging but honest sentence>
CORRECT_ANSWER: <a concise model answer based only on the book content>
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_QUESTION_RE = re.compile(r"^\s*\d+[.)]\s*(.+)", re.MULTILINE)
_RATING_RE = re.compile(r"RATING:\s*(CORRECT|PARTIAL|INCORRECT)", re.IGNORECASE)
_FEEDBACK_RE = re.compile(r"FEEDBACK:\s*(.+?)(?=CORRECT_ANSWER:|$)", re.IGNORECASE | re.DOTALL)
_CORRECT_ANSWER_RE = re.compile(r"CORRECT_ANSWER:\s*(.+)", re.IGNORECASE | re.DOTALL)

_SCORE_MAP = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}


def _parse_questions(raw: str, n: int) -> list[Question]:
    """Extract numbered questions from LLM output.

    Handles both the two-step format (CONCEPTS: / QUESTIONS: sections)
    and plain numbered lists as a fallback.
    """
    # If the model used the two-step format, extract only the QUESTIONS section
    questions_section = raw
    if "QUESTIONS:" in raw.upper():
        idx = raw.upper().index("QUESTIONS:")
        questions_section = raw[idx + len("QUESTIONS:"):]

    matches = _QUESTION_RE.findall(questions_section)
    questions = [Question(number=i + 1, text=m.strip()) for i, m in enumerate(matches)]
    if not questions:
        # Fallback: split non-empty lines that look like sentences
        lines = [ln.strip() for ln in questions_section.splitlines() if ln.strip() and "?" in ln]
        questions = [Question(number=i + 1, text=ln) for i, ln in enumerate(lines[:n])]
    if not questions:
        # Last resort: any non-empty line
        lines = [ln.strip() for ln in questions_section.splitlines() if ln.strip()]
        questions = [Question(number=i + 1, text=ln) for i, ln in enumerate(lines[:n])]
    return questions[:n]


def _parse_evaluation(raw: str, question: str, user_answer: str) -> AnswerResult:
    """Parse RATING/FEEDBACK/CORRECT_ANSWER from LLM output, with fallback for malformed responses."""
    rating_match = _RATING_RE.search(raw)
    feedback_match = _FEEDBACK_RE.search(raw)
    correct_answer_match = _CORRECT_ANSWER_RE.search(raw)

    if rating_match and feedback_match:
        rating = rating_match.group(1).upper()
        feedback = feedback_match.group(1).strip().splitlines()[0].strip()
    else:
        # Fallback: treat any response as feedback, mark as partial
        rating = "PARTIAL"
        feedback = raw.strip().splitlines()[0][:200] if raw.strip() else "Could not evaluate."

    correct_answer = ""
    if correct_answer_match:
        correct_answer = correct_answer_match.group(1).strip().splitlines()[0].strip()

    return AnswerResult(
        question=question,
        user_answer=user_answer,
        rating=rating,
        feedback=feedback,
        score=_SCORE_MAP.get(rating, 0.5),
        correct_answer=correct_answer,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Truncate chunk text sent to LLM to avoid overflowing context window
_MAX_CHUNK_CHARS = 3000

# Phrases that signal the user wants the answer revealed instead of evaluated
REVEAL_PHRASES: frozenset[str] = frozenset({
    "i don't know", "i dont know", "idk", "don't know", "dont know",
    "reveal", "show", "show answer", "show me", "tell me", "?", "??",
    "no idea", "not sure", "skip", "pass",
})

_REVEAL_PROMPT = """\
You are a reading tutor.

Book: "{book_title}"
Section: "{section_title}"
Question: {question}

Give a concise, accurate answer to this question based only on the book content.
Output ONLY the answer with no preamble, no labels, no extra text.
"""


def generate_questions(
    client: OllamaClient,
    book_title: str,
    section_title: str,
    chunk_text: str,
    n: int = 3,
) -> list[Question]:
    """Ask Ollama to generate n comprehension questions for a chunk.

    Returns a list of Question objects. May return fewer than n if the model
    output is unparseable, but never raises on parse failure.
    """
    truncated = chunk_text[:_MAX_CHUNK_CHARS]
    prompt = _QUESTION_PROMPT.format(
        n=n,
        chunk_text=truncated,
    )
    raw = client.generate(prompt)
    return _parse_questions(raw, n)


def evaluate_answer(
    client: OllamaClient,
    book_title: str,
    section_title: str,
    question: str,
    user_answer: str,
) -> AnswerResult:
    """Ask Ollama to evaluate the user's answer to a question.

    Always returns an AnswerResult — never raises on parse failure.
    """
    prompt = _EVALUATE_PROMPT.format(
        book_title=book_title,
        section_title=section_title,
        question=question,
        user_answer=user_answer,
    )
    raw = client.generate(prompt)
    return _parse_evaluation(raw, question, user_answer)


def reveal_answer(
    client: OllamaClient,
    book_title: str,
    section_title: str,
    question: str,
) -> str:
    """Ask Ollama for the correct answer without evaluating a student response.

    Returns the answer as a plain string, or empty string on failure.
    """
    prompt = _REVEAL_PROMPT.format(
        book_title=book_title,
        section_title=section_title,
        question=question,
    )
    try:
        return client.generate(prompt).strip().splitlines()[0].strip()
    except Exception:
        return ""
