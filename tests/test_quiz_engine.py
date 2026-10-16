"""Unit tests for the quiz engine (no live Ollama required)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from easy_read.quiz.quiz_engine import (
    AnswerResult,
    Question,
    _parse_evaluation,
    _parse_questions,
    evaluate_answer,
    generate_questions,
)


# ---------------------------------------------------------------------------
# _parse_questions
# ---------------------------------------------------------------------------

def test_parse_questions_numbered_dot() -> None:
    raw = "1. What is reliability?\n2. How does scalability work?\n3. Define maintainability."
    result = _parse_questions(raw, 3)
    assert len(result) == 3
    assert result[0].text == "What is reliability?"
    assert result[1].number == 2
    assert result[2].text == "Define maintainability."


def test_parse_questions_numbered_paren() -> None:
    raw = "1) First question\n2) Second question"
    result = _parse_questions(raw, 3)
    assert len(result) == 2
    assert result[0].text == "First question"


def test_parse_questions_respects_n_limit() -> None:
    raw = "1. Q1\n2. Q2\n3. Q3\n4. Q4\n5. Q5"
    result = _parse_questions(raw, 3)
    assert len(result) == 3


def test_parse_questions_fallback_on_no_numbers() -> None:
    raw = "What is a fault?\nHow does replication help?\nWhat is a leader?"
    result = _parse_questions(raw, 3)
    assert len(result) == 3
    assert result[0].text == "What is a fault?"


def test_parse_questions_empty_response() -> None:
    result = _parse_questions("", 3)
    assert result == []


def test_parse_questions_with_preamble() -> None:
    raw = "Here are your questions:\n1. What is X?\n2. Explain Y.\n3. Compare Z."
    result = _parse_questions(raw, 3)
    assert len(result) == 3
    assert "What is X?" in result[0].text


# ---------------------------------------------------------------------------
# _parse_evaluation
# ---------------------------------------------------------------------------

def test_parse_evaluation_correct() -> None:
    raw = "RATING: CORRECT\nFEEDBACK: Great answer, you understood the concept well."
    result = _parse_evaluation(raw, "Q?", "My answer")
    assert result.rating == "CORRECT"
    assert result.score == 1.0
    assert "Great answer" in result.feedback


def test_parse_evaluation_partial() -> None:
    raw = "RATING: PARTIAL\nFEEDBACK: You got the main idea but missed the trade-offs."
    result = _parse_evaluation(raw, "Q?", "My answer")
    assert result.rating == "PARTIAL"
    assert result.score == 0.5


def test_parse_evaluation_incorrect() -> None:
    raw = "RATING: INCORRECT\nFEEDBACK: That's not quite right — reliability is about faults."
    result = _parse_evaluation(raw, "Q?", "My answer")
    assert result.rating == "INCORRECT"
    assert result.score == 0.0


def test_parse_evaluation_case_insensitive() -> None:
    raw = "rating: correct\nfeedback: Well done!"
    result = _parse_evaluation(raw, "Q?", "answer")
    assert result.rating == "CORRECT"


def test_parse_evaluation_fallback_on_malformed() -> None:
    """Malformed response should not raise — returns PARTIAL with raw text."""
    raw = "I think this is good but not perfect."
    result = _parse_evaluation(raw, "Q?", "answer")
    assert result.rating == "PARTIAL"
    assert result.score == 0.5
    assert result.feedback  # non-empty


def test_parse_evaluation_preserves_question_and_answer() -> None:
    raw = "RATING: CORRECT\nFEEDBACK: Nice."
    result = _parse_evaluation(raw, "What is X?", "X is Y")
    assert result.question == "What is X?"
    assert result.user_answer == "X is Y"


# ---------------------------------------------------------------------------
# generate_questions (with mocked client)
# ---------------------------------------------------------------------------

def _mock_client(response: str) -> MagicMock:
    client = MagicMock()
    client.generate.return_value = response
    return client


def test_generate_questions_returns_questions() -> None:
    client = _mock_client("1. What is a fault?\n2. How is reliability measured?\n3. What is MTTR?")
    questions = generate_questions(client, "DDIA", "Reliability", "chunk text", n=3)
    assert len(questions) == 3
    assert isinstance(questions[0], Question)


def test_generate_questions_truncates_long_chunk() -> None:
    """Chunk longer than _MAX_CHUNK_CHARS should be truncated before sending."""
    from easy_read.quiz.quiz_engine import _MAX_CHUNK_CHARS

    client = _mock_client("1. Q1\n2. Q2\n3. Q3")
    long_text = "word " * 10000
    generate_questions(client, "Book", "Section", long_text, n=3)

    _, kwargs = client.generate.call_args
    # The prompt is passed as a positional arg
    prompt = client.generate.call_args[0][0]
    assert len(prompt) < len(long_text)


def test_generate_questions_n_passed_to_prompt() -> None:
    client = _mock_client("1. Q1\n2. Q2")
    generate_questions(client, "Book", "Section", "text", n=2)
    prompt = client.generate.call_args[0][0]
    assert "2" in prompt


# ---------------------------------------------------------------------------
# evaluate_answer (with mocked client)
# ---------------------------------------------------------------------------

def test_evaluate_answer_returns_result() -> None:
    client = _mock_client("RATING: CORRECT\nFEEDBACK: Excellent understanding.")
    result = evaluate_answer(client, "DDIA", "Reliability", "What is a fault?", "A thing that goes wrong")
    assert isinstance(result, AnswerResult)
    assert result.rating == "CORRECT"
    assert result.score == 1.0


def test_evaluate_answer_never_raises_on_bad_response() -> None:
    client = _mock_client("I have no idea how to respond to this.")
    result = evaluate_answer(client, "Book", "Sec", "Q?", "A")
    assert result is not None
    assert result.rating in ("CORRECT", "PARTIAL", "INCORRECT")
