"""Unit tests for cognition_engine.repetition."""

import pytest

from cognition_engine.repetition import (
    compute_repetition_score,
    compute_self_repetition,
    track_repetition_over_turns,
)


def test_identical_texts_jaccard_one():
    """Identical texts should give Jaccard 1.0."""
    a = "hello world foo bar"
    assert compute_repetition_score(a, a) == 1.0


def test_different_texts_jaccard_low():
    """Completely different texts should give Jaccard 0.0 or very low."""
    a = "The quick brown fox jumps over the lazy dog"
    b = "Python is a programming language for data science"
    assert compute_repetition_score(a, b) == 0.0


def test_empty_and_empty():
    """Two empty texts: Jaccard 1.0 (by convention)."""
    assert compute_repetition_score("", "") == 1.0


def test_track_over_turns_length():
    """track_repetition_over_turns returns len(responses)-1 scores."""
    responses = ["First.", "Second.", "Third."]
    scores = track_repetition_over_turns(responses)
    assert len(scores) == 2


def test_self_repetition_high_for_repeated():
    """Repeating the same word in long text gives high self-repetition."""
    long_repeat = "word " * 250
    score = compute_self_repetition(long_repeat, window_size=100)
    assert score >= 0.99


def test_identical_texts():
    """Identical texts should give Jaccard 1.0."""
    score = compute_repetition_score(
        "the quick brown fox jumps", "the quick brown fox jumps"
    )
    assert score == 1.0


def test_completely_different():
    """Completely different texts should give Jaccard < 0.1."""
    score = compute_repetition_score(
        "the quick brown fox jumps over the lazy dog today",
        "python is a great programming language for beginners",
    )
    assert score < 0.1


def test_self_repetition_repeated_text():
    """Highly repeated phrase should give high self-repetition score."""
    repeated = " ".join(["hello world foo bar"] * 20)
    score = compute_self_repetition(repeated, window_size=20)
    assert score > 0.8, f"Highly repeated text should score high, got {score}"
