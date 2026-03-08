"""
Repetition detection (TASK-010) — second CHI signal.

Jaccard similarity on n-grams to detect when the model is repeating
(low entropy but saying the same thing over and over).
"""


def _text_to_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    """Split text into words and return set of n-grams (e.g. trigrams)."""
    words = text.split()
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def compute_repetition_score(text_a: str, text_b: str, n: int = 3) -> float:
    """
    Compute Jaccard similarity between two text outputs.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    where A and B are sets of n-grams (e.g., trigrams).

    Returns: 0.0 (completely different) to 1.0 (identical)
    """
    a = _text_to_ngrams(text_a, n)
    b = _text_to_ngrams(text_b, n)
    union = a | b
    if len(union) == 0:
        return 1.0 if (len(a) == 0 and len(b) == 0) else 0.0
    return len(a & b) / len(union)


def compute_self_repetition(text: str, window_size: int = 100, n: int = 3) -> float:
    """
    Check if the model is repeating itself WITHIN a single response.

    Split the text into windows (by word count) and check Jaccard similarity
    between consecutive windows.

    High self-repetition = the model is stuck in a loop.
    Returns: average pairwise Jaccard between consecutive windows.
    """
    words = text.split()
    if len(words) < 2 * window_size:
        return 0.0
    scores = []
    for i in range(0, len(words) - window_size, window_size):
        w1 = words[i : i + window_size]
        w2 = words[i + window_size : i + 2 * window_size]
        if len(w2) < n:
            break
        window_a = " ".join(w1)
        window_b = " ".join(w2)
        scores.append(compute_repetition_score(window_a, window_b, n))
    return sum(scores) / len(scores) if scores else 0.0


def track_repetition_over_turns(responses: list[str], n: int = 3) -> list[float]:
    """
    Given a list of model responses (one per turn),
    compute Jaccard similarity between consecutive responses.

    Returns: list of similarity scores (length = len(responses) - 1).
    Rising similarity = model converging to repetitive output.
    """
    return [
        compute_repetition_score(responses[i], responses[i + 1], n)
        for i in range(len(responses) - 1)
    ]


if __name__ == "__main__":
    # Identical texts should give 1.0
    assert (
        compute_repetition_score("hello world foo bar", "hello world foo bar") == 1.0
    ), "Identical texts should be 1.0"

    # Completely different texts should give 0.0 (or very low)
    score = compute_repetition_score(
        "The quick brown fox jumps over the lazy dog",
        "Python is a programming language for data science",
    )
    print(f"Different texts: {score:.4f}")  # Should be near 0

    # Slightly similar texts should give mid-range
    score = compute_repetition_score(
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "def factorial(n): return 1 if n < 2 else n * factorial(n-1)",
    )
    print(f"Similar structure: {score:.4f}")  # Should be moderate (0.2-0.5)

    # Self-repetition and track over turns (smoke test)
    long_repeat = "word " * 250
    self_rep = compute_self_repetition(long_repeat, window_size=100)
    print(f"Self-repetition (repeated 'word'): {self_rep:.4f}")

    turns = ["First response here.", "Second response there.", "First response here."]
    track = track_repetition_over_turns(turns)
    print(f"Track over 3 turns: {[f'{t:.4f}' for t in track]}")

    print("All manual tests passed.")
