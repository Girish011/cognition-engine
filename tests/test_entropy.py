"""Unit tests for cognition_engine.entropy."""

import pytest
import mlx.core as mx

from cognition_engine.entropy import compute_token_entropy


def test_compute_token_entropy_shape():
    """Entropy output has shape (seq_len,) for input (1, seq_len, vocab_size)."""
    batch, seq_len, vocab = 1, 5, 100
    logits = mx.random.normal(shape=(batch, seq_len, vocab))
    entropy = compute_token_entropy(logits)
    assert entropy.shape == (seq_len,)


def test_compute_token_entropy_non_negative():
    """Shannon entropy is non-negative."""
    logits = mx.random.normal(shape=(1, 3, 50))
    entropy = compute_token_entropy(logits)
    mx.eval(entropy)
    flat = entropy.reshape(-1)
    for i in range(flat.size):
        assert float(flat[i].item()) >= 0.0


def test_compute_token_entropy_uniform_upper_bound():
    """For uniform distribution over vocab_size, entropy ≈ log2(vocab_size)."""
    import math
    vocab = 64
    # Logits that softmax to nearly uniform
    logits = mx.ones((1, 1, vocab)) * 0.1
    entropy = compute_token_entropy(logits)
    mx.eval(entropy)
    h = float(entropy[0].item())
    expected = math.log2(vocab)
    assert abs(h - expected) < 0.5


def test_uniform_distribution_high_entropy():
    """Uniform distribution over vocab should give high entropy."""
    vocab_size = 1000
    logits = mx.zeros((1, 1, vocab_size))  # All equal = uniform after softmax
    entropy = compute_token_entropy(logits)
    mx.eval(entropy)
    # log2(1000) ≈ 9.97, so entropy should be close to ~10
    assert entropy[0].item() > 9.0, (
        f"Uniform entropy should be high, got {entropy[0].item()}"
    )


def test_peaked_distribution_low_entropy():
    """One dominant logit should give low entropy."""
    vocab_size = 1000
    logits = mx.zeros((1, 1, vocab_size))
    logits = logits.at[0, 0, 0].add(100.0)  # One huge logit
    entropy = compute_token_entropy(logits)
    mx.eval(entropy)
    assert entropy[0].item() < 1.0, (
        f"Peaked entropy should be low, got {entropy[0].item()}"
    )


def test_entropy_shape():
    """Entropy should have shape (seq_len,)."""
    logits = mx.zeros((1, 5, 100))
    entropy = compute_token_entropy(logits)
    mx.eval(entropy)
    assert entropy.shape == (5,), f"Expected (5,), got {entropy.shape}"
