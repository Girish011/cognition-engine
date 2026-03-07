"""
Token-level entropy from logits (TASK-008).

Computes Shannon entropy over the model's next-token probability distribution.
High entropy = uncertain/confused; low entropy = confident.
"""

import mlx.core as mx


def compute_token_entropy(logits: mx.array) -> mx.array:
    """
    Compute Shannon entropy of the next-token distribution.

    Args:
        logits: shape (1, seq_len, vocab_size) — raw model output

    Returns:
        entropy: shape (seq_len,) — entropy per position

    Formula: H = -Σ p(x) * log2(p(x))
    """
    # Step 1: Convert logits to probabilities via softmax
    probs = mx.softmax(logits, axis=-1)

    # Step 2 & 4: p * log2(p), with 0*log(0) = 0 for numerical stability
    log2_p = mx.log2(mx.maximum(probs, 1e-12))
    plog2p = mx.where(probs > 0, probs * log2_p, 0.0)

    # Step 3: Sum across vocab dimension, negate
    entropy = -mx.sum(plog2p, axis=-1)

    # Return (seq_len,) — squeeze batch dim
    return entropy[0]


def get_logits_for_sequence(model, tokenizer, text: str) -> mx.array:
    """
    Run a single forward pass and return raw logits.
    Unlike generate(), this doesn't sample — just returns
    the model's probability distribution at every position.

    Returns:
        logits: shape (1, seq_len, vocab_size)
    """
    tokens = mx.array(tokenizer.encode(text))[None]  # (1, seq_len)
    logits = model(tokens)  # (1, seq_len, vocab_size)
    mx.eval(logits)
    return logits


if __name__ == "__main__":
    from mlx_lm import load

    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

    # High entropy expected (many valid next tokens):
    e1 = compute_token_entropy(get_logits_for_sequence(model, tokenizer, "The"))
    mean1 = mx.mean(e1)
    mx.eval(mean1)
    print(f"'The' entropy: {mean1.item():.4f}")  # Should be high (6-10+)

    # Lower entropy expected (very predictable continuation):
    e2 = compute_token_entropy(
        get_logits_for_sequence(model, tokenizer, "def fibonacci(n):")
    )
    mean2 = mx.mean(e2)
    mx.eval(mean2)
    print(f"'def fibonacci(n):' entropy: {mean2.item():.4f}")  # Should be lower

    print("(Code prompt should have meaningfully lower average entropy than open-ended.)")
