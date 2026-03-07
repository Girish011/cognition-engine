"""
TASK-006: Verify hidden states are meaningful.
Run two very different prompts, capture hidden states at the same layer,
average over sequence length, compute cosine similarity.
Expectation: similarity should be < 1 (different prompts -> different representations).
Run from repo root: python scripts/verify_states.py
"""
import sys
from pathlib import Path

# Ensure repo root is on path for engine import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
from mlx_lm import load
from engine.hidden_states import extract_hidden_states


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Cosine similarity between two 1D vectors: (a·b) / (||a|| ||b||)."""
    dot = float(mx.sum(a * b))
    na = float(mx.sqrt(mx.sum(a * a)))
    nb = float(mx.sqrt(mx.sum(b * b)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def main():
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

    layer_idx = 16
    max_tokens = 80

    prompt_a = "Write a Python sorting algorithm"
    prompt_b = "Write a poem about the ocean"

    print("Prompt A:", prompt_a)
    print("Prompt B:", prompt_b)
    print(f"Layer index: {layer_idx}, max_tokens: {max_tokens}\n")

    print("Running prompt A...")
    text_a, states_a = extract_hidden_states(model, tokenizer, prompt_a, layer_idx=layer_idx, max_tokens=max_tokens)
    print("Running prompt B...")
    text_b, states_b = extract_hidden_states(model, tokenizer, prompt_b, layer_idx=layer_idx, max_tokens=max_tokens)

    if states_a is None or states_b is None:
        print("ERROR: Failed to capture hidden states from one or both prompts.")
        sys.exit(1)

    # Average over sequence length: (1, seq_len, hidden_dim) -> (hidden_dim,)
    vec_a = mx.mean(states_a, axis=1).squeeze(0)
    vec_b = mx.mean(states_b, axis=1).squeeze(0)
    mx.eval(vec_a, vec_b)

    sim = cosine_similarity(vec_a, vec_b)
    print(f"\nHidden state shapes: A {states_a.shape}, B {states_b.shape}")
    print(f"Averaged vector shape: {vec_a.shape}")
    print(f"Cosine similarity (averaged over seq): {sim:.4f}")
    print("\n(Similarity < 1 indicates different prompts produced different representations.)")


if __name__ == "__main__":
    main()
