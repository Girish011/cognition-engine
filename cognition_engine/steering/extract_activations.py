"""
Extract hidden states from contrastive pairs across all transformer layers.

For each pair, we run both positive and negative through the model,
capture the hidden state at every layer, and store them for steering
vector computation.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import numpy as np

from cognition_engine.steering.contrastive_pairs import format_pair_for_model, get_all_pairs


class CaptureWrapper:
    """Wrap a layer and capture its output on each forward pass."""

    def __init__(self, layer: Any):
        self.layer = layer
        self.captured: list[Any] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.layer(*args, **kwargs)
        self.captured.append(out)
        return out


def extract_all_layer_states(
    model: Any,
    tokenizer: Any,
    text: str,
    num_layers: int | None = None,
) -> np.ndarray:
    """
    Run text through model and capture last-token hidden state at every layer.

    Returns:
        np.ndarray with shape (num_layers, hidden_dim)
    """
    if num_layers is None:
        num_layers = len(model.model.layers)

    tokens = mx.array(tokenizer.encode(text))[None]  # (1, seq_len)

    originals: list[Any] = []
    wrappers: list[CaptureWrapper] = []

    for i in range(num_layers):
        original = model.model.layers[i]
        wrapper = CaptureWrapper(original)
        model.model.layers[i] = wrapper
        originals.append(original)
        wrappers.append(wrapper)

    try:
        output = model(tokens, cache=None)
        mx.eval(output)

        states: list[np.ndarray] = []
        for wrapper in wrappers:
            if not wrapper.captured:
                raise RuntimeError("Layer output was not captured during forward pass.")

            h = wrapper.captured[0]
            if isinstance(h, tuple):
                h = h[0]

            last_token_state = h[0, -1, :]  # (hidden_dim,)
            mx.eval(last_token_state)
            states.append(np.array(last_token_state))

        return np.stack(states)  # (num_layers, hidden_dim)

    finally:
        for i, original in enumerate(originals):
            model.model.layers[i] = original


def extract_contrastive_activations(
    model: Any,
    tokenizer: Any,
    pairs: list[dict[str, str]],
    pair_name: str,
) -> dict[str, np.ndarray]:
    """
    Extract activations for all pairs in a contrastive set.

    Returns:
        {
          "positive": np.ndarray (num_pairs, num_layers, hidden_dim),
          "negative": np.ndarray (num_pairs, num_layers, hidden_dim),
        }
    """
    num_layers = len(model.model.layers)
    pos_states: list[np.ndarray] = []
    neg_states: list[np.ndarray] = []

    for i, pair in enumerate(pairs):
        pos_text, neg_text = format_pair_for_model(pair, tokenizer)
        print(f"  [{pair_name}] Pair {i + 1}/{len(pairs)}...")

        pos_state = extract_all_layer_states(model, tokenizer, pos_text, num_layers)
        neg_state = extract_all_layer_states(model, tokenizer, neg_text, num_layers)

        pos_states.append(pos_state)
        neg_states.append(neg_state)

    return {
        "positive": np.stack(pos_states),
        "negative": np.stack(neg_states),
    }


if __name__ == "__main__":
    from mlx_lm import load

    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    pairs = get_all_pairs()

    print("Extracting activations for task_focus pairs...")
    result = extract_contrastive_activations(
        model,
        tokenizer,
        pairs["task_focus"],
        "task_focus",
    )

    print(f"Positive shape: {result['positive'].shape}")
    print(f"Negative shape: {result['negative'].shape}")

    os.makedirs("data", exist_ok=True)
    np.savez("data/activations_task_focus.npz", **result)
    print("Saved to data/activations_task_focus.npz")

