"""
Compute steering vectors from contrastive activations.

The steering vector for layer L is:
  v_L = mean(positive_activations_L) - mean(negative_activations_L)

This gives us a direction in activation space that points from
"bad behavior" toward "good behavior."
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def compute_steering_vectors(activations: dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute per-layer steering vectors from contrastive activations.

    Args:
        activations: dict with "positive" and "negative" arrays,
            each shaped (num_pairs, num_layers, hidden_dim).

    Returns:
        np.ndarray of shape (num_layers, hidden_dim).
    """
    if "positive" not in activations or "negative" not in activations:
        raise KeyError("activations must contain 'positive' and 'negative' keys.")

    positive = activations["positive"]
    negative = activations["negative"]
    if positive.shape != negative.shape:
        raise ValueError(
            "positive and negative activation arrays must have matching shapes."
        )
    if positive.ndim != 3:
        raise ValueError(
            "activation arrays must have shape (num_pairs, num_layers, hidden_dim)."
        )

    pos_mean = positive.mean(axis=0)  # (num_layers, hidden_dim)
    neg_mean = negative.mean(axis=0)  # (num_layers, hidden_dim)
    return pos_mean - neg_mean


def find_best_layer(steering_vectors: np.ndarray) -> int:
    """
    Find the layer with the highest steering-vector norm.

    Returns:
        Index of the highest-norm layer.
    """
    if steering_vectors.ndim != 2:
        raise ValueError("steering_vectors must have shape (num_layers, hidden_dim).")

    norms = np.linalg.norm(steering_vectors, axis=1)
    best_layer = int(np.argmax(norms))

    for i, norm in enumerate(norms):
        marker = " <-- BEST" if i == best_layer else ""
        middle = " (middle)" if 11 <= i <= 17 else ""
        print(f"  Layer {i:2d}: norm = {norm:.4f}{middle}{marker}")

    middle_start = 11
    middle_end = min(17, len(norms) - 1)
    if middle_start <= middle_end:
        best_middle = int(middle_start + np.argmax(norms[middle_start : middle_end + 1]))
        print(f"\nBest layer by norm: {best_layer}")
        print(f"Best middle layer ({middle_start}-{middle_end}): {best_middle}")
    else:
        print(f"\nBest layer by norm: {best_layer}")
        print("Best middle layer: unavailable (model has fewer than 12 layers)")

    return best_layer


def save_steering_vectors(
    vectors: np.ndarray,
    name: str,
    output_dir: str = "data/steering_vectors",
) -> str:
    """Save steering vectors to disk and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.npz")
    np.savez(path, vectors=vectors)
    print(f"Saved {name} steering vectors to {path}")
    print(f"Shape: {vectors.shape}")
    return path


def load_steering_vectors(
    name: str,
    output_dir: str = "data/steering_vectors",
) -> np.ndarray:
    """Load precomputed steering vectors from disk."""
    path = os.path.join(output_dir, f"{name}.npz")
    data = np.load(path)
    return data["vectors"]


def _load_activation_file(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {"positive": data["positive"], "negative": data["negative"]}


def _print_best_layer_stats(vectors: np.ndarray, best_layer: int) -> None:
    print(f"\nVector at best layer ({best_layer}):")
    print(f"  Norm: {np.linalg.norm(vectors[best_layer]):.4f}")
    print(f"  Mean: {vectors[best_layer].mean():.6f}")
    print(f"  Std:  {vectors[best_layer].std():.6f}")


if __name__ == "__main__":
    for pair_set in ["task_focus", "tool_format", "code_coherence"]:
        path = Path(f"data/activations_{pair_set}.npz")
        if not path.exists():
            print(f"Skipping {pair_set} — run TASK-035 first")
            continue

        print(f"\n{'=' * 60}")
        print(f"Computing steering vectors: {pair_set}")
        print(f"{'=' * 60}")

        activations = _load_activation_file(path)
        vectors = compute_steering_vectors(activations)
        best = find_best_layer(vectors)
        save_steering_vectors(vectors, pair_set)
        _print_best_layer_stats(vectors, best)

