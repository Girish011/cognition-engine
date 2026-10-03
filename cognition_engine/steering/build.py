"""Build steering vectors from scratch: contrastive pairs -> activations -> vectors."""

from __future__ import annotations

import os

import numpy as np

from cognition_engine.steering.compute_vectors import (
    compute_steering_vectors,
    find_best_layer,
    save_steering_vectors,
)
from cognition_engine.steering.contrastive_pairs import get_all_pairs

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"


def build_vectors(
    model_id: str = DEFAULT_MODEL,
    names: list[str] | None = None,
    output_dir: str = "data/steering_vectors",
) -> dict[str, int]:
    """Extract activations and save vectors for each pair set. Returns best layer per set."""
    from mlx_lm import load

    from cognition_engine.steering.extract_activations import extract_contrastive_activations

    model, tokenizer = load(model_id)
    pairs = get_all_pairs()
    selected = names or list(pairs)
    unknown = [n for n in selected if n not in pairs]
    if unknown:
        raise ValueError(f"Unknown pair sets: {unknown}. Available: {list(pairs)}")

    os.makedirs("data", exist_ok=True)
    best: dict[str, int] = {}
    for name in selected:
        print(f"[{name}] extracting activations ({len(pairs[name])} pairs)...")
        acts = extract_contrastive_activations(model, tokenizer, pairs[name], name)
        np.savez(f"data/activations_{name}.npz", **acts)
        vectors = compute_steering_vectors(acts)
        path = save_steering_vectors(vectors, name, output_dir=output_dir)
        best[name] = find_best_layer(vectors)
        print(f"[{name}] saved {path} (strongest layer: {best[name]})")
    return best
