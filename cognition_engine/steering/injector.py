"""
Steering vector injection during inference.

Level 2 intervention: wrap target transformer layers and add a
precomputed steering vector to layer outputs.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def compute_adaptive_alpha(chi_score: float, base_alpha: float = 0.5) -> float:
    """
    Lower CHI = stronger steering. Piecewise linear (P-only controller).

    TASK-047E: recalibrated to match new CHI range (0.82-0.95 healthy).
    CHI >= 0.90 -> alpha = 0.0 (healthy, no steering needed)
    CHI >= 0.30 -> linear ramp from 0.0 (CHI=0.90) to 1.5*base (CHI=0.30)
    CHI < 0.30 -> alpha = 1.5*base (capped; escalate to Level 3/4)
    """
    if chi_score >= 0.90:
        return 0.0
    if chi_score >= 0.30:
        t = (0.90 - chi_score) / 0.60  # 0.0 at CHI=0.90, 1.0 at CHI=0.30
        return float(base_alpha) * (0.6 + t * 0.9)
    return float(base_alpha) * 1.5


class SteeringWrapper(nn.Module):
    """
    Wraps a transformer layer and injects a steering vector.

    During forward pass:
    1. Run original layer
    2. Add alpha * steering_vector to hidden states
    3. Return steered output
    """

    def __init__(self, layer: nn.Module, steering_vector: np.ndarray, alpha: float = 1.0):
        super().__init__()
        self.layer = layer
        self.steering_vector = mx.array(steering_vector)  # (hidden_dim,)
        self.alpha = float(alpha)
        self.active = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.layer(*args, **kwargs)

        if not self.active or self.alpha <= 0:
            return out

        delta = self.alpha * self.steering_vector

        if isinstance(out, tuple):
            hidden = out[0]
            hidden = hidden + delta  # broadcast over (batch, seq, hidden)
            return (hidden,) + out[1:]

        return out + delta


class SteeringEngine:
    """Manage layer wrapping, activation, and cleanup for steering."""

    def __init__(self, model: Any):
        self.model = model
        self.active_wrappers: dict[int, SteeringWrapper] = {}
        self.originals: dict[int, nn.Module] = {}

    def load_vectors(
        self,
        name: str,
        layer_idx: int | None = None,
        alpha: float = 1.0,
        *,
        normalize: bool = False,
        vectors_dir: str = "data/steering_vectors",
    ) -> int:
        """
        Load precomputed vectors and wrap target layer.

        Returns:
            The selected layer index.
        """
        from cognition_engine.steering.compute_vectors import load_steering_vectors

        vectors = load_steering_vectors(name, vectors_dir)

        if vectors.ndim != 2:
            raise ValueError("Loaded steering vectors must have shape (num_layers, hidden_dim).")

        if layer_idx is None:
            norms = np.linalg.norm(vectors, axis=1)
            layer_idx = int(np.argmax(norms))
            print(f"Auto-selected layer {layer_idx} (norm={norms[layer_idx]:.4f})")

        if layer_idx < 0 or layer_idx >= len(self.model.model.layers):
            raise IndexError(f"Invalid layer_idx={layer_idx} for model with {len(self.model.model.layers)} layers.")

        vector = vectors[layer_idx]
        if normalize:
            n = float(np.linalg.norm(vector))
            if n <= 0:
                raise ValueError(f"Cannot normalize zero-norm steering vector at layer {layer_idx}.")
            vector = vector / n
        self._wrap_layer(layer_idx, vector, alpha)
        norm_tag = " (normalized)" if normalize else ""
        print(f"Loaded '{name}' steering at layer {layer_idx}, alpha={alpha}{norm_tag}")
        return layer_idx

    def _wrap_layer(self, layer_idx: int, vector: np.ndarray, alpha: float) -> None:
        if layer_idx in self.active_wrappers:
            self._unwrap_layer(layer_idx)

        original = self.model.model.layers[layer_idx]
        wrapper = SteeringWrapper(original, vector, alpha)
        self.model.model.layers[layer_idx] = wrapper
        self.originals[layer_idx] = original
        self.active_wrappers[layer_idx] = wrapper

    def _unwrap_layer(self, layer_idx: int) -> None:
        if layer_idx not in self.originals:
            return
        self.model.model.layers[layer_idx] = self.originals[layer_idx]
        del self.originals[layer_idx]
        del self.active_wrappers[layer_idx]

    def activate(self) -> None:
        for wrapper in self.active_wrappers.values():
            wrapper.active = True

    def deactivate(self) -> None:
        for wrapper in self.active_wrappers.values():
            wrapper.active = False

    def set_alpha(self, alpha: float) -> None:
        for wrapper in self.active_wrappers.values():
            wrapper.alpha = float(alpha)

    def cleanup(self) -> None:
        for layer_idx in list(self.originals.keys()):
            self._unwrap_layer(layer_idx)

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            # Avoid raising from destructor during interpreter shutdown.
            pass


if __name__ == "__main__":
    from mlx_lm import generate, load

    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    engine = SteeringEngine(model)

    prompt = "Write a calculator class in Python"

    print("=== WITHOUT STEERING ===")
    response_before = generate(model, tokenizer, prompt=prompt, max_tokens=200)
    print(response_before[:300])

    selected = engine.load_vectors("task_focus", alpha=2.0)
    engine.activate()
    print(f"\n=== WITH STEERING (task_focus, layer={selected}, alpha=2.0) ===")
    response_after = generate(model, tokenizer, prompt=prompt, max_tokens=200)
    print(response_after[:300])

    engine.cleanup()
    print(f"\nResponses identical: {response_before == response_after}")

