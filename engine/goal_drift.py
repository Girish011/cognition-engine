"""
Goal drift detection (TASK-011) — third CHI signal.

Measures how far the model's current output has drifted from the original task
via cosine similarity of hidden states (lower = more drift).
"""

import sys
from pathlib import Path

# Ensure project root is on path when running as script (e.g. python engine/goal_drift.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from engine.hidden_states import extract_hidden_states, extract_hidden_states_forward_only


def compute_goal_drift(
    model,
    tokenizer,
    original_task: str,
    current_output: str,
    layer_idx: int = 27,
    use_forward_only: bool = True,
) -> float:
    """
    Measure how far the model's current output has drifted from the original task.

    Returns: cosine similarity in [-1, 1]. Higher = more on-task; lower = more drift.

    Uses hidden states from layer_idx (default 27 = last layer for better semantic
    discrimination), averaged over sequence length. Numerically stable for large norms.

    When use_forward_only=True (default): single forward pass on each text, no generation.
    This avoids degenerate similarity when current_output is long (e.g. 300 tokens) where
    "prompt + 1 generated token" would average over a very long sequence and distort the
    comparison. Both task and response are compared as plain text representations.
    """
    if use_forward_only:
        task_states = extract_hidden_states_forward_only(
            model, tokenizer, original_task, layer_idx=layer_idx
        )
        output_states = extract_hidden_states_forward_only(
            model, tokenizer, current_output, layer_idx=layer_idx
        )
    else:
        _, task_states = extract_hidden_states(
            model, tokenizer, original_task, layer_idx=layer_idx, max_tokens=1
        )
        _, output_states = extract_hidden_states(
            model, tokenizer, current_output, layer_idx=layer_idx, max_tokens=1
        )

    if task_states is None or output_states is None:
        raise ValueError("Hidden state extraction returned None")

    # Task: mean over sequence (short text). Output: for long sequences use last-token
    # hidden state only; averaging over 300+ positions can produce degenerate (e.g. ~0)
    # similarity due to scale/numerics. Last token encodes the full sequence in causal LMs.
    task_vec = mx.mean(task_states, axis=(0, 1))
    seq_len = output_states.shape[1]
    max_mean_positions = 128
    if seq_len > max_mean_positions:
        output_vec = output_states[0, -1, :]  # (hidden_dim,)
    else:
        output_vec = mx.mean(output_states, axis=(0, 1))

    # Numerically stable cosine: scale to avoid overflow, then unit vectors
    scale_a = mx.max(mx.abs(task_vec)) + 1e-12
    scale_b = mx.max(mx.abs(output_vec)) + 1e-12
    a = task_vec / scale_a
    b = output_vec / scale_b
    na = mx.sqrt(mx.sum(a**2) + 1e-12)
    nb = mx.sqrt(mx.sum(b**2) + 1e-12)
    a_unit = a / na
    b_unit = b / nb
    cos_sim = mx.sum(a_unit * b_unit)

    mx.eval(cos_sim)
    out = float(cos_sim.item())
    if out != out:  # NaN
        return 0.0
    return max(-1.0, min(1.0, out))


if __name__ == "__main__":
    from mlx_lm import load

    model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

    task = "Build a Python TaskManager class"
    on_task = "class TaskManager:\n    def __init__(self):\n        self.tasks = []"
    off_task = "The weather in Paris is beautiful this time of year."

    score_on = compute_goal_drift(model, tokenizer, task, on_task)
    score_off = compute_goal_drift(model, tokenizer, task, off_task)

    print(f"On-task similarity:  {score_on:.4f}")   # Should be high (0.7+)
    print(f"Off-task similarity: {score_off:.4f}")  # Should be lower
    assert score_on > score_off, (
        f"On-task ({score_on:.4f}) should be > off-task ({score_off:.4f})"
    )
    print("Assert passed: on-task > off-task.")
