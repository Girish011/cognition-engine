"""
Full 3-signal degradation experiment (TASK-013) — proto-CHI.

Runs the same multi-turn coding conversation but tracks ALL three signals:
- Entropy per turn (TASK-008)
- Repetition between consecutive responses (TASK-010)
- Goal drift from original task (TASK-011)

Saves data to data/full_degradation.json and generates data/full_degradation_graph.png
with a combined 3-panel graph.

Usage (from project root):
  python scripts/full_degradation_experiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import mlx.core as mx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cognition_engine.entropy import compute_token_entropy, get_logits_for_sequence
from cognition_engine.repetition import compute_repetition_score
from cognition_engine.goal_drift import compute_goal_drift
from mlx_lm import load, generate

from scripts.multiturn_entropy import SYSTEM_PROMPT, TASK_SEQUENCE


def run_full_degradation_experiment(
    model,
    tokenizer,
    turns=None,
    max_tokens: int = 300,
    context_limit: int = 28000,
    debug: bool = False,
):
    """
    Run multi-turn conversation and track entropy, repetition, and goal drift.

    Returns:
        list of dicts with turn, entropy stats, repetition (vs previous), goal_drift, etc.
    """
    if turns is None:
        turns = TASK_SEQUENCE

    original_task = turns[0]  # First user message = "original task" for drift
    conversation = SYSTEM_PROMPT + "\n"
    responses: list[str] = []
    results = []

    for i, user_msg in enumerate(turns):
        conversation += f"\nUser: {user_msg}\nAssistant: "

        # Generate response
        response = generate(model, tokenizer, prompt=conversation, max_tokens=max_tokens)
        responses.append(response)
        full_text = conversation + response

        # --- 1. Entropy (TASK-008) ---
        logits = get_logits_for_sequence(model, tokenizer, full_text)
        entropy = compute_token_entropy(logits)
        mx.eval(entropy)
        response_tokens = len(tokenizer.encode(response))
        response_entropy = (
            entropy[-response_tokens:] if response_tokens > 0 else entropy
        )
        mean_ent = mx.mean(response_entropy)
        mx.eval(mean_ent)

        # --- 2. Repetition vs previous response (TASK-010) ---
        if i == 0:
            repetition_score = 0.0  # No previous response
        else:
            repetition_score = compute_repetition_score(responses[i - 1], response)

        # --- 3. Goal drift from original task (TASK-011) ---
        # Pass ONLY current turn's response (not full conversation). Uses forward-only
        # hidden state so long response text doesn't produce degenerate similarity.
        try:
            similarity = compute_goal_drift(
                model, tokenizer, original_task, response, layer_idx=27, use_forward_only=True
            )
            # Store as "drift" so that higher = worse (align with entropy & repetition)
            goal_drift = 1.0 - similarity
            if debug:
                print(f"  [Turn {i}] original_task = '{original_task[:80]}...'")
                print(f"  [Turn {i}] current_response = '{response[:80]}...'")
                print(f"  [Turn {i}] goal_drift raw cosine (similarity) = {similarity:.4f}")
        except Exception as e:
            goal_drift = float("nan")
            similarity = float("nan")
            print(f"  [Turn {i}] Goal drift error: {e}")

        context_length = len(tokenizer.encode(full_text))
        result = {
            "turn": i,
            "prompt": user_msg[:80],
            "response_length": response_tokens,
            "context_length": context_length,
            "response_entropy_mean": mean_ent.item(),
            "repetition_vs_previous": repetition_score,
            "goal_drift": goal_drift,
            "goal_similarity": similarity,
        }
        results.append(result)

        print(
            f"Turn {i:3d} | ctx {context_length:5d} | "
            f"entropy={result['response_entropy_mean']:.4f} | "
            f"repetition={repetition_score:.4f} | "
            f"drift={result['goal_drift']:.4f}"
        )

        conversation += response + "\n"

        if context_length > context_limit:
            print(f"Stopping at turn {i}: approaching context limit")
            break

    return results


def plot_full_degradation(
    data_path: str = "data/full_degradation.json",
    output_path: str = "data/full_degradation_graph.png",
):
    """Generate 3-panel graph: entropy, repetition, goal drift over turns."""
    with open(data_path) as f:
        results = json.load(f)

    turns = [r["turn"] for r in results]
    entropy_mean = [r["response_entropy_mean"] for r in results]
    repetition = [r["repetition_vs_previous"] for r in results]
    goal_drift = [r["goal_drift"] for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    # Panel 1: Entropy
    ax1 = axes[0]
    ax1.plot(turns, entropy_mean, "b-", linewidth=2, label="Response entropy (mean)")
    ax1.set_ylabel("Entropy (bits)")
    ax1.set_title("Full 3-Signal Degradation — Entropy, Repetition, Goal Drift")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95, "↑ worse: more confused/random", transform=ax1.transAxes,
             fontsize=8, verticalalignment="top", color="gray")

    # Panel 2: Repetition (consecutive responses)
    ax2 = axes[1]
    ax2.plot(turns, repetition, "g-", linewidth=2, label="Repetition vs previous")
    ax2.set_ylabel("Jaccard (repetition)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.95, "↑ worse: more copy-paste vs last reply", transform=ax2.transAxes,
             fontsize=8, verticalalignment="top", color="gray")

    # Panel 3: Goal drift (1 - similarity; higher = more drift)
    ax3 = axes[2]
    ax3.plot(turns, goal_drift, "r-", linewidth=2, label="Goal drift (1 − similarity)")
    ax3.set_ylabel("Goal drift")
    ax3.set_xlabel("Turn Number")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.95, "↑ worse: further from original task", transform=ax3.transAxes,
             fontsize=8, verticalalignment="top", color="gray")

    # How-to-read box at bottom (visible when you come back to the graph)
    howto = (
        "How to read: All three panels share the same x-axis (turn number). "
        "In each panel, a line going UP = degradation (worse). "
        "When entropy, repetition, or goal drift rise together → that’s where CHI would intervene. "
        "See data/FULL_DEGRADATION_GRAPH_README.md for details."
    )
    fig.text(0.5, 0.01, howto, ha="center", fontsize=8, wrap=True,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full 3-signal degradation experiment")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Max turns to run (default: all). Use e.g. 3 for a quick smoke test.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate the graph from existing data/full_degradation.json.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print goal-drift debug info (original_task, response snippet, raw cosine) each turn.",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    out_json = data_dir / "full_degradation.json"
    out_png = data_dir / "full_degradation_graph.png"

    if args.plot_only:
        plot_full_degradation(data_path=str(out_json), output_path=str(out_png))
    else:
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
        turns = TASK_SEQUENCE[: args.max_turns] if args.max_turns else None
        results = run_full_degradation_experiment(
            model, tokenizer, turns=turns, debug=args.debug
        )
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {out_json}")
        plot_full_degradation(data_path=str(out_json), output_path=str(out_png))
        print(f"Completed {len(results)} turns.")
