"""
Generate the "degradation graph" — the visual proof that models lose coherence
and that we can detect exactly when it happens.

Usage: run from project root after TASK-009 (data/multiturn_entropy.json).
  python scripts/visualize_degradation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


def plot_degradation(
    data_path: str = "data/multiturn_entropy.json",
    output_path: str = "data/degradation_graph.png",
):
    with open(data_path) as f:
        results = json.load(f)

    turns = [r["turn"] for r in results]
    entropy_mean = [r["response_entropy_mean"] for r in results]
    entropy_max = [r["response_entropy_max"] for r in results]
    context_len = [r["context_length"] for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Top plot: Entropy over turns
    ax1 = axes[0]
    # Shaded regions: stable vs degradation
    n = len(turns)
    stable_end = min(11, n)
    ax1.axvspan(0, stable_end - 0.5, alpha=0.15, color="green", label="_nolegend_")
    ax1.axvspan(stable_end - 0.5, n, alpha=0.15, color="orange", label="_nolegend_")
    ax1.plot(turns, entropy_mean, "b-", linewidth=2, label="Mean Entropy")
    ax1.fill_between(
        turns, entropy_mean, entropy_max, alpha=0.2, color="blue", label="Max Entropy"
    )
    ax1.set_ylabel("Response Entropy (bits)")
    ax1.set_title(
        "Cognitive Degradation Over Multi-Turn Conversation\n"
        "Qwen2.5-Coder-7B on Apple M4 Pro"
    )
    ax1.text(2, ax1.get_ylim()[1] * 0.85, "Stable\n(low entropy)", fontsize=9, color="darkgreen")
    ax1.text(stable_end + 1, ax1.get_ylim()[1] * 0.85, "Degradation\n(entropy rises)", fontsize=9, color="darkorange")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Takeaway text box
    takeaway = (
        "Read this: Low entropy = model confident & on-topic. "
        "High entropy = confused/erratic. Rise after ~turn 10 = degradation."
    )
    fig.text(0.5, 0.02, takeaway, ha="center", fontsize=9, wrap=True,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # Bottom plot: Context length over turns
    ax2 = axes[1]
    ax2.plot(turns, context_len, "r-", linewidth=2)
    ax2.set_ylabel("Context Length (tokens)")
    ax2.set_xlabel("Turn Number")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_degradation()
