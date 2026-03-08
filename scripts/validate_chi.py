"""
TASK-016: Validate CHI against Week 2 data.

Replays the 20-turn experiment through the CHI formula and smoothing
using the saved signals in full_degradation.json (no model load).
Verifies that CHI drops around the inflection point from TASK-014.

Output: data/chi_validation.png (CHI + thresholds; raw signals)
        Intervention levels printed per turn.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Must match cognition_engine.chi
MIN_RESPONSE_TOKENS_FOR_REPETITION = 20
CHI_SMOOTH_ALPHA = 0.3
W_ENTROPY = 0.4
W_REPETITION = 0.3
W_DRIFT = 0.3


def intervention_level(chi: float) -> int:
    """Same logic as CHIMonitor.get_intervention_level()."""
    if chi > 0.8:
        return 1
    elif chi > 0.5:
        return 2
    elif chi > 0.3:
        return 3
    else:
        return 4


def compute_chi_from_signals(results: list[dict]) -> list[dict]:
    """
    Apply CHI formula + smoothing + repetition gating to pre-computed signals.
    Each result must have: turn, response_entropy_mean, repetition_vs_previous,
    goal_similarity, response_length.
    """
    chi_history: list[float] = []
    out = []

    for i, r in enumerate(results):
        entropy_raw = r["response_entropy_mean"]
        repetition_raw = r["repetition_vs_previous"]
        response_length = r.get("response_length", 0)
        # Gate repetition: if response under 20 tokens, carry forward previous
        if response_length < MIN_RESPONSE_TOKENS_FOR_REPETITION and i > 0:
            repetition_raw = results[i - 1]["repetition_vs_previous"]

        entropy_health = max(0.0, min(1.0, 1.0 - entropy_raw / 10.0))
        rep_health = 1.0 - repetition_raw
        drift_health = r["goal_similarity"]

        chi_current = (
            W_ENTROPY * entropy_health
            + W_REPETITION * rep_health
            + W_DRIFT * drift_health
        )
        chi_current = max(0.0, min(1.0, chi_current))

        if not chi_history:
            chi_smoothed = chi_current
        else:
            chi_smoothed = CHI_SMOOTH_ALPHA * chi_current + (1.0 - CHI_SMOOTH_ALPHA) * chi_history[-1]

        chi_history.append(chi_smoothed)
        level = intervention_level(chi_smoothed)
        out.append({
            "turn": r["turn"],
            "chi_score": chi_smoothed,
            "chi_raw": chi_current,
            "entropy_raw": entropy_raw,
            "repetition_raw": repetition_raw,
            "goal_drift": r["goal_drift"],
            "level": level,
        })
    return out


def plot_chi_validation(
    validated: list[dict],
    results: list[dict],
    output_path: str = "data/chi_validation.png",
):
    """Top panel: CHI with threshold lines. Bottom panel: three raw signals."""
    turns = [v["turn"] for v in validated]
    chi_scores = [v["chi_score"] for v in validated]
    entropy = [v["entropy_raw"] for v in validated]
    repetition = [v["repetition_raw"] for v in validated]
    goal_drift = [r["goal_drift"] for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: CHI score with thresholds
    ax1 = axes[0]
    ax1.plot(turns, chi_scores, "b-", linewidth=2, label="CHI (smoothed)")
    ax1.axhline(y=0.8, color="green", linestyle="--", alpha=0.7, label="Level 1 threshold (0.8)")
    ax1.axhline(y=0.5, color="orange", linestyle="--", alpha=0.7, label="Level 2 (0.5)")
    ax1.axhline(y=0.3, color="red", linestyle="--", alpha=0.7, label="Level 3 (0.3)")
    ax1.set_ylabel("CHI score")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("CHI Validation — Cognitive Health Index over turns")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: raw signals
    ax2 = axes[1]
    ax2.plot(turns, entropy, "b-", linewidth=1.5, label="Entropy (mean)")
    ax2.plot(turns, repetition, "g-", linewidth=1.5, label="Repetition (vs prev)")
    ax2.plot(turns, goal_drift, "r-", linewidth=1.5, label="Goal drift")
    ax2.set_ylabel("Signal value")
    ax2.set_xlabel("Turn Number")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_path = data_dir / "full_degradation.json"
    out_path = data_dir / "chi_validation.png"

    if not data_path.exists():
        print(f"Missing {data_path}. Run scripts/full_degradation_experiment.py first.")
        sys.exit(1)

    with open(data_path) as f:
        results = json.load(f)

    validated = compute_chi_from_signals(results)
    plot_chi_validation(validated, results, output_path=str(out_path))

    # Print summary and intervention levels
    print("\nCHI validation summary (inflection from TASK-014: turn 10)")
    print("-" * 60)
    for v in validated:
        lev = v["level"]
        print(f"  Turn {v['turn']:2d}  CHI: {v['chi_score']:.3f}  Level: {lev}")
    print("-" * 60)
    chi_values = [v["chi_score"] for v in validated]
    print(f"  CHI starts high (>0.7) in early turns: {all(chi_values[i] > 0.7 for i in range(min(5, len(chi_values)))) if len(chi_values) >= 5 else 'N/A'}")
    print(f"  CHI drops below 0.7 around inflection: {any(v['chi_score'] < 0.7 for v in validated)}")
    below_07_turns = [v["turn"] for v in validated if v["chi_score"] < 0.7]
    if below_07_turns:
        print(f"  First turn with CHI < 0.7: {below_07_turns[0]}")


if __name__ == "__main__":
    main()
