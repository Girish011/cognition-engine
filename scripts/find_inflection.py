"""
Find where cognitive degradation begins (TASK-014).

Analyzes data from TASK-013 (full_degradation.json) to identify the turn
where degradation begins. This is the turn number where CHI would start
triggering Level 2 intervention.

Usage (from project root):
  python scripts/find_inflection.py
  python scripts/find_inflection.py --data data/full_degradation.json
"""

import argparse
import json


def find_inflection(data_path: str = "data/full_degradation.json"):
    with open(data_path) as f:
        results = json.load(f)

    entropy = [r["response_entropy_mean"] for r in results]

    # Method 1: Moving average crossover
    # If short-term average exceeds long-term average, degradation has begun
    window_short = 3
    window_long = 7

    if len(entropy) < window_long:
        print(f"Not enough data points ({len(entropy)}). Need at least {window_long} turns.")
        print(f"Run a longer experiment: python scripts/full_degradation_experiment.py")
        return None

    short_ma = [
        sum(entropy[max(0, i - window_short) : i + 1]) / min(i + 1, window_short)
        for i in range(len(entropy))
    ]
    long_ma = [
        sum(entropy[max(0, i - window_long) : i + 1]) / min(i + 1, window_long)
        for i in range(len(entropy))
    ]

    # Find first crossover after initial stabilization
    crossover_turn = None
    for i in range(window_long, len(entropy)):
        if short_ma[i] > long_ma[i] * 1.05:  # 5% threshold
            crossover_turn = i
            break

    # Method 2: Maximum rate of change (turn where entropy jump is largest)
    if len(entropy) >= 3:
        diffs = [entropy[i + 1] - entropy[i] for i in range(len(entropy) - 1)]
        max_increase_idx = max(range(len(diffs)), key=lambda i: diffs[i])
        # The big increase happens after this turn (between max_increase_idx and max_increase_idx+1)
        max_increase_turn = max_increase_idx
    else:
        max_increase_turn = None

    print(f"Total turns analyzed: {len(entropy)}")
    print(f"Entropy range: {min(entropy):.4f} — {max(entropy):.4f}")
    print(f"Moving average crossover at turn: {crossover_turn}")
    print(f"Maximum entropy increase at turn: {max_increase_turn} (largest jump into turn {max_increase_turn + 1})")

    if crossover_turn is not None:
        print(f"\n→ Degradation likely begins around turn {crossover_turn}")
        print(f"  This is where CHI would start triggering Level 2 intervention")
        return crossover_turn
    else:
        print(f"\n→ No clear inflection found. Model may not have degraded")
        print(f"  within the number of turns tested. Try longer conversations.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find inflection point from full_degradation.json")
    parser.add_argument(
        "--data",
        default="data/full_degradation.json",
        help="Path to full_degradation.json",
    )
    args = parser.parse_args()
    find_inflection(data_path=args.data)
