"""
First steering experiment (TASK-038).

Protocol:
1. Run 10-turn conversation WITHOUT steering → measure CHI per turn
2. Run SAME 10-turn conversation WITH steering → measure CHI per turn
3. Compare: does steering keep CHI higher for longer?

Success criteria (reported in summary; manual review for qualitative focus):
- Steered run has higher average CHI after turn 5
- Steered run crosses 0.7 threshold LATER than unsteered

Saves:
- data/steering_experiment.json (or paths from CLI)
- data/steering_experiment.png

Usage (from project root):
  python scripts/steering_experiment.py
  python scripts/steering_experiment.py --vector-name task_focus --alpha 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SYSTEM_PROMPT = "You are a coding assistant. Follow instructions precisely."

EXPERIMENT_TURNS = [
    "Create a Calculator class with add and subtract methods",
    "Add multiply and divide methods. Handle division by zero.",
    "Write unit tests for all four operations",
    "Add a history feature that stores the last 10 operations",
    "Refactor to use a Strategy pattern for the operations",
    "Add logging to track which operations are called most",
    "Write a CLI wrapper so the calculator can be used from terminal",
    "Add support for chaining operations like calc.add(1).multiply(3)",
    "Write integration tests for the CLI",
    "Add a square root method and update all tests",
]

CHI_THRESHOLD = 0.7


def run_condition(
    model,
    tokenizer,
    turns: list[str],
    *,
    steering_engine: object | None,
    max_tokens: int,
    context_limit: int,
    label: str,
) -> list[dict]:
    """
    Run multi-turn chat; collect per-turn CHI via CHIMonitor.

    When steering_engine is set, it is active during generate() and
    deactivated during CHI update so logits/drift use the same unsteered
    forward pass for both conditions (fair comparison of response quality).
    """
    from cognition_engine.chi import CHIMonitor
    from cognition_engine.inference import generate

    original_task = turns[0]
    monitor = CHIMonitor(model, tokenizer, original_task=original_task)
    conversation = SYSTEM_PROMPT + "\n"
    rows: list[dict] = []

    for i, user_msg in enumerate(turns):
        conversation += f"\nUser: {user_msg}\nAssistant: "

        if steering_engine is not None:
            steering_engine.activate()

        response = generate(model, tokenizer, prompt=conversation, max_tokens=max_tokens)

        if steering_engine is not None:
            steering_engine.deactivate()

        full_text = conversation + response
        chi_out = monitor.update(current_response=response, full_context=full_text)

        context_length = len(tokenizer.encode(full_text))
        rows.append(
            {
                "turn": i,
                "user_message": user_msg,
                "chi_score": chi_out["chi_score"],
                "chi_raw": chi_out["chi_raw"],
                "condition": label,
                "response_preview": response[:500],
                "response_length_tokens": len(tokenizer.encode(response)),
                "context_length_tokens": context_length,
            }
        )

        print(
            f"[{label}] Turn {i:2d} | CHI={chi_out['chi_score']:.4f} | "
            f"ctx={context_length:5d} tok"
        )

        conversation += response + "\n"

        if context_length > context_limit:
            print(f"[{label}] Stopping at turn {i}: approaching context limit")
            break

    return rows


def first_turn_below(chi_values: list[float], threshold: float) -> int | None:
    """First turn index where CHI < threshold; None if never."""
    for i, c in enumerate(chi_values):
        if c < threshold:
            return i
    return None


def summarize(unsteered: list[dict], steered: list[dict]) -> dict:
    u_chi = [r["chi_score"] for r in unsteered]
    s_chi = [r["chi_score"] for r in steered]

    # "After turn 5" → turns with index >= 5 (0-based)
    after_5_u = u_chi[5:] if len(u_chi) > 5 else []
    after_5_s = s_chi[5:] if len(s_chi) > 5 else []

    mean_after_5_u = sum(after_5_u) / len(after_5_u) if after_5_u else None
    mean_after_5_s = sum(after_5_s) / len(after_5_s) if after_5_s else None

    cross_u = first_turn_below(u_chi, CHI_THRESHOLD)
    cross_s = first_turn_below(s_chi, CHI_THRESHOLD)

    higher_after_5 = (
        mean_after_5_s is not None
        and mean_after_5_u is not None
        and mean_after_5_s > mean_after_5_u
    )
    # Steered crosses 0.7 at a strictly later turn index (or never), vs unsteered.
    if cross_s is None:
        steered_crosses_later = True
    elif cross_u is None:
        steered_crosses_later = False
    else:
        steered_crosses_later = cross_s > cross_u

    return {
        "chi_threshold": CHI_THRESHOLD,
        "mean_chi_after_turn_5": {
            "unsteered": mean_after_5_u,
            "steered": mean_after_5_s,
        },
        "first_turn_chi_below_threshold": {
            "unsteered": cross_u,
            "steered": cross_s,
        },
        "success_criteria": {
            "steered_higher_avg_chi_after_turn_5": higher_after_5,
            "steered_crosses_threshold_later": steered_crosses_later,
        },
    }


def plot_experiment(
    unsteered: list[dict],
    steered: list[dict],
    output_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    turns_u = [r["turn"] for r in unsteered]
    turns_s = [r["turn"] for r in steered]
    chi_u = [r["chi_score"] for r in unsteered]
    chi_s = [r["chi_score"] for r in steered]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(turns_u, chi_u, "b-o", linewidth=2, markersize=6, label="Unsteered")
    ax.plot(turns_s, chi_s, "r-s", linewidth=2, markersize=6, label="Steered")
    ax.axhline(CHI_THRESHOLD, color="gray", linestyle="--", linewidth=1.5, label=f"CHI = {CHI_THRESHOLD}")
    ax.set_xlabel("Turn")
    ax.set_ylabel("CHI (smoothed)")
    ax.set_title("Steering experiment — CHI vs turn")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Steered vs unsteered CHI experiment")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        help="MLX model id or path",
    )
    parser.add_argument("--vector-name", default="task_focus", help="Steering vectors name (data/steering_vectors/<name>.npz)")
    parser.add_argument("--alpha", type=float, default=2.0, help="Steering strength")
    parser.add_argument("--layer", type=int, default=None, help="Layer index (default: auto by norm)")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--context-limit", type=int, default=28000)
    parser.add_argument("--output-json", default="data/steering_experiment.json")
    parser.add_argument("--output-png", default="data/steering_experiment.png")
    args = parser.parse_args()

    vec_path = root / "data" / "steering_vectors" / f"{args.vector_name}.npz"
    if not vec_path.is_file():
        print(
            f"ERROR: Steering vectors not found: {vec_path}\n"
            "Run TASK-035 (activations) and TASK-036 (compute_vectors) first, "
            "or pass --vector-name pointing to an existing .npz base name.",
            file=sys.stderr,
        )
        sys.exit(1)

    from cognition_engine.inference import load
    from cognition_engine.steering.injector import SteeringEngine

    model, tokenizer = load(args.model)

    print("=== Condition 1: WITHOUT steering ===")
    unsteered = run_condition(
        model,
        tokenizer,
        EXPERIMENT_TURNS,
        steering_engine=None,
        max_tokens=args.max_tokens,
        context_limit=args.context_limit,
        label="unsteered",
    )

    engine = SteeringEngine(model)
    print("\n=== Condition 2: WITH steering ===")
    try:
        engine.load_vectors(
            args.vector_name,
            layer_idx=args.layer,
            alpha=args.alpha,
        )
        steered = run_condition(
            model,
            tokenizer,
            EXPERIMENT_TURNS,
            steering_engine=engine,
            max_tokens=args.max_tokens,
            context_limit=args.context_limit,
            label="steered",
        )
    finally:
        engine.cleanup()

    summary = summarize(unsteered, steered)
    payload = {
        "protocol": "TASK-038",
        "model": args.model,
        "vector_name": args.vector_name,
        "alpha": args.alpha,
        "layer": args.layer,
        "unsteered": unsteered,
        "steered": steered,
        "summary": summary,
    }

    out_json = Path(args.output_json)
    if not out_json.is_absolute():
        out_json = root / out_json
    out_png = Path(args.output_png)
    if not out_png.is_absolute():
        out_png = root / out_png
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved JSON: {out_json}")

    plot_experiment(unsteered, steered, str(out_png))

    s = summary
    print("\n--- Summary ---")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
