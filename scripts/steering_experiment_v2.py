"""
TASK-040 (updated): Calibrated steering experiment across 4 conditions.

Conditions:
- baseline (unsteered)
- task_focus (layer 16, alpha 0.5)
- tool_format (layer 16, alpha 0.5)
- code_coherence (layer 16, alpha 0.5)
- dual_vector (tool_format@16 alpha 0.3 + code_coherence@17 alpha 0.25)

Key requirements:
- 10 turns (same sequence as TASK-038)
- Track per-turn CHI + raw signals + response/context length
- Use raw vectors (no normalization)
- Support one-condition runs and plot-only aggregation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SYSTEM_PROMPT = "You are a coding assistant. Follow instructions precisely."
CHI_THRESHOLD = 0.7

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

CONDITIONS = ["baseline", "task_focus", "tool_format", "code_coherence", "dual_vector"]
DUAL_VECTOR_CONFIG = {
    "tool_format": {"layer": 16, "alpha": 0.3},
    "code_coherence": {"layer": 17, "alpha": 0.25},
}


def _abs_path(root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (root / p)


def run_condition(
    model: Any,
    tokenizer: Any,
    *,
    condition: str,
    steering_engines: list[Any] | None,
    max_tokens: int,
    context_limit: int,
) -> list[dict]:
    from cognition_engine.chi import CHIMonitor
    from cognition_engine.inference import generate

    monitor = CHIMonitor(model, tokenizer, original_task=EXPERIMENT_TURNS[0])
    conversation = SYSTEM_PROMPT + "\n"
    rows: list[dict] = []

    for i, user_msg in enumerate(EXPERIMENT_TURNS):
        print(f"[{condition}] Generating turn {i}...", flush=True)
        conversation += f"\nUser: {user_msg}\nAssistant: "

        if steering_engines:
            for engine in steering_engines:
                engine.activate()

        response = generate(model, tokenizer, prompt=conversation, max_tokens=max_tokens)

        if steering_engines:
            for engine in reversed(steering_engines):
                engine.deactivate()

        full_text = conversation + response
        chi_out = monitor.update(current_response=response, full_context=full_text)

        response_len = len(tokenizer.encode(response))
        context_len = len(tokenizer.encode(full_text))
        rows.append(
            {
                "turn": i,
                "chi_score": chi_out["chi_score"],
                "chi_raw": chi_out["chi_raw"],
                "entropy": chi_out["entropy_raw"],
                "repetition": chi_out["repetition_raw"],
                "goal_drift": chi_out["drift_raw"],
                "response_length": response_len,
                "context_length": context_len,
                "response_preview": response[:100],
            }
        )

        print(
            f"[{condition}] Turn {i:2d} | CHI={chi_out['chi_score']:.4f} | "
            f"resp={response_len:4d} tok | ctx={context_len:5d} tok"
            ,
            flush=True,
        )

        conversation += response + "\n"
        if context_len > context_limit:
            print(f"[{condition}] Stopping at turn {i}: context limit approached")
            break

    return rows


def first_turn_below(values: list[float], threshold: float) -> int | None:
    for i, v in enumerate(values):
        if v < threshold:
            return i
    return None


def mean_after_turn(rows: list[dict], turn_idx: int) -> float | None:
    vals = [r["chi_score"] for r in rows if r["turn"] >= turn_idx]
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize_conditions(payload: dict) -> dict:
    conditions_data: dict[str, list[dict]] = payload.get("conditions", {})
    baseline = conditions_data.get("baseline", [])
    baseline_after_5 = mean_after_turn(baseline, 5)
    baseline_mean_len = (sum(r["response_length"] for r in baseline) / len(baseline)) if baseline else None

    per_condition: dict[str, dict] = {}
    at_least_one_better_after_5 = False
    all_vectors_alive = True

    for name, rows in conditions_data.items():
        chi_vals = [r["chi_score"] for r in rows]
        mean_len = (sum(r["response_length"] for r in rows) / len(rows)) if rows else None
        mean_after_5 = mean_after_turn(rows, 5)

        responses_alive = all(r["response_length"] > 20 for r in rows) and len(rows) == len(EXPERIMENT_TURNS)
        if name != "baseline" and not responses_alive:
            all_vectors_alive = False

        length_parity_ok = None
        if name != "baseline" and baseline_mean_len and mean_len is not None:
            length_parity_ok = 0.5 * baseline_mean_len <= mean_len <= 1.5 * baseline_mean_len

        better_after_5 = None
        if name != "baseline" and mean_after_5 is not None and baseline_after_5 is not None:
            better_after_5 = mean_after_5 > baseline_after_5
            if better_after_5:
                at_least_one_better_after_5 = True

        per_condition[name] = {
            "mean_chi": (sum(chi_vals) / len(chi_vals)) if chi_vals else None,
            "mean_chi_turns_5_to_9": mean_after_5,
            "first_turn_chi_below_0_7": first_turn_below(chi_vals, CHI_THRESHOLD),
            "mean_response_length": mean_len,
            "responses_alive_gt_20_all_turns": responses_alive,
            "length_parity_within_50pct_of_baseline": length_parity_ok,
            "better_than_baseline_after_turn_5": better_after_5,
        }

    best_single_after_5 = None
    single_vector_conditions = [
        n for n in ("task_focus", "tool_format", "code_coherence")
        if per_condition.get(n, {}).get("mean_chi_turns_5_to_9") is not None
    ]
    if single_vector_conditions:
        best_single_after_5 = max(
            per_condition[n]["mean_chi_turns_5_to_9"] for n in single_vector_conditions
        )

    dual_after_5 = per_condition.get("dual_vector", {}).get("mean_chi_turns_5_to_9")
    dual_beats_best_single = None
    if dual_after_5 is not None and best_single_after_5 is not None:
        dual_beats_best_single = dual_after_5 > best_single_after_5

    return {
        "chi_threshold": CHI_THRESHOLD,
        "baseline": {
            "mean_chi_turns_5_to_9": baseline_after_5,
            "mean_response_length": baseline_mean_len,
        },
        "per_condition": per_condition,
        "dual_vector_vs_single": {
            "best_single_vector_mean_chi_turns_5_to_9": best_single_after_5,
            "dual_vector_mean_chi_turns_5_to_9": dual_after_5,
            "dual_beats_best_single_vector": dual_beats_best_single,
        },
        "success_criteria": {
            "responses_alive_all_3_vectors": all_vectors_alive,
            "at_least_one_vector_improves_mean_chi_after_turn_5": at_least_one_better_after_5,
        },
    }


def plot_payload(payload: dict, output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions_data: dict[str, list[dict]] = payload.get("conditions", {})

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    color_map = {
        "baseline": "black",
        "task_focus": "tab:blue",
        "tool_format": "tab:green",
        "code_coherence": "tab:red",
        "dual_vector": "tab:purple",
    }

    for name in CONDITIONS:
        rows = conditions_data.get(name, [])
        if not rows:
            continue
        turns = [r["turn"] for r in rows]
        chi = [r["chi_score"] for r in rows]
        rlen = [r["response_length"] for r in rows]

        mean_chi = sum(chi) / len(chi)
        label = f"{name} (mean CHI={mean_chi:.3f})"
        ax1.plot(turns, chi, marker="o", linewidth=2, markersize=5, color=color_map.get(name), label=label)
        ax2.plot(turns, rlen, marker="s", linewidth=1.8, markersize=4, color=color_map.get(name), label=name)

    ax1.axhline(CHI_THRESHOLD, color="gray", linestyle="--", linewidth=1.3, label=f"CHI={CHI_THRESHOLD}")
    ax1.set_ylabel("CHI (smoothed)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("TASK-040 calibrated steering experiment — CHI and response length")
    ax1.legend(loc="best")

    ax2.set_xlabel("Turn")
    ax2.set_ylabel("Response length (tokens)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_png}")


def validate_vector_files(root: Path) -> None:
    missing: list[str] = []
    for vector_name in ["task_focus", "tool_format", "code_coherence"]:
        p = root / "data" / "steering_vectors" / f"{vector_name}.npz"
        if not p.is_file():
            missing.append(str(p))
    if missing:
        joined = "\n".join(missing)
        print(f"ERROR: Missing steering vectors:\n{joined}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="TASK-040 calibrated steering experiment")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    parser.add_argument("--layer", type=int, default=16, help="Fixed steering layer for all 3 vectors")
    parser.add_argument("--alpha", type=float, default=0.5, help="Fixed steering alpha for all 3 vectors")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--context-limit", type=int, default=28000)
    parser.add_argument("--condition", choices=["all"] + CONDITIONS, default="all")
    parser.add_argument("--cooldown-between-runs", type=int, default=120)
    parser.add_argument("--output-json", default="data/steering_experiment_v2.json")
    parser.add_argument("--output-png", default="data/steering_experiment_v2.png")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    # Ensure progress prints show up even when stdout is piped.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    out_json = _abs_path(root, args.output_json)
    out_png = _abs_path(root, args.output_png)

    if args.plot_only:
        if not out_json.is_file():
            print(f"ERROR: Cannot plot-only; JSON not found: {out_json}", file=sys.stderr)
            sys.exit(1)
        with open(out_json, "r") as f:
            payload = json.load(f)
        payload["summary"] = summarize_conditions(payload)
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        plot_payload(payload, out_png)
        print(json.dumps(payload["summary"], indent=2))
        return

    from cognition_engine.inference import load
    from cognition_engine.steering.injector import SteeringEngine

    print(f"Loading model: {args.model}", flush=True)
    model, tokenizer = load(args.model)
    print("Model loaded.", flush=True)

    if args.condition == "all":
        run_order = CONDITIONS
    else:
        run_order = [args.condition]

    if any(c != "baseline" for c in run_order):
        validate_vector_files(root)

    if out_json.is_file():
        with open(out_json, "r") as f:
            payload = json.load(f)
    else:
        payload = {
            "protocol": "TASK-040",
            "model": args.model,
            "turns": EXPERIMENT_TURNS,
            "layer": args.layer,
            "alpha": args.alpha,
            "vector_mode": "raw",
            "conditions": {},
        }

    payload["model"] = args.model
    payload["layer"] = args.layer
    payload["alpha"] = args.alpha
    payload["max_tokens"] = args.max_tokens
    payload["context_limit"] = args.context_limit
    payload["vector_mode"] = "raw"

    for idx, cond in enumerate(run_order):
        print(f"\n=== Running condition: {cond} ===", flush=True)
        if cond == "baseline":
            rows = run_condition(
                model,
                tokenizer,
                condition=cond,
                steering_engines=None,
                max_tokens=args.max_tokens,
                context_limit=args.context_limit,
            )
        else:
            engines: list[Any] = []
            try:
                if cond == "dual_vector":
                    print(f"[{cond}] Steering config: vectors=raw, dual-vector safe params", flush=True)
                    for vector_name, cfg in DUAL_VECTOR_CONFIG.items():
                        engine = SteeringEngine(model)
                        engines.append(engine)
                        print(
                            f"[{cond}] Loading {vector_name}: layer={cfg['layer']}, alpha={cfg['alpha']}",
                            flush=True,
                        )
                        engine.load_vectors(
                            vector_name,
                            layer_idx=cfg["layer"],
                            alpha=cfg["alpha"],
                            normalize=False,
                        )
                else:
                    engine = SteeringEngine(model)
                    engines.append(engine)
                    print(
                        f"[{cond}] Steering config: layer={args.layer}, alpha={args.alpha}, vectors=raw",
                        flush=True,
                    )
                    engine.load_vectors(cond, layer_idx=args.layer, alpha=args.alpha, normalize=False)

                rows = run_condition(
                    model,
                    tokenizer,
                    condition=cond,
                    steering_engines=engines,
                    max_tokens=args.max_tokens,
                    context_limit=args.context_limit,
                )
            finally:
                for engine in reversed(engines):
                    engine.cleanup()

        payload["conditions"][cond] = rows
        payload["summary"] = summarize_conditions(payload)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON: {out_json}", flush=True)

        is_last = idx == (len(run_order) - 1)
        if args.condition == "all" and not is_last and args.cooldown_between_runs > 0:
            print(f"Cooldown {args.cooldown_between_runs}s...", flush=True)
            time.sleep(args.cooldown_between_runs)

    plot_payload(payload, out_png)
    print("\n--- Summary ---")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

