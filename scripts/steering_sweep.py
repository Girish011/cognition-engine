"""
TASK-039: Systematic alpha/layer sweep for steering calibration.

Protocol:
1) Run + cache one unsteered baseline (fixed prompt/turns).
2) Sweep (layer, alpha) grid with steering enabled and measure:
   - response_alive: >10 tokens on all turns
   - mean_response_length: mean tokens per turn
   - mean_chi: mean CHI across turns
   - chi_drop: CHI(turn 0) - CHI(turn 4)
   - response_changed: any turn differs from cached baseline
3) Second pass: re-run the top 5 (layer, alpha) pairs using L2-normalized
   steering vectors (vector / ||vector||, then scaled by alpha in injection).

Outputs:
- data/steering_sweep_baseline.json
- data/steering_sweep.json
- data/steering_sweep.png
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SYSTEM_PROMPT = "You are a coding assistant. Follow instructions precisely."

SWEEP_LAYERS = [12, 14, 16, 17, 18, 20, 22, 24, 27]
SWEEP_ALPHAS = [0.25, 0.5, 1.0, 1.5, 2.0]

TURNS_5 = [
    "Create a Calculator class with add and subtract methods",
    "Add multiply and divide methods. Handle division by zero.",
    "Write unit tests for all four operations",
    "Add a history feature that stores the last 10 operations",
    "Refactor to use a Strategy pattern for the operations",
]


@dataclass(frozen=True)
class SweepKey:
    layer: int
    alpha: float


def _ensure_abs(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def run_5_turn_chat(
    model: Any,
    tokenizer: Any,
    *,
    steering_engine: Any | None,
    turns: list[str],
    max_tokens: int,
    context_limit: int,
    label: str,
) -> list[dict]:
    from cognition_engine.chi import CHIMonitor
    from cognition_engine.inference import generate

    monitor = CHIMonitor(model, tokenizer, original_task=turns[0])
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

        ctx_len = len(tokenizer.encode(full_text))
        resp_len = len(tokenizer.encode(response))
        rows.append(
            {
                "turn": i,
                "user_message": user_msg,
                "chi_score": chi_out["chi_score"],
                "chi_raw": chi_out["chi_raw"],
                "response_text": response,
                "response_length_tokens": resp_len,
                "context_length_tokens": ctx_len,
                "condition": label,
            }
        )

        print(f"[{label}] Turn {i} | CHI={chi_out['chi_score']:.4f} | resp={resp_len:4d} tok | ctx={ctx_len:5d} tok")

        conversation += response + "\n"
        if ctx_len > context_limit:
            print(f"[{label}] Stopping early: approaching context limit at turn {i}")
            break

    return rows


def compute_metrics(
    *,
    rows: list[dict],
    baseline_rows: list[dict],
) -> dict:
    chi = [r["chi_score"] for r in rows]
    resp_lens = [r["response_length_tokens"] for r in rows]
    response_alive = all(n > 10 for n in resp_lens) and len(resp_lens) == len(baseline_rows)

    mean_resp_len = (sum(resp_lens) / len(resp_lens)) if resp_lens else 0.0
    mean_chi = (sum(chi) / len(chi)) if chi else 0.0
    chi_drop = (chi[0] - chi[4]) if len(chi) >= 5 else None

    changed = False
    for r, b in zip(rows, baseline_rows, strict=False):
        if r.get("response_text", "") != b.get("response_text", ""):
            changed = True
            break

    return {
        "response_alive": response_alive,
        "mean_response_length": mean_resp_len,
        "mean_chi": mean_chi,
        "chi_drop": chi_drop,
        "response_changed": changed,
    }


def pick_top5(
    results: list[dict],
    *,
    baseline_mean_chi: float,
) -> list[dict]:
    eligible: list[dict] = []
    for r in results:
        m = r["metrics"]
        if not (m["response_alive"] and m["response_changed"]):
            continue
        if m["mean_response_length"] < 50:
            continue
        delta = m["mean_chi"] - baseline_mean_chi
        if delta <= 0:
            continue
        eligible.append({**r, "_delta_mean_chi": delta})

    eligible.sort(key=lambda x: x["_delta_mean_chi"], reverse=True)
    return eligible[:5]


def plot_heatmap(
    *,
    results: list[dict],
    layers: list[int],
    alphas: list[float],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    layer_to_x = {l: i for i, l in enumerate(layers)}
    alpha_to_y = {a: i for i, a in enumerate(alphas)}

    grid = np.full((len(alphas), len(layers)), np.nan, dtype=float)
    dead = np.zeros((len(alphas), len(layers)), dtype=bool)

    for r in results:
        x = layer_to_x[r["layer"]]
        y = alpha_to_y[r["alpha"]]
        grid[y, x] = r["metrics"]["mean_chi"]
        dead[y, x] = not r["metrics"]["response_alive"]

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_xticks(list(range(len(layers))), [str(l) for l in layers])
    ax.set_yticks(list(range(len(alphas))), [str(a) for a in alphas])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Alpha")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean_chi")

    ys, xs = np.where(dead)
    for y, x in zip(ys.tolist(), xs.tolist()):
        ax.text(x, y, "X", ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap: {output_path}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="TASK-039 steering sweep (layer × alpha)")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", help="MLX model id or path")
    parser.add_argument("--vector-name", default="task_focus", help="Steering vectors name (data/steering_vectors/<name>.npz)")
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--context-limit", type=int, default=28000)
    parser.add_argument("--layers", default=",".join(map(str, SWEEP_LAYERS)))
    parser.add_argument("--alphas", default=",".join(map(str, SWEEP_ALPHAS)))
    parser.add_argument("--output-json", default="data/steering_sweep.json")
    parser.add_argument("--output-png", default="data/steering_sweep.png")
    parser.add_argument("--baseline-json", default="data/steering_sweep_baseline.json")
    parser.add_argument("--recompute-baseline", action="store_true", help="Ignore cached baseline and rerun it")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    from cognition_engine.inference import load
    from cognition_engine.steering.injector import SteeringEngine

    model, tokenizer = load(args.model)

    baseline_path = _ensure_abs(root, args.baseline_json)
    baseline: list[dict]
    if baseline_path.exists() and not args.recompute_baseline:
        with open(baseline_path, "r") as f:
            baseline_payload = json.load(f)
        baseline = baseline_payload["rows"]
        print(f"Loaded cached baseline: {baseline_path}")
    else:
        print("=== Baseline (unsteered): running and caching ===")
        baseline = run_5_turn_chat(
            model,
            tokenizer,
            steering_engine=None,
            turns=TURNS_5,
            max_tokens=args.max_tokens,
            context_limit=args.context_limit,
            label="baseline_unsteered",
        )
        baseline_payload = {
            "model": args.model,
            "vector_name": args.vector_name,
            "turns": TURNS_5,
            "max_tokens": args.max_tokens,
            "context_limit": args.context_limit,
            "rows": baseline,
        }
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(baseline_payload, f, indent=2)
        print(f"Saved baseline cache: {baseline_path}")

    baseline_mean_chi = sum(r["chi_score"] for r in baseline) / len(baseline) if baseline else 0.0

    vec_path = root / "data" / "steering_vectors" / f"{args.vector_name}.npz"
    if not vec_path.is_file():
        print(
            f"ERROR: Steering vectors not found: {vec_path}\n"
            "Run TASK-035 and TASK-036 first.",
            file=sys.stderr,
        )
        sys.exit(1)

    results: list[dict] = []
    print("\n=== Sweep pass (raw vectors) ===")
    for layer in layers:
        for alpha in alphas:
            key = SweepKey(layer=layer, alpha=alpha)
            print(f"\n--- Sweep: layer={key.layer}, alpha={key.alpha} ---")
            engine = SteeringEngine(model)
            try:
                engine.load_vectors(args.vector_name, layer_idx=layer, alpha=alpha, normalize=False)
                rows = run_5_turn_chat(
                    model,
                    tokenizer,
                    steering_engine=engine,
                    turns=TURNS_5,
                    max_tokens=args.max_tokens,
                    context_limit=args.context_limit,
                    label=f"steered_l{layer}_a{alpha}",
                )
            finally:
                engine.cleanup()

            metrics = compute_metrics(rows=rows, baseline_rows=baseline)
            results.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "normalize": False,
                    "metrics": metrics,
                }
            )

    top5 = pick_top5(results, baseline_mean_chi=baseline_mean_chi)
    normalized_followup: list[dict] = []

    print("\n=== Follow-up pass (top 5, normalized vectors) ===")
    for r in top5:
        layer = int(r["layer"])
        alpha = float(r["alpha"])
        print(f"\n--- Normalized follow-up: layer={layer}, alpha={alpha} ---")
        engine = SteeringEngine(model)
        try:
            engine.load_vectors(args.vector_name, layer_idx=layer, alpha=alpha, normalize=True)
            rows = run_5_turn_chat(
                model,
                tokenizer,
                steering_engine=engine,
                turns=TURNS_5,
                max_tokens=args.max_tokens,
                context_limit=args.context_limit,
                label=f"steered_norm_l{layer}_a{alpha}",
            )
        finally:
            engine.cleanup()

        metrics = compute_metrics(rows=rows, baseline_rows=baseline)
        normalized_followup.append(
            {
                "layer": layer,
                "alpha": alpha,
                "normalize": True,
                "metrics": metrics,
            }
        )

    out_json = _ensure_abs(root, args.output_json)
    out_png = _ensure_abs(root, args.output_png)
    payload = {
        "protocol": "TASK-039",
        "model": args.model,
        "vector_name": args.vector_name,
        "turns": TURNS_5,
        "max_tokens": args.max_tokens,
        "context_limit": args.context_limit,
        "baseline_cache": str(baseline_path),
        "baseline_mean_chi": baseline_mean_chi,
        "sweep_grid": {"layers": layers, "alphas": alphas},
        "results": results,
        "top5_raw": [
            {
                "layer": int(r["layer"]),
                "alpha": float(r["alpha"]),
                "mean_chi_delta_vs_baseline": float(r["_delta_mean_chi"]),
                "metrics": r["metrics"],
            }
            for r in top5
        ],
        "normalized_followup": normalized_followup,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved sweep JSON: {out_json}")

    plot_heatmap(
        results=results,
        layers=layers,
        alphas=alphas,
        output_path=out_png,
        title="Steering sweep (raw vectors) — mean CHI; X=dead (response_alive=False)",
    )


if __name__ == "__main__":
    main()

