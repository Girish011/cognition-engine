"""
TASK-044: End-to-end integration test — full intervention ladder (A/B/C).

Modes:
  baseline     — CHI only; no steering; no compaction
  steering     — CHI + Level 2 steering (adaptive alpha); no compaction
  full_ladder  — CHI + steering + Level 3 compaction + Level 4 log line

Outputs:
  data/task044_{mode}.json
  --plot-only (no --dir): data/task044_comparison.png + data/task044_report.md
  --plot-comparison without --dir: same as --plot-only
  --plot-comparison with --dir: run agent, then rebuild comparison artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TASK044_TASK = (
    "Create a Python file called calculator.py with a Calculator class "
    "that has add, subtract, multiply, and divide methods. "
    "Then create test_calculator.py with tests for each method. "
    "Run the tests to verify they pass."
)

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

MODE_TO_JSON = {
    "baseline": "data/task044_baseline.json",
    "steering": "data/task044_steering.json",
    "full_ladder": "data/task044_full_ladder.json",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _abs(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def _pytest_ran(tool_history: list[dict[str, Any]]) -> bool:
    for call in tool_history:
        if call.get("tool") != "execute":
            continue
        cmd = str(call.get("command", "")).lower()
        if "pytest" in cmd:
            return True
    return False


def _max_think_streak(turns: list[dict[str, Any]]) -> int:
    best = cur = 0
    for row in turns:
        if row.get("tool_used") == "think":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _steering_activation_count(turns: list[dict[str, Any]]) -> int:
    n = 0
    prev = False
    for row in turns:
        active = bool(row.get("steering_active"))
        if active and not prev:
            n += 1
        prev = active
    return n


def _action_tool_calls(tool_history: list[dict[str, Any]]) -> int:
    return sum(
        1
        for c in tool_history
        if c.get("tool") in {"read", "write", "edit", "execute"}
    )


def compute_metrics(
    *,
    turns: list[dict[str, Any]],
    tool_history: list[dict[str, Any]],
    working_dir: str,
    max_turns: int,
    compaction_count: int,
    chi_min: float | None,
    chi_final: float | None,
) -> dict[str, Any]:
    wd = Path(working_dir)
    has_calc = (wd / "calculator.py").is_file()
    has_test = (wd / "test_calculator.py").is_file()
    pytest_ran = _pytest_ran(tool_history)
    reached_done = any(row.get("tool_used") == "done" for row in turns)

    # Heuristic completion: files exist + successful test execution (even if model never emits [DONE]).
    heuristic_completed = False
    turns_to_completion = None
    if has_calc and has_test:
        for row in turns:
            if row.get("tool_used") != "execute":
                continue
            cmd = str(row.get("execute_command") or "")
            code = row.get("execute_exit_code")
            if code == 0 and ("pytest" in cmd.lower() or "unittest" in cmd.lower()):
                heuristic_completed = True
                turns_to_completion = int(row["turn"]) + 1
                break

    turns_to_done = None
    for row in turns:
        if row.get("tool_used") == "done":
            turns_to_done = int(row["turn"]) + 1
            break

    turns_to_any_completion = turns_to_done if turns_to_done is not None else turns_to_completion

    minimum_bar = bool(has_calc and has_test and pytest_ran and (reached_done or heuristic_completed))

    return {
        "task_completed_minimum_bar": minimum_bar,
        "calculator_py_exists": has_calc,
        "test_calculator_py_exists": has_test,
        "pytest_executed": pytest_ran,
        "reached_explicit_done": reached_done,
        "reached_heuristic_completion": heuristic_completed,
        "turns_to_completion_or_done": turns_to_any_completion,
        "turns_recorded": len(turns),
        "max_turns_budget": max_turns,
        "total_action_tool_calls": _action_tool_calls(tool_history),
        "chi_min": chi_min,
        "chi_final": chi_final,
        "max_consecutive_think_only_turns": _max_think_streak(turns),
        "steering_activation_count": _steering_activation_count(turns),
        "compaction_firings_total": compaction_count,
        "compaction_events_in_turn_log": sum(
            1 for t in turns if t.get("compaction_fired")
        ),
    }


def run_mode(
    *,
    mode: str,
    working_dir: str,
    max_turns: int,
    model_id: str,
    output_json: Path,
    auto_approve: bool,
) -> dict[str, Any]:
    root = _repo_root()
    wd_path = Path(working_dir)
    if wd_path.exists():
        shutil.rmtree(wd_path)
    wd_path.mkdir(parents=True, exist_ok=True)

    from mlx_lm import load

    from cognition_engine.agent import CognitionAgent

    print(f"Loading model: {model_id}", flush=True)
    model, tokenizer = load(model_id)
    print("Model loaded.", flush=True)

    enable_steering = mode in ("steering", "full_ladder")
    enable_compaction = mode == "full_ladder"
    turn_log: list[dict[str, Any]] = []

    print(
        f"Mode={mode} | steering={enable_steering} | compaction={enable_compaction}",
        flush=True,
    )

    # TASK-045F: calculator harness expects pytest; avoid confounding the agent with pip.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "pytest"],
        capture_output=True,
        check=False,
    )

    agent = CognitionAgent(
        model,
        tokenizer,
        TASK044_TASK,
        working_dir=str(wd_path),
        auto_approve=auto_approve,
        enable_steering=enable_steering,
        enable_compaction=enable_compaction,
        turn_log=turn_log,
    )
    agent.run(max_turns=max_turns)

    chi_min = min(agent.chi_monitor.chi_history) if agent.chi_monitor.chi_history else None
    chi_final = (
        agent.chi_monitor.chi_history[-1] if agent.chi_monitor.chi_history else None
    )

    metrics = compute_metrics(
        turns=turn_log,
        tool_history=agent.tool_history,
        working_dir=str(wd_path),
        max_turns=max_turns,
        compaction_count=agent.compactor.compaction_count,
        chi_min=chi_min,
        chi_final=chi_final,
    )

    payload = {
        "protocol": "TASK-044",
        "mode": mode,
        "model": model_id,
        "working_dir": str(wd_path.resolve()),
        "task": TASK044_TASK,
        "max_turns": max_turns,
        "enable_steering": enable_steering,
        "enable_compaction": enable_compaction,
        "turns": turn_log,
        "metrics": metrics,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {output_json}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    return payload


def plot_comparison(
    baseline_path: Path,
    steering_path: Path,
    full_path: Path,
    output_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: list[tuple[str, str, str]] = [
        ("baseline", "black", baseline_path),
        ("steering", "tab:green", steering_path),
        ("full_ladder", "tab:purple", full_path),
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    for label, color, path in series:
        if not path.is_file():
            continue
        with open(path) as f:
            data = json.load(f)
        turns = data.get("turns") or []
        if not turns:
            continue
        xs = [t["turn"] for t in turns]
        ys = [t["chi_score"] for t in turns]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, color=color, label=label)

    ax.axhline(0.7, color="gray", linestyle="--", linewidth=1, label="CHI 0.7")
    ax.axhline(0.5, color="orange", linestyle="--", linewidth=1, label="CHI 0.5")
    ax.set_xlabel("Turn")
    ax.set_ylabel("CHI (smoothed)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title("TASK-044 — CHI trajectories (baseline vs steering vs full ladder)")
    ax.legend(loc="best")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_png}", flush=True)


def write_report_md(
    paths: dict[str, Path],
    output_md: Path,
) -> None:
    sections: list[str] = ["# TASK-044 report\n", "\n## Run metrics\n\n"]
    for mode, path in paths.items():
        sections.append(f"### {mode}\n\n")
        if not path.is_file():
            sections.append(f"_Missing file: `{path}`_\n\n")
            continue
        with open(path) as f:
            data = json.load(f)
        m = data.get("metrics") or {}
        sections.append("```json\n")
        sections.append(json.dumps(m, indent=2))
        sections.append("\n```\n\n")

    sections.append("## Interpretation notes\n\n")
    sections.append(
        "- **If Run A and Run B fail but Run C succeeds** → compaction likely broke a "
        "degenerate context loop.\n"
        "- **If Run B succeeds but Run A fails** → steering helped.\n"
        "- **If all succeed** → task may be too easy for this comparison; try a harder task later.\n"
        "- **If all fail** → model/prompt variance dominates; rerun or tune prompts.\n\n"
    )
    sections.append(
        "_Fill in a short narrative after each manual run (interesting turns, "
        "compaction events, [THINK] streaks)._ \n"
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("".join(sections), encoding="utf-8")
    print(f"Saved report: {output_md}", flush=True)


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="TASK-044 steering + ladder integration test")
    parser.add_argument(
        "--mode",
        choices=["baseline", "steering", "full_ladder"],
        default="baseline",
        help="Intervention configuration for this run",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Clean working directory for this run (required unless --plot-only)",
    )
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Override output path (default: data/task044_{mode}.json)",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve [EXEC] (required for unattended pytest runs)",
    )
    parser.add_argument(
        "--plot-comparison",
        action="store_true",
        help=(
            "With --dir: after this run, rebuild comparison PNG + report. "
            "Without --dir: only rebuild from existing data/task044_*.json (same as --plot-only)."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only rebuild data/task044_comparison.png and data/task044_report.md from saved JSON (no model run)",
    )
    args = parser.parse_args()

    def _regenerate_plots_and_report() -> None:
        data_dir = root / "data"
        plot_comparison(
            data_dir / "task044_baseline.json",
            data_dir / "task044_steering.json",
            data_dir / "task044_full_ladder.json",
            data_dir / "task044_comparison.png",
        )
        write_report_md(
            {
                "baseline": data_dir / "task044_baseline.json",
                "steering": data_dir / "task044_steering.json",
                "full_ladder": data_dir / "task044_full_ladder.json",
            },
            data_dir / "task044_report.md",
        )

    if args.plot_only:
        _regenerate_plots_and_report()
        return

    if args.plot_comparison and args.dir is None:
        print(
            "No --dir: skipping agent run; rebuilding comparison from data/task044_*.json",
            flush=True,
        )
        _regenerate_plots_and_report()
        return

    if args.dir is None:
        parser.error(
            "--dir is required for agent runs. "
            "To rebuild plots from data/task044_*.json without loading the model, use "
            "--plot-only (or --plot-comparison with no --dir)."
        )

    out = (
        _abs(root, args.output_json)
        if args.output_json
        else _abs(root, MODE_TO_JSON[args.mode])
    )

    run_mode(
        mode=args.mode,
        working_dir=args.dir,
        max_turns=args.max_turns,
        model_id=args.model,
        output_json=out,
        auto_approve=args.auto_approve,
    )

    if args.plot_comparison:
        data_dir = root / "data"
        plot_comparison(
            data_dir / "task044_baseline.json",
            data_dir / "task044_steering.json",
            data_dir / "task044_full_ladder.json",
            data_dir / "task044_comparison.png",
        )
        write_report_md(
            {
                "baseline": data_dir / "task044_baseline.json",
                "steering": data_dir / "task044_steering.json",
                "full_ladder": data_dir / "task044_full_ladder.json",
            },
            data_dir / "task044_report.md",
        )


if __name__ == "__main__":
    main()
