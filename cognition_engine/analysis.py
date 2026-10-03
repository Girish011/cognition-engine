"""Summaries of recorded agent runs (the JSON written by scripts/test_steering_agent.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# A CHI trace that spans less than this is "pinned": it cannot drive the ladder.
CHI_MIN_USEFUL_RANGE = 0.15
# Fewer post-cold-start points than this and the range says nothing about pinning.
MIN_POINTS_FOR_RANGE = 4


def chi_trace(run: dict[str, Any]) -> list[float]:
    return [float(t["chi_score"]) for t in run.get("turns", []) if "chi_score" in t]


def chi_range(run: dict[str, Any]) -> float | None:
    """Width of the post-cold-start CHI trace (first two turns are neutral by design).

    None when the run is too short to say anything.
    """
    trace = chi_trace(run)[2:]
    if len(trace) < MIN_POINTS_FOR_RANGE:
        return None
    return max(trace) - min(trace)


def levels_reached(run: dict[str, Any]) -> list[int]:
    return sorted({int(t["intervention_level"]) for t in run.get("turns", []) if "intervention_level" in t})


def summarize_run(run: dict[str, Any], name: str = "") -> dict[str, Any]:
    turns = run.get("turns", [])
    metrics = run.get("metrics", {})
    trace = chi_trace(run)
    rng = chi_range(run)
    return {
        "run": name or run.get("mode", "?"),
        "mode": run.get("mode"),
        "completed": bool(metrics.get("task_completed_minimum_bar")),
        "turns": len(turns),
        "chi_min": min(trace) if trace else None,
        "chi_final": trace[-1] if trace else None,
        "chi_range": rng,
        "chi_pinned": rng is not None and rng < CHI_MIN_USEFUL_RANGE,
        "levels": levels_reached(run),
        "steering_turns": sum(1 for t in turns if t.get("steering_active")),
        "compactions": sum(1 for t in turns if t.get("compaction_fired")),
        "escalations": sum(1 for t in turns if t.get("escalation_fired")),
    }


def load_runs(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for p in paths:
        path = Path(p)
        rows.append(summarize_run(json.loads(path.read_text()), path.stem))
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    def f(x: Any) -> str:
        return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

    head = "| run | done | turns | CHI min | CHI final | CHI range | levels | steer turns | compactions | escalations |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| {r['run']} | {'yes' if r['completed'] else 'no'} | {r['turns']} | {f(r['chi_min'])} "
            f"| {f(r['chi_final'])} | {f(r['chi_range'])}{' (pinned)' if r['chi_pinned'] else ''} "
            f"| {','.join(map(str, r['levels'])) or '-'} | {r['steering_turns']} "
            f"| {r['compactions']} | {r['escalations']} |"
        )
    return "\n".join(lines)
