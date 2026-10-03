"""Run-summary tests."""

from cognition_engine.analysis import (
    chi_range,
    format_table,
    summarize_run,
)


def run_with(chis, **flags):
    turns = []
    for i, c in enumerate(chis):
        t = {"turn": i, "chi_score": c, "intervention_level": 1 if c > 0.8 else 2}
        t.update({k: v for k, v in flags.items()})
        turns.append(t)
    return {"mode": "steering", "turns": turns, "metrics": {"task_completed_minimum_bar": True}}


def test_range_is_none_for_short_runs():
    assert chi_range(run_with([0.85, 0.85, 0.9])) is None


def test_pinned_trace_is_flagged():
    s = summarize_run(run_with([0.85, 0.85, 0.69, 0.70, 0.69, 0.70, 0.69]))
    assert s["chi_pinned"] is True


def test_wide_trace_is_not_pinned():
    s = summarize_run(run_with([0.85, 0.85, 0.95, 0.80, 0.60, 0.55, 0.9]))
    assert s["chi_pinned"] is False
    assert s["chi_range"] > 0.15
    assert s["levels"] == [1, 2]


def test_table_renders_one_row_per_run():
    rows = [summarize_run(run_with([0.9] * 8), "a"), summarize_run(run_with([0.9] * 8), "b")]
    assert len(format_table(rows).splitlines()) == 4
