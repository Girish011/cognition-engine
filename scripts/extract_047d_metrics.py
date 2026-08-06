"""TASK-047D: Extract per-run metrics from replication experiment."""

import json
import os
import re
import sys

def check_code_patterns(working_dir):
    """Check models.py for auto-load bug vs explicit save/load pattern."""
    models_path = os.path.join(working_dir, "models.py")
    test_path = os.path.join(working_dir, "test_models.py")
    result = {
        "auto_load_in_init": None,
        "explicit_save_load": None,
        "test_uses_autosave": None,
    }
    if os.path.exists(models_path):
        with open(models_path) as f:
            content = f.read()
        # Auto-load in __init__: calls load/read in constructor
        result["auto_load_in_init"] = bool(
            re.search(r"def __init__.*?self\.(load|_load|read)", content, re.DOTALL)
        )
        # Explicit save/load: separate methods with filename parameter
        has_save = bool(re.search(r"def save\w*\(", content))
        has_load = bool(re.search(r"def load\w*\(", content))
        result["explicit_save_load"] = has_save and has_load
    if os.path.exists(test_path):
        with open(test_path) as f:
            test_content = f.read()
        # Test assumes auto-save: instantiates tracker and immediately checks file exists
        result["test_uses_autosave"] = bool(
            re.search(r"ExpenseTracker\(\).*?exists|assert.*json.*exists", test_content, re.DOTALL)
        )
    return result


def extract_pytest_info(turns):
    """Find first and second pytest runs, return (turn_num, exit_code) tuples."""
    pytest_runs = []
    for row in turns:
        if row.get("tool_used") != "execute":
            continue
        cmd = str(row.get("execute_command") or "")
        if "pytest" in cmd.lower() or "test" in cmd.lower():
            pytest_runs.append({
                "turn": row.get("turn"),
                "exit_code": row.get("execute_exit_code"),
                "command": cmd[:80],
            })
    return pytest_runs


def max_think_streak(turns):
    best = cur = 0
    for row in turns:
        if row.get("tool_used") == "think":
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def guard_fire_count(turns):
    """Count turns where the think-loop guard fired."""
    return sum(1 for t in turns if t.get("guard_fired") or t.get("think_guard_fired"))


def steering_active_turns(turns):
    return [t["turn"] for t in turns if t.get("steering_active")]


def extract_run(mode_short, run_num):
    """Extract metrics for one run."""
    json_path = f"data/task047d_{mode_short}_{run_num}.json"
    working_dir = f"/tmp/task047d_{mode_short}_{run_num}"

    run = {
        "mode": mode_short,
        "run": run_num,
        "json_exists": os.path.exists(json_path),
        "dir_exists": os.path.exists(working_dir),
    }

    if not os.path.exists(json_path):
        return run

    with open(json_path) as f:
        data = json.load(f)

    turns = data.get("turns", [])
    metrics = data.get("metrics", {})

    run["completed"] = metrics.get("task_completed_minimum_bar", False)
    run["reached_done"] = metrics.get("reached_explicit_done", False)
    run["heuristic_completion"] = metrics.get("reached_heuristic_completion", False)
    run["total_turns"] = metrics.get("turns_recorded", len(turns))
    run["turns_to_completion"] = metrics.get("turns_to_completion_or_done")
    run["chi_min"] = metrics.get("chi_min")
    run["chi_final"] = metrics.get("chi_final")
    run["max_think_streak"] = max_think_streak(turns)
    run["guard_fires"] = guard_fire_count(turns)
    run["pytest_runs"] = extract_pytest_info(turns)

    if mode_short == "steer":
        run["steering_active_turns"] = steering_active_turns(turns)

    # Code pattern analysis
    code = check_code_patterns(working_dir)
    run.update(code)

    return run


def main():
    print("=" * 70)
    print("TASK-047D REPLICATION RESULTS")
    print("=" * 70)

    all_runs = []
    for mode in ["base", "steer"]:
        for i in range(1, 6):
            run = extract_run(mode, i)
            all_runs.append(run)

    # Summary table
    print(f"\n{'Mode':<8} {'Run':<4} {'Done':<5} {'Turns':<6} {'CHI_min':<9} {'CHI_fin':<9} "
          f"{'AutoLoad':<10} {'ExplSave':<10} {'ThinkMax':<10} {'Guards':<7}")
    print("-" * 90)
    for r in all_runs:
        if not r.get("json_exists"):
            print(f"{r['mode']:<8} {r['run']:<4} {'N/A - no JSON'}")
            continue
        print(
            f"{r['mode']:<8} {r['run']:<4} "
            f"{'Y' if r.get('completed') else 'N':<5} "
            f"{r.get('total_turns', '?'):<6} "
            f"{r.get('chi_min', 0):<9.4f} "
            f"{r.get('chi_final', 0):<9.4f} "
            f"{'Y' if r.get('auto_load_in_init') else 'N':<10} "
            f"{'Y' if r.get('explicit_save_load') else 'N':<10} "
            f"{r.get('max_think_streak', '?'):<10} "
            f"{r.get('guard_fires', '?'):<7}"
        )

    # Aggregate stats
    print("\n" + "=" * 70)
    print("AGGREGATE")
    print("=" * 70)
    for mode in ["base", "steer"]:
        mode_runs = [r for r in all_runs if r["mode"] == mode and r.get("json_exists")]
        if not mode_runs:
            print(f"\n{mode}: no data")
            continue
        n = len(mode_runs)
        completed = sum(1 for r in mode_runs if r.get("completed"))
        autoload = sum(1 for r in mode_runs if r.get("auto_load_in_init"))
        explicit = sum(1 for r in mode_runs if r.get("explicit_save_load"))
        chi_mins = [r["chi_min"] for r in mode_runs if r.get("chi_min") is not None]
        chi_finals = [r["chi_final"] for r in mode_runs if r.get("chi_final") is not None]
        print(f"\n{mode.upper()} (n={n}):")
        print(f"  Completion rate: {completed}/{n}")
        print(f"  Auto-load bug:  {autoload}/{n}")
        print(f"  Explicit save:  {explicit}/{n}")
        if chi_mins:
            print(f"  CHI min range:  {min(chi_mins):.4f} - {max(chi_mins):.4f} (mean {sum(chi_mins)/len(chi_mins):.4f})")
        if chi_finals:
            print(f"  CHI final range: {min(chi_finals):.4f} - {max(chi_finals):.4f} (mean {sum(chi_finals)/len(chi_finals):.4f})")

    # Detailed pytest info
    print("\n" + "=" * 70)
    print("PYTEST DETAILS")
    print("=" * 70)
    for r in all_runs:
        if not r.get("json_exists"):
            continue
        pytest_runs = r.get("pytest_runs", [])
        status = "completed" if r.get("completed") else "incomplete"
        print(f"\n{r['mode']} run {r['run']} ({status}):")
        if pytest_runs:
            for p in pytest_runs:
                print(f"  Turn {p['turn']}: exit={p['exit_code']}  cmd={p['command']}")
        else:
            print("  No pytest runs detected")

    # Raw JSON dump for full details
    print("\n" + "=" * 70)
    print("RAW DATA")
    print("=" * 70)
    print(json.dumps(all_runs, indent=2, default=str))


if __name__ == "__main__":
    main()
