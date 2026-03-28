# TASK-044 report

## Run metrics

### baseline

```json
{
  "task_completed_minimum_bar": false,
  "calculator_py_exists": true,
  "test_calculator_py_exists": true,
  "pytest_executed": true,
  "reached_explicit_done": false,
  "turns_to_completion_or_done": null,
  "turns_recorded": 20,
  "max_turns_budget": 20,
  "total_action_tool_calls": 12,
  "chi_min": 0.6324898316434227,
  "chi_final": 0.6937648905521521,
  "max_consecutive_think_only_turns": 2,
  "steering_activation_count": 0,
  "compaction_firings_total": 0,
  "compaction_events_in_turn_log": 0
}
```

### steering

```json
{
  "task_completed_minimum_bar": false,
  "calculator_py_exists": true,
  "test_calculator_py_exists": true,
  "pytest_executed": true,
  "reached_explicit_done": false,
  "turns_to_completion_or_done": null,
  "turns_recorded": 20,
  "max_turns_budget": 20,
  "total_action_tool_calls": 13,
  "chi_min": 0.5998765908296799,
  "chi_final": 0.7428903444539591,
  "max_consecutive_think_only_turns": 2,
  "steering_activation_count": 1,
  "compaction_firings_total": 0,
  "compaction_events_in_turn_log": 0
}
```

### full_ladder

```json
{
  "task_completed_minimum_bar": false,
  "calculator_py_exists": true,
  "test_calculator_py_exists": true,
  "pytest_executed": true,
  "reached_explicit_done": false,
  "turns_to_completion_or_done": null,
  "turns_recorded": 20,
  "max_turns_budget": 20,
  "total_action_tool_calls": 13,
  "chi_min": 0.5998765908296799,
  "chi_final": 0.7428903444539591,
  "max_consecutive_think_only_turns": 2,
  "steering_activation_count": 1,
  "compaction_firings_total": 0,
  "compaction_events_in_turn_log": 0
}
```

## Interpretation notes

- **If Run A and Run B fail but Run C succeeds** → compaction likely broke a degenerate context loop.
- **If Run B succeeds but Run A fails** → steering helped.
- **If all succeed** → task may be too easy for this comparison; try a harder task later.
- **If all fail** → model/prompt variance dominates; rerun or tune prompts.

_Fill in a short narrative after each manual run (interesting turns, compaction events, [THINK] streaks)._ 
