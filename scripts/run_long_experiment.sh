#!/usr/bin/env bash
# Long-task steering policy comparison (Apple Silicon + model required).
#
# Runs the 6-step library-management task under four policies:
#   baseline         no steering, no compaction
#   steering         CHI-triggered steering (on only while CHI is low)
#   steering_always  steering on from turn 0 (control)
#   full_ladder      CHI-triggered steering + compaction
#
# Generation is deterministic at temperature 0, so each policy is one run per task.
# Results land in data/long_<mode>.json; summarise with scripts/summarize_runs.py.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
MAX_TURNS="${MAX_TURNS:-25}"

for mode in baseline steering steering_always full_ladder; do
  echo "=== ${mode} ==="
  "$PY" -u scripts/test_steering_agent.py \
    --mode "$mode" --preset long --dir "/tmp/cognition_long_${mode}" \
    --max-turns "$MAX_TURNS" --auto-approve \
    --output-json "data/long_${mode}.json"
done

"$PY" scripts/summarize_runs.py data/long_*.json
