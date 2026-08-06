#!/usr/bin/env bash
# TASK-047D: n=5 replication of TASK-047C steering effect
# Expense tracker task, 15 turns max, baseline vs steering
set -euo pipefail
cd "$(dirname "$0")/.."

EXPENSE_TASK="Build a command-line expense tracker. Step 1: Create models.py with an Expense dataclass (amount: float, category: str, date: str, description: str) and an ExpenseTracker class with methods: add_expense(expense), get_expenses_by_category(category), get_total(), get_monthly_summary(month_str). ExpenseTracker should store expenses in a list and save/load from expenses.json. Step 2: Create cli.py using argparse with subcommands: add --amount --category --description, list --category (optional filter), summary --month. cli.py should import and use ExpenseTracker from models.py. Step 3: Create test_models.py with at least 6 tests covering add_expense, get_expenses_by_category, get_total, get_monthly_summary, empty tracker edge case, and save/load round-trip. Step 4: Run the tests with pytest to verify they pass."

echo "=== TASK-047D REPLICATION START: $(date) ==="

echo ""
echo "=== BASELINE RUNS (5x) ==="
for i in 1 2 3 4 5; do
  echo ""
  echo "--- baseline run $i --- $(date)"
  caffeinate -dimsu .venv/bin/python -u scripts/test_steering_agent.py \
    --mode baseline --dir /tmp/task047d_base_$i \
    --max-turns 15 --task "$EXPENSE_TASK" --auto-approve \
    --output-json "data/task047d_base_$i.json" 2>&1 | tee "data/task047d_base_$i.log"
  echo "--- baseline run $i done --- $(date)"
  sleep 60
done

echo ""
echo "=== STEERING RUNS (5x) ==="
for i in 1 2 3 4 5; do
  echo ""
  echo "--- steering run $i --- $(date)"
  caffeinate -dimsu .venv/bin/python -u scripts/test_steering_agent.py \
    --mode steering --dir /tmp/task047d_steer_$i \
    --max-turns 15 --task "$EXPENSE_TASK" --auto-approve \
    --output-json "data/task047d_steer_$i.json" 2>&1 | tee "data/task047d_steer_$i.log"
  echo "--- steering run $i done --- $(date)"
  sleep 60
done

echo ""
echo "=== TASK-047D REPLICATION COMPLETE: $(date) ==="
