#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

EXPENSE_TASK="Build a command-line expense tracker. Step 1: Create models.py with an Expense dataclass (amount: float, category: str, date: str, description: str) and an ExpenseTracker class with methods: add_expense(expense), get_expenses_by_category(category), get_total(), get_monthly_summary(month_str). ExpenseTracker should store expenses in a list and save/load from expenses.json. Step 2: Create cli.py using argparse with subcommands: add --amount --category --description, list --category (optional filter), summary --month. cli.py should import and use ExpenseTracker from models.py. Step 3: Create test_models.py with at least 6 tests covering add_expense, get_expenses_by_category, get_total, get_monthly_summary, empty tracker edge case, and save/load round-trip. Step 4: Run the tests with pytest to verify they pass."

echo "=== BASELINE (with think-loop guard, no steering, no compaction) ==="
caffeinate -dimsu .venv/bin/python -u scripts/test_steering_agent.py \
  --mode baseline --dir /tmp/task047c_exp_base \
  --max-turns 25 --task "$EXPENSE_TASK" --auto-approve \
  --output-json data/task047c_base.json 2>&1 | tee data/task047c_base.log

sleep 120

echo "=== STEERING (think-loop guard + tool_format L16/α0.5 adaptive) ==="
caffeinate -dimsu .venv/bin/python -u scripts/test_steering_agent.py \
  --mode steering --dir /tmp/task047c_exp_steer \
  --max-turns 25 --task "$EXPENSE_TASK" --auto-approve \
  --output-json data/task047c_steer.json 2>&1 | tee data/task047c_steer.log

sleep 120

echo "=== FULL LADDER (think-loop guard + steering + compaction) ==="
caffeinate -dimsu .venv/bin/python -u scripts/test_steering_agent.py \
  --mode full_ladder --dir /tmp/task047c_exp_full \
  --max-turns 25 --task "$EXPENSE_TASK" --auto-approve \
  --output-json data/task047c_full.json 2>&1 | tee data/task047c_full.log

echo "=== ALL 3 RUNS COMPLETE ==="
