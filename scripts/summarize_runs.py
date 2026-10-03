"""Summarise recorded agent runs: python scripts/summarize_runs.py data/long_*.json

Prints a markdown table with completion, CHI range (flags a "pinned" CHI trace narrower
than 0.15, which cannot drive the intervention ladder), levels reached, and interventions fired.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognition_engine.analysis import format_table, load_runs  # noqa: E402


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    print(format_table(load_runs(paths)))


if __name__ == "__main__":
    main()
